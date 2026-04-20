"""Tests for mr1.mr1 — orchestrator logic (no subprocess spawning)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
from mr1.mr1 import (
    MR1,
    MR1Process,
    StateManager,
    _load_agent_config,
    _generate_task_id,
)


class TestStateManager:
    @pytest.fixture
    def state_mgr(self, tmp_path):
        state_path = tmp_path / "mr1_state.json"
        return StateManager(state_path=state_path)

    def test_init_creates_state(self, state_mgr):
        assert state_mgr.session_id is not None
        assert len(state_mgr.active_tasks) == 0

    def test_add_and_complete_task(self, state_mgr):
        state_mgr.add_task("t1", "kazi", "test task", 123)
        assert "t1" in state_mgr.active_tasks

        state_mgr.complete_task("t1", "completed")
        assert "t1" not in state_mgr.active_tasks

    def test_add_decision(self, state_mgr):
        state_mgr.add_decision("user said hi", "direct_answer")
        status = state_mgr.format_status()
        assert "direct_answer" in status

    def test_decisions_rolling_window(self, state_mgr):
        for i in range(60):
            state_mgr.add_decision(f"input {i}", f"action_{i}")
        # Should be capped.
        assert "action_59" in state_mgr.format_status()

    def test_save_and_reload(self, tmp_path):
        state_path = tmp_path / "mr1_state.json"
        mgr1 = StateManager(state_path=state_path)
        mgr1.add_task("t1", "kazi", "task one", 100)
        mgr1.save()

        mgr2 = StateManager(state_path=state_path)
        assert "t1" in mgr2.active_tasks

    def test_format_tasks(self, state_mgr):
        assert state_mgr.format_tasks() == "No tasks."
        state_mgr.add_task("t1", "kazi", "test", 100)
        formatted = state_mgr.format_tasks()
        assert "t1" in formatted
        assert "kazi" in formatted

    def test_format_for_prompt_empty(self, state_mgr):
        assert state_mgr.format_for_prompt() == "No active tasks."

    def test_format_for_prompt_with_tasks(self, state_mgr):
        state_mgr.add_task("t1", "mr2", "complex task", 200)
        prompt = state_mgr.format_for_prompt()
        assert "t1" in prompt
        assert "mr2" in prompt

    def test_agent_pids(self, state_mgr):
        state_mgr.add_agent_pid(100)
        state_mgr.add_agent_pid(200)
        state_mgr.add_agent_pid(100)  # Duplicate — should not add.
        assert state_mgr._state["agent_pids"] == [100, 200]

        state_mgr.remove_agent_pid(100)
        assert state_mgr._state["agent_pids"] == [200]


class TestParseResponse:
    def test_plain_text(self):
        text, directive = MR1._parse_response("Just a normal answer.")
        assert text == "Just a normal answer."
        assert directive is None

    def test_delegation_directive_kazi(self):
        raw = 'I will handle this. [DELEGATE]{"agent": "kazi", "task": "read file", "context": "none"}[/DELEGATE]'
        text, directive = MR1._parse_response(raw)
        assert text == "I will handle this."
        assert directive["agent"] == "kazi"
        assert directive["task"] == "read file"

    def test_delegation_directive_mr2(self):
        raw = 'Working on it. [DELEGATE]{"agent": "mr2", "task": "refactor code", "context": "src/"}[/DELEGATE]'
        text, directive = MR1._parse_response(raw)
        assert text == "Working on it."
        assert directive["agent"] == "mr2"

    def test_invalid_json_in_directive(self):
        raw = "text [DELEGATE]{broken json[/DELEGATE]"
        text, directive = MR1._parse_response(raw)
        assert directive is None

    def test_missing_required_fields(self):
        raw = '[DELEGATE]{"task": "no agent field"}[/DELEGATE]'
        text, directive = MR1._parse_response(raw)
        assert directive is None

    def test_invalid_agent_type(self):
        raw = '[DELEGATE]{"agent": "rogue", "task": "hack"}[/DELEGATE]'
        text, directive = MR1._parse_response(raw)
        assert directive is None

    def test_kami_no_longer_accepted(self):
        raw = '[DELEGATE]{"agent": "kami", "task": "old style"}[/DELEGATE]'
        text, directive = MR1._parse_response(raw)
        assert directive is None


class TestHelpers:
    def test_generate_task_id(self):
        tid = _generate_task_id()
        assert tid.startswith("task-")
        assert len(tid) > 10

    def test_load_agent_config(self):
        from mr1.mr1 import _MR1_CONFIG_PATH
        config = _load_agent_config(_MR1_CONFIG_PATH)
        assert config["name"] == "mr1"
        assert "model" in config


class TestBuiltinCommands:
    @pytest.fixture
    def mr1_instance(self, tmp_path):
        instance = MR1()
        # Replace the state manager with one using a fresh tmp path
        # so tests aren't polluted by real state.
        instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        return instance

    def test_status_command(self, mr1_instance):
        result = mr1_instance._handle_builtin("/status")
        assert result is not None
        assert "Session:" in result

    def test_tasks_command(self, mr1_instance):
        result = mr1_instance._handle_builtin("/tasks")
        assert result is not None

    def test_history_empty(self, mr1_instance):
        result = mr1_instance._handle_builtin("/history")
        assert "No recent decisions" in result

    def test_unknown_command_returns_none(self, mr1_instance):
        result = mr1_instance._handle_builtin("not a command")
        assert result is None

    def test_memdltr_is_recognized(self, mr1_instance):
        # /memdltr needs a running process — mock it.
        mock_process = MagicMock(spec=MR1Process)
        mock_process.send.return_value = "Summary written. [MR1:DUMP_COMPLETE]"
        mock_process.alive = True
        mock_process.kill.return_value = None
        mr1_instance._process = mock_process

        with patch("mr1.mr1.MR1.start"):
            with patch("mr1.mini.mem_dltr.distill") as mock_distill:
                mock_result = MagicMock()
                mock_result.forgotten = 5
                mock_result.dumped = 3
                mock_result.rag_chunks = 2
                mock_distill.return_value = mock_result

                result = mr1_instance._handle_builtin("/memdltr")
                assert "Memory compressed" in result
                assert "confirmed" in result

    def test_vizualize_launches_visualizer(self, mr1_instance):
        result = mr1_instance._handle_builtin("/vizualize")
        assert "primary MR1 interface" in result
        assert "python main.py" in result

    def test_visualize_alias_is_supported(self, mr1_instance):
        result = mr1_instance._handle_builtin("/visualize")
        assert "npm run viz" in result

    def test_visualize_web_launches_browser_ui(self, mr1_instance):
        with patch("mr1.web_viz.webbrowser.open") as mock_open:
            result = mr1_instance._handle_builtin("/visualize-web")
        assert "web visualizer running at http://" in result
        assert "python main.py --web" in result
        mock_open.assert_called_once()

    def test_test_spawn_agents_command_is_supported(self, mr1_instance):
        with patch.object(mr1_instance, "spawn_test_agents", return_value="spawned synthetic tree") as mock_spawn:
            result = mr1_instance._handle_builtin("/test spawn agents 3")
        assert result == "spawned synthetic tree"
        mock_spawn.assert_called_once_with(3)

    def test_test_spawn_agents_requires_height(self, mr1_instance):
        result = mr1_instance._handle_builtin("/test spawn agents nope")
        assert result == "Usage: /test spawn agents <height>"

    def test_test_kill_agents_command_is_supported(self, mr1_instance):
        with patch.object(mr1_instance, "kill_test_agents", return_value="killed synthetic agents") as mock_kill:
            result = mr1_instance._handle_builtin("/test kill agents")
        assert result == "killed synthetic agents"
        mock_kill.assert_called_once()

    def test_vizualize_handles_missing_npm(self, mr1_instance):
        result = mr1_instance._handle_builtin("/vizualize")
        assert "legacy loop" in result


class TestMR1Process:
    def test_not_alive_before_start(self):
        proc = MR1Process("prompt", "model", ["Read"])
        assert proc.alive is False
        assert proc.pid is None

    def test_send_when_not_alive(self):
        proc = MR1Process("prompt", "model", ["Read"])
        result = proc.send("hello")
        assert "ERROR" in result

    def test_send_updates_resumed_session_id(self):
        proc = MR1Process("prompt", "haiku", ["Read"])
        proc._available = True

        with patch.object(proc, "_invoke", return_value=("hello", None)) as mock_invoke:
            proc._session_id = "sess-1"
            result = proc.send("hello")

        assert result == "hello"
        mock_invoke.assert_called_once_with("hello", resume=True)
        assert proc.session_id == "sess-1"

    def test_send_retries_without_resume_on_error(self):
        proc = MR1Process("prompt", "haiku", ["Read"], session_id="sess-1")
        proc._available = True

        with patch.object(
            proc,
            "_invoke",
            side_effect=[("", "resume failed"), ("fresh answer", None)],
        ) as mock_invoke:
            result = proc.send("hello")

        assert result == "fresh answer"
        assert mock_invoke.call_args_list[0].kwargs == {"resume": True}
        assert mock_invoke.call_args_list[1].kwargs == {"resume": False}
        assert proc.session_id is None


class TestStep:
    @pytest.fixture
    def mr1_with_mock_process(self):
        mr1_instance = MR1()
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mr1_instance._process = mock_process
        return mr1_instance, mock_process

    def test_direct_answer(self, mr1_with_mock_process):
        mr1_instance, mock_process = mr1_with_mock_process
        mock_process.send.return_value = "Here is the answer."

        result = mr1_instance.step("what is 2+2?")
        assert result == "Here is the answer."
        mock_process.send.assert_called_once_with("what is 2+2?")
