"""Tests for mr1.mr1 — orchestrator logic (no subprocess spawning)."""

import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
from mr1.capability_policy import (
    CapabilityApprovalRequest,
    CapabilityRequest,
    build_scope_context,
)
from mr1.dataflow import Artifact, ResolvedTaskInput, TaskOutput
from mr1.kazi_runner import RunStatus
from mr1.scoped_agents import PersistentAgent, new_agent_id
from mr1.workflow_models import TaskStatus, WorkflowStatus
from mr1.mr1 import (
    MR1,
    MR1Process,
    StateManager,
    StateCorruptionError,
    _ORCHESTRATOR_PROMPT,
    _load_agent_config,
    _generate_task_id,
)
from mr1.kazi_runner import MockRunner
from mr1.mrn_run import MRnRunResult
from mr1.orchestrator import state as orchestrator_state
from mr1.workflow_store import WorkflowStore


class FakeCompiler:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, prompt: str) -> str:
        self.prompts.append((system_prompt, prompt))
        if not self.responses:
            raise AssertionError("no compiler responses configured")
        return self.responses.pop(0)


def _seed_legacy_agent(
    mr1_instance: MR1,
    *,
    parent_agent_id: str,
    title: str,
    status: str = "active",
    run_status: str = "idle",
) -> PersistentAgent:
    parent = mr1_instance._scoped_agents.require_agent(parent_agent_id)
    agent = PersistentAgent(
        agent_id=new_agent_id(),
        agent_type="mrn",
        title=title,
        tree_level=parent.tree_level + 1,
        parent_agent_id=parent_agent_id,
        status=status,
        run_status=run_status,
    )
    path = mr1_instance._scoped_agents.agent_path(agent.agent_id)
    path.write_text(json.dumps(agent.to_dict(), indent=2), encoding="utf-8")
    mr1_instance._scoped_agents.ensure_agent_files(agent.agent_id)
    return agent


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

    def test_save_fsyncs_file_and_directory(self, tmp_path, monkeypatch):
        state_path = tmp_path / "mr1_state.json"
        manager = StateManager(state_path=state_path)
        fsync_calls: list[int] = []

        def fake_fsync(fd: int) -> None:
            fsync_calls.append(fd)

        monkeypatch.setattr(orchestrator_state.os, "fsync", fake_fsync)

        manager.save()

        assert state_path.exists()
        assert len(fsync_calls) >= 2

    def test_corrupted_state_raises_without_discarding_file(self, tmp_path):
        state_path = tmp_path / "mr1_state.json"
        state_path.write_text("{not json", encoding="utf-8")

        with pytest.raises(StateCorruptionError, match="runtime state is corrupted"):
            StateManager(state_path=state_path)

        assert state_path.read_text(encoding="utf-8") == "{not json"

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

    def test_active_tasks_returns_detached_snapshot(self, state_mgr):
        state_mgr.add_task("t1", "kazi", "test task", 123)

        active = state_mgr.active_tasks
        active["t1"]["status"] = "completed"

        assert state_mgr.get_task("t1")["status"] == "running"
        assert "t1" in state_mgr.active_tasks

    def test_pending_workflow_returns_detached_snapshot(self, state_mgr):
        state_mgr.set_pending_workflow({
            "spec": {
                "title": "draft",
                "tasks": [{"label": "a"}],
            }
        })

        pending = state_mgr.pending_workflow
        assert pending is not None
        pending["spec"]["tasks"].append({"label": "b"})

        assert state_mgr.pending_workflow == {
            "spec": {
                "title": "draft",
                "tasks": [{"label": "a"}],
            }
        }

    def test_concurrent_decision_logging_preserves_all_entries(self, state_mgr, monkeypatch):
        monkeypatch.setattr(orchestrator_state, "_MAX_DECISIONS", 200)
        worker_count = 40
        barrier = threading.Barrier(worker_count)

        def worker(index: int) -> None:
            barrier.wait()
            state_mgr.add_decision(f"user input {index}", f"action_{index}")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(worker, range(worker_count)))

        actions = {entry["action"] for entry in state_mgr._state["decisions"]}
        assert len(state_mgr._state["decisions"]) == worker_count
        assert actions == {f"action_{index}" for index in range(worker_count)}

    def test_concurrent_conversation_updates_preserve_all_entries(self, state_mgr, monkeypatch):
        monkeypatch.setattr(orchestrator_state, "_MAX_CONVERSATION", 200)
        worker_count = 40
        barrier = threading.Barrier(worker_count)

        def worker(index: int) -> None:
            barrier.wait()
            state_mgr.add_conversation("user", f"message {index}", task_id=f"task-{index}")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(worker, range(worker_count)))

        conversation = state_mgr.conversation
        assert len(conversation) == worker_count
        assert {entry["text"] for entry in conversation} == {
            f"message {index}" for index in range(worker_count)
        }

    def test_pending_workflow_snapshot_is_stable_while_source_mutates(self, state_mgr):
        draft = {
            "spec": {
                "title": "draft",
                "tasks": [],
            }
        }
        ready = threading.Event()
        mutate = threading.Event()
        mutated = threading.Event()
        writer_done = threading.Event()
        observed_lengths: list[int] = []

        def writer() -> None:
            state_mgr.set_pending_workflow(draft)
            ready.set()
            for index in range(5):
                mutate.wait(timeout=1.0)
                mutate.clear()
                draft["spec"]["tasks"].append({"label": f"task-{index}"})
                mutated.set()
            writer_done.set()

        thread = threading.Thread(target=writer)
        thread.start()
        assert ready.wait(timeout=1.0)

        for _ in range(5):
            mutate.set()
            assert mutated.wait(timeout=1.0)
            mutated.clear()
            pending = state_mgr.pending_workflow
            assert pending is not None
            observed_lengths.append(len(pending["spec"]["tasks"]))

        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert writer_done.is_set()
        assert observed_lengths == [0, 0, 0, 0, 0]


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
        instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            inbox_auto_triage=False,
        )
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

    def test_agent_builtin_shows_mission_and_iteration(self, mr1_instance):
        child = mr1_instance._scoped_agents.create_child_agent(
            mr1_instance._root_agent_id,
            "builtin-mission-test",
        )
        child.mission = "Investigate the repo"
        child.run_status = "working"
        child.current_iteration = 2
        mr1_instance._scoped_agents.save_agent(child)

        result = mr1_instance._handle_builtin(f"/agent {child.agent_id}")

        assert "mission:      Investigate the repo" in result
        assert "run_status:   working" in result
        assert "iteration:    2" in result

    def test_agent_builtin_shows_runtime_activity_clarity(self, mr1_instance, tmp_path):
        child = mr1_instance._scoped_agents.create_child_agent(
            mr1_instance._root_agent_id,
            "builtin-runtime-test",
        )
        child.run_status = "working"
        mr1_instance._scoped_agents.save_agent(child)

        result = mr1_instance._handle_builtin(f"/agent {child.agent_id}")

        assert "run_status:   working" in result
        assert "active_jobs:  0" in result
        assert "runtime_activity: no active jobs" in result
        assert "has_running_processes: no" in result

    def test_kill_builtin_clarifies_persistent_agents_are_not_affected(self, mr1_instance):
        with patch.object(mr1_instance, "_running_test_agent_count", return_value=0):
            with patch.object(mr1_instance._spawner, "kill_all", return_value=0):
                with patch.object(mr1_instance, "kill_test_agents", return_value="No synthetic test agents are running."):
                    result = mr1_instance._handle_builtin("/kill")

        assert "Persistent agents are not affected." in result
        assert "/agent kill <ag-id>" in result

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
                assert mr1_instance._state.claude_session_id is None

    def test_vizualize_launches_visualizer(self, mr1_instance):
        result = mr1_instance._handle_builtin("/vizualize")
        assert "read-only viewer" in result
        assert "python -m mr1.tui" in result

    def test_visualize_alias_is_supported(self, mr1_instance):
        result = mr1_instance._handle_builtin("/visualize")
        assert "python main.py --tui" in result

    def test_visualize_web_alias_points_to_tui(self, mr1_instance):
        result = mr1_instance._handle_builtin("/visualize-web")
        assert "browser visualizer was removed" in result
        assert "python main.py --tui" in result

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
        assert "python -m mr1.tui" in result


class TestWorkflowBuiltinCommands:
    @pytest.fixture
    def mr1_instance(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        instance = MR1(
            workflow_store=store,
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
        )
        instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        return instance

    def _write_spec(self, tmp_path: Path) -> Path:
        path = tmp_path / "workflow.json"
        path.write_text(
            json.dumps(
                {
                    "title": "Example workflow",
                    "tasks": [
                        {
                            "label": "a",
                            "title": "Task A",
                            "prompt": "Do A",
                        },
                        {
                            "label": "b",
                            "title": "Task B",
                            "prompt": "Do B",
                            "depends_on": ["a"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _write_fragment(self, tmp_path: Path, name: str, tasks: list[dict]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        return path

    def test_workflow_submit_and_listing_commands(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)

        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        assert submit_result is not None
        assert submit_result.startswith("submitted: wf-")

        workflows_result = mr1_instance._handle_builtin("/workflows")
        assert workflows_result is not None
        assert "Example workflow" in workflows_result

    def test_workflow_and_task_detail_commands(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)

        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        assert wf is not None

        workflow_result = mr1_instance._handle_builtin(f"/workflow {wf_id}")
        assert workflow_result is not None
        assert f"workflow: {wf_id}" in workflow_result
        assert "tasks:" in workflow_result

        task_id = wf.label_to_task_id["a"]
        task_result = mr1_instance._handle_builtin(f"/task {task_id}")
        assert task_result is not None
        assert f"task:       {task_id}" in task_result
        assert "label:      a" in task_result

    @pytest.mark.parametrize(
        ("cmd", "expected_prefix"),
        [
            ("/workflow", "Usage: /workflow <workflow_id>"),
            ("/workflow rerun", "Usage: /workflow rerun <workflow_id> <task>"),
            ("/workflow cancel", "Usage: /workflow cancel <workflow_id>"),
            ("/workflow append", "Usage: /workflow append <workflow_id> <path>"),
            ("/workflow insert", "Usage: /workflow insert <workflow_id> <after_task> <path>"),
            ("/workflow replace", "Usage: /workflow replace [-r] <workflow_id> <task> <path>"),
            ("/workflow replace -r", "Usage: /workflow replace [-r] <workflow_id> <task> <path>"),
            ("/workflow trigger", "Usage: /workflow trigger <workflow_id> <label-or-task-id> [event_name]"),
        ],
    )
    def test_workflow_builtin_returns_usage_for_bare_and_incomplete_commands(
        self,
        mr1_instance,
        cmd,
        expected_prefix,
    ):
        result = mr1_instance._handle_builtin(cmd)

        assert result is not None
        assert result.startswith(expected_prefix)

    def test_jobs_events_and_scheduler_tick_commands(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)

        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]

        tick_result = mr1_instance._handle_builtin("/scheduler tick")
        assert tick_result == "scheduler ticked."

        jobs_result = mr1_instance._handle_builtin("/jobs")
        assert jobs_result is not None
        assert "WORKFLOW_ID" in jobs_result
        assert wf_id in jobs_result

        events_result = mr1_instance._handle_builtin(f"/events {wf_id}")
        assert events_result is not None
        assert "workflow_submitted" in events_result

    def test_result_inputs_and_artifacts_commands(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        a = wf.task_by_label("a")
        b = wf.task_by_label("b")
        mr1_instance._workflow_store.write_task_output(
            wf_id,
            a.task_id,
            TaskOutput(
                task_id=a.task_id,
                workflow_id=wf_id,
                status="succeeded",
                summary="done",
                text="hello world",
                artifacts=[
                    Artifact(
                        artifact_id="art-1",
                        workflow_id=wf_id,
                        task_id=a.task_id,
                        name="report",
                        kind="json",
                        path="/tmp/report.json",
                    )
                ],
            ),
        )
        mr1_instance._workflow_store.write_task_inputs(
            wf_id,
            b.task_id,
            [
                ResolvedTaskInput(
                    name="producer_text",
                    source="a.result.text",
                    resolved_task_id=a.task_id,
                    resolved_type="text",
                    value="hello world",
                )
            ],
        )
        a.artifacts = [
            Artifact(
                artifact_id="art-1",
                workflow_id=wf_id,
                task_id=a.task_id,
                name="report",
                kind="json",
                path="/tmp/report.json",
            )
        ]
        mr1_instance._workflow_store.save_workflow(wf)

        result_output = mr1_instance._handle_builtin(f"/result {a.task_id}")
        assert "hello world" in result_output

        inputs_output = mr1_instance._handle_builtin(f"/inputs {b.task_id}")
        assert "producer_text" in inputs_output

        artifacts_output = mr1_instance._handle_builtin(f"/artifacts {wf_id}")
        assert "report" in artifacts_output

    def test_workflow_replace_flag_controls_autorun(self, mr1_instance, tmp_path):
        replace_path = tmp_path / "replace.json"
        replace_path.write_text(json.dumps({
            "tasks": [{
                "label": "a",
                "title": "A2",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "replacement",
            }]
        }), encoding="utf-8")

        path = self._write_spec(tmp_path)
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]
        mr1_instance._handle_builtin("/scheduler tick")
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        mr1_instance._scheduler._runner.complete(a_id, RunStatus.FAILED, error="boom")
        mr1_instance._handle_builtin("/scheduler tick")

        result = mr1_instance._handle_builtin(f"/workflow replace {wf_id} a {replace_path}")
        assert result == f"workflow updated: {wf_id}"
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        assert wf.task_by_label("a").status is TaskStatus.READY

        path2 = self._write_spec(tmp_path)
        submit_result2 = mr1_instance._handle_builtin(f"/workflow submit {path2}")
        wf_id2 = submit_result2.split(": ", 1)[1]
        mr1_instance._handle_builtin("/scheduler tick")
        wf2 = mr1_instance._workflow_store.load_workflow(wf_id2)
        a_id2 = wf2.label_to_task_id["a"]
        mr1_instance._scheduler._runner.complete(a_id2, RunStatus.FAILED, error="boom")
        mr1_instance._handle_builtin("/scheduler tick")

        result = mr1_instance._handle_builtin(f"/workflow replace -r {wf_id2} a {replace_path}")
        assert result == f"workflow updated and rerun: {wf_id2}"
        wf = mr1_instance._workflow_store.load_workflow(wf_id2)
        assert wf.task_by_label("a").status is TaskStatus.RUNNING

    def test_workflow_rerun_rejects_cancelled_workflow(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]

        assert mr1_instance._handle_builtin(f"/workflow cancel {wf_id}") == f"workflow cancelled: {wf_id}"

        result = mr1_instance._handle_builtin(f"/workflow rerun {wf_id} a")

        assert result == f"workflow cancelled and cannot be mutated: {wf_id}"
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.CANCELLED
        assert wf.task_by_label("a").status is TaskStatus.CANCELLED

    def test_workflow_append_rejects_cancelled_workflow(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)
        append_path = self._write_fragment(
            tmp_path,
            "append.json",
            [{
                "label": "c",
                "title": "Task C",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "Do C",
            }],
        )
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]

        assert mr1_instance._handle_builtin(f"/workflow cancel {wf_id}") == f"workflow cancelled: {wf_id}"

        result = mr1_instance._handle_builtin(f"/workflow append {wf_id} {append_path}")

        assert result == f"workflow cancelled and cannot be mutated: {wf_id}"
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.CANCELLED
        assert "c" not in wf.label_to_task_id

    def test_workflow_insert_rejects_cancelled_workflow(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)
        insert_path = self._write_fragment(
            tmp_path,
            "insert.json",
            [{
                "label": "x",
                "title": "Task X",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "Do X",
            }],
        )
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]

        assert mr1_instance._handle_builtin(f"/workflow cancel {wf_id}") == f"workflow cancelled: {wf_id}"

        result = mr1_instance._handle_builtin(f"/workflow insert {wf_id} a {insert_path}")

        assert result == f"workflow cancelled and cannot be mutated: {wf_id}"
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.CANCELLED
        assert "x" not in wf.label_to_task_id

    def test_workflow_replace_rejects_cancelled_workflow(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)
        replace_path = self._write_fragment(
            tmp_path,
            "replace_cancelled.json",
            [{
                "label": "a",
                "title": "Task A Replacement",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "Replacement",
            }],
        )
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]

        assert mr1_instance._handle_builtin(f"/workflow cancel {wf_id}") == f"workflow cancelled: {wf_id}"

        result = mr1_instance._handle_builtin(f"/workflow replace {wf_id} a {replace_path}")

        assert result == f"workflow cancelled and cannot be mutated: {wf_id}"
        wf = mr1_instance._workflow_store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.CANCELLED
        assert wf.task_by_label("a").prompt == "Do A"

    def test_workflow_cancel_reports_already_cancelled(self, mr1_instance, tmp_path):
        path = self._write_spec(tmp_path)
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {path}")
        wf_id = submit_result.split(": ", 1)[1]

        assert mr1_instance._handle_builtin(f"/workflow cancel {wf_id}") == f"workflow cancelled: {wf_id}"

        result = mr1_instance._handle_builtin(f"/workflow cancel {wf_id}")

        assert result == f"workflow already cancelled: {wf_id}"

    def test_agent_create_rejects_reserved_all_title(self, mr1_instance):
        result = mr1_instance._handle_builtin("/agent create all")

        assert result == "agent title 'all' is reserved"

    def test_agent_create_rejects_null_byte_in_title(self, mr1_instance):
        result = mr1_instance._handle_builtin("/agent create Title\x00Null")

        assert result == "agent title must not contain control characters"

    def test_agent_create_rejects_duplicate_exact_title(self, mr1_instance):
        first = mr1_instance._handle_builtin("/agent create Alpha")
        second = mr1_instance._handle_builtin("/agent create Alpha")

        assert first.startswith("ag-")
        assert second == "agent title already exists: Alpha"

    def test_agent_create_rejects_duplicate_case_insensitive_title(self, mr1_instance):
        first = mr1_instance._handle_builtin("/agent create Alpha")
        second = mr1_instance._handle_builtin("/agent create alpha")

        assert first.startswith("ag-")
        assert second == "agent title already exists: Alpha"

    def test_agent_create_allows_distinct_titles(self, mr1_instance):
        first = mr1_instance._handle_builtin("/agent create RepoResearcher")
        second = mr1_instance._handle_builtin("/agent create PaperResearcher")

        assert first.startswith("ag-")
        assert second.startswith("ag-")
        assert first != second

    def test_shutdown_stops_scheduler(self, mr1_instance):
        mr1_instance._process = MagicMock(spec=MR1Process)

        with patch.object(mr1_instance._spawner, "kill_all", return_value=0):
            with patch.object(mr1_instance, "kill_test_agents", return_value="No synthetic test agents running."):
                with patch.object(mr1_instance._scheduler, "shutdown") as mock_shutdown:
                    mr1_instance.shutdown("test")

        mock_shutdown.assert_called_once_with(cancel_running=True)

    def test_shutdown_preserves_claude_session_id(self, mr1_instance):
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._state.set_claude_session_id("sess-123")

        with patch.object(mr1_instance._spawner, "kill_all", return_value=0):
            with patch.object(mr1_instance, "kill_test_agents", return_value="No synthetic test agents running."):
                with patch.object(mr1_instance._scheduler, "shutdown"):
                    mr1_instance.shutdown("test")

        assert mr1_instance._state.claude_session_id == "sess-123"


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

    def test_send_does_not_retry_without_explicit_retry_policy(self):
        proc = MR1Process("prompt", "haiku", ["Read"], session_id="sess-1")
        proc._available = True

        with patch.object(
            proc,
            "_invoke",
            side_effect=[("", "resume failed"), ("fresh answer", None)],
        ) as mock_invoke:
            result = proc.send("hello")

        assert "resume failed" in result
        assert mock_invoke.call_count == 1
        assert mock_invoke.call_args.kwargs == {"resume": True}
        assert proc.session_id is None

    def test_send_retries_when_explicitly_retriable(self):
        proc = MR1Process("prompt", "haiku", ["Read"], session_id="sess-1")
        proc._available = True

        with patch.object(
            proc,
            "_invoke",
            side_effect=[("", "resume failed"), ("fresh answer", None)],
        ) as mock_invoke:
            result = proc.send("hello", retriable=True)

        assert result == "fresh answer"
        assert mock_invoke.call_args_list[0].kwargs == {"resume": True}
        assert mock_invoke.call_args_list[1].kwargs == {"resume": False}
        assert proc.session_id is None

    def test_start_uses_governed_read_only_brain_tools(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
            inbox_auto_triage=False,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")

        with patch("mr1.orchestrator.root.MR1Process") as mock_process_cls:
            mock_process = mock_process_cls.return_value
            mock_process.session_id = None
            mr1_instance.start()

        assert mock_process_cls.call_args.kwargs["tools"] == ["Read", "Glob", "Grep"]


class TestStep:
    def _seed_completed_workflow(self, mr1_instance, tmp_path, *, title="Inspection workflow"):
        spec_path = tmp_path / f"{title.replace(' ', '_').lower()}.json"
        spec_path.write_text(json.dumps({
            "title": title,
            "tasks": [
                {
                    "label": "analyze",
                    "title": "Analyze",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Analyze the request",
                },
                {
                    "label": "synthesize_findings",
                    "title": "Synthesize Findings",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Summarize findings",
                    "depends_on": ["analyze"],
                },
            ],
        }), encoding="utf-8")
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {spec_path}")
        workflow_id = submit_result.split(": ", 1)[1]
        workflow = mr1_instance._workflow_store.load_workflow(workflow_id)
        assert workflow is not None
        workflow.status = WorkflowStatus.SUCCEEDED
        workflow.finished_at = "2026-05-03T04:41:13+00:00"
        analyze = workflow.task_by_label("analyze")
        report = workflow.task_by_label("synthesize_findings")
        analyze.status = TaskStatus.SUCCEEDED
        analyze.result_summary = "Collected source material."
        report.status = TaskStatus.SUCCEEDED
        report.result_summary = "Found two regressions and one approval blocker."
        report_output_path = mr1_instance._workflow_store.write_task_output(
            workflow_id,
            report.task_id,
            TaskOutput(
                task_id=report.task_id,
                workflow_id=workflow_id,
                status="succeeded",
                summary="Found two regressions and one approval blocker.",
                text=(
                    "The workflow found two regressions in inspection routing and one approval blocker. "
                    "The existing workflow should be inspected directly, and blocked tasks need explicit rerun guidance."
                ),
            ),
        )
        report.output_path = str(report_output_path)
        mr1_instance._workflow_store.save_workflow(workflow)
        return workflow_id, workflow

    def _seed_failed_task(self, mr1_instance, tmp_path):
        workflow_id, workflow = self._seed_completed_workflow(
            mr1_instance,
            tmp_path,
            title="Failure inspection workflow",
        )
        workflow.status = WorkflowStatus.FAILED
        task = workflow.task_by_label("synthesize_findings")
        task.status = TaskStatus.FAILED
        task.last_error = "outside_actor_scope"
        task.last_error_type = "policy_block"
        result_path = mr1_instance._workflow_store.write_result(
            workflow_id,
            task.task_id,
            {
                "task_id": task.task_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "summary": "outside_actor_scope",
                "text": "",
                "data": {
                    "approval_request_id": "cap_approval_test",
                    "decision": {"reason": "outside_actor_scope"},
                },
                "error": "outside_actor_scope",
                "error_type": "policy_block",
                "failure_type": "policy_block",
            },
        )
        task.result_path = str(result_path)
        mr1_instance._workflow_store.save_workflow(workflow)
        return workflow_id, task.task_id

    def _seed_semantic_mismatch_workflow(self, mr1_instance, tmp_path):
        workflow_id, workflow = self._seed_completed_workflow(
            mr1_instance,
            tmp_path,
            title="Semantic mismatch workflow",
        )
        workflow.status = WorkflowStatus.FAILED
        task = workflow.task_by_label("synthesize_findings")
        task.status = TaskStatus.FAILED
        task.last_error_type = "semantic_mismatch"
        task.last_error = (
            "task output asked a question or requested clarification instead of completing the intended action"
        )
        task.result_summary = "task completed mechanically but did not satisfy intended postcondition"
        result_path = mr1_instance._workflow_store.write_result(
            workflow_id,
            task.task_id,
            {
                "task_id": task.task_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "summary": "task completed mechanically but did not satisfy intended postcondition",
                "text": "I checked cron jobs. Which agent should I terminate?",
                "data": {},
                "error": task.last_error,
                "error_type": "semantic_mismatch",
                "failure_type": "semantic_mismatch",
            },
        )
        task.result_path = str(result_path)
        mr1_instance._workflow_store.save_workflow(workflow)
        return workflow_id

    @pytest.fixture
    def mr1_with_mock_process(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mr1_instance._process = mock_process
        return mr1_instance, mock_process

    def test_direct_answer(self, mr1_with_mock_process):
        mr1_instance, mock_process = mr1_with_mock_process
        mock_process.send.return_value = "Here is the answer."

        result = mr1_instance.step("what is 2+2?")
        assert result == "Here is the answer."
        sent = mock_process.send.call_args.args[0]
        assert "Answer this request directly" in sent
        assert "what is 2+2?" in sent

    def test_unknown_slash_command_is_refused_without_reaching_brain(self, mr1_with_mock_process):
        mr1_instance, mock_process = mr1_with_mock_process

        result = mr1_instance.step("/notacommand")

        assert result.startswith("Unknown slash command: /notacommand.")
        mock_process.send.assert_not_called()

    @pytest.mark.parametrize(
        "brain_reply",
        [
            "Got it-renaming Librarian to PaperLibrarian now.",
            "PaperLibrarian is paused.",
            "Created a new agent called Archivist.",
            "Deleted the Archivist agent.",
        ],
    )
    def test_direct_answer_blocks_unverified_mutation_claims(self, mr1_with_mock_process, brain_reply):
        mr1_instance, mock_process = mr1_with_mock_process
        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Librarian")
        child.run_status = "working"
        mr1_instance._scoped_agents.save_agent(child)
        before = [
            (agent.agent_id, agent.title, agent.status, agent.run_status)
            for agent in mr1_instance._scoped_agents.list_visible_agents(root.agent_id)
        ]
        mock_process.send.return_value = brain_reply

        result = mr1_instance.step("what is 2+2?")

        after = [
            (agent.agent_id, agent.title, agent.status, agent.run_status)
            for agent in mr1_instance._scoped_agents.list_visible_agents(root.agent_id)
        ]
        assert result == (
            "I did not execute any runtime mutation on this turn, so I cannot claim that an agent or workflow "
            "was created, renamed, paused, or deleted. Use an explicit operational command instead."
        )
        assert before == after

    def test_kill_named_agent_title_routes_operationally(self, mr1_with_mock_process):
        mr1_instance, mock_process = mr1_with_mock_process
        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Archivist")

        result = mr1_instance.step("kill the archivist permanently")

        updated = mr1_instance._scoped_agents.require_agent(child.agent_id)
        artifact = json.loads(sorted((tmp_path := mr1_instance._state._path.parent / "mr1_turns").glob("turn-*.json"))[-1].read_text(encoding="utf-8"))
        decision_event = mr1_instance._event_log.list_events()[-1]
        assert result == f"terminated agent: {child.agent_id}"
        assert updated.status == "terminated"
        assert artifact["routing_decision"]["final_action"] == "run_commands"
        assert artifact["routing_decision"]["matched_signals"]["matched_operational_verbs"] == ["kill"]
        assert decision_event.event_type == "runtime_turn_decided"
        assert decision_event.metadata["final_action"] == "run_commands"
        mock_process.send.assert_not_called()

    def test_direct_answer_observation_loop_reads_full_message_body(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mr1_instance._process = mock_process

        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Sentinel")
        long_body = ("Parent review " + ("needed " * 900)).strip()
        message = mr1_instance._message_store.create_message(
            from_agent_id=child.agent_id,
            to_agent_id=root.agent_id,
            kind="question",
            subject="Need parent review",
            body=long_body,
        )
        grounding = mr1_instance.build_runtime_grounding()
        grounded_message = next(
            item for item in grounding["messages"] if item["message_id"] == message.message_id
        )
        assert grounded_message["body_full_available"] is True
        assert grounded_message["body_truncated"] is True
        assert grounded_message["body_preview"] != long_body

        mock_process.send.side_effect = [
            f'[OBSERVE]{{"calls":[{{"name":"read_message","args":{{"message_id":"{message.message_id}"}}}}]}}[/OBSERVE]',
            f"The child message said in full: {long_body}",
        ]

        result = mr1_instance.step("What did that child message say exactly?")

        assert result == f"The child message said in full: {long_body}"
        assert "[OBSERVE]" not in result
        assert mock_process.send.call_count == 2
        second_prompt = mock_process.send.call_args_list[1].args[0]
        assert "=== OBSERVATION RESULTS ===" in second_prompt
        assert '"request_valid": true' in second_prompt
        assert f'"message_id": "{message.message_id}"' in second_prompt
        assert long_body in second_prompt

    def test_direct_answer_invalid_observation_returns_error_to_brain_without_execution(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mock_process.send.side_effect = [
            '[OBSERVE]{"calls":[{"name":"read_message","args":{"message_id":"msg-123","full":true}}]}[/OBSERVE]',
            "Final answer after invalid observation.",
        ]
        mr1_instance._process = mock_process
        mr1_instance._runtime_access.read_message = MagicMock(
            side_effect=AssertionError("invalid observation should not execute")
        )

        result = mr1_instance.step("What is the full message?")

        assert result == "Final answer after invalid observation."
        assert mock_process.send.call_count == 2
        mr1_instance._runtime_access.read_message.assert_not_called()
        second_prompt = mock_process.send.call_args_list[1].args[0]
        assert "=== OBSERVATION RESULTS ===" in second_prompt
        assert '"request_valid": false' in second_prompt
        assert '"executed": false' in second_prompt
        assert "unexpected args: full" in second_prompt

    def test_direct_answer_observation_loop_is_capped(self, mr1_with_mock_process):
        mr1_instance, mock_process = mr1_with_mock_process
        mock_process.send.side_effect = [
            '[OBSERVE]{"calls":[{"name":"list_agents","args":{"limit":1}}]}[/OBSERVE]',
            '[OBSERVE]{"calls":[{"name":"list_agents","args":{"limit":1}}]}[/OBSERVE]',
            '[OBSERVE]{"calls":[{"name":"list_agents","args":{"limit":1}}]}[/OBSERVE]',
        ]

        result = mr1_instance.step("what is the current runtime state?")

        assert result == "MR1 could not safely inspect additional runtime detail for this answer."
        assert mock_process.send.call_count == 3

    @pytest.mark.parametrize(
        "user_text",
        [
            "I use Slack because I'm part of a research group.",
            "Educate me about Slack for a research group.",
        ],
    )
    def test_plain_research_group_language_routes_direct_answer(
        self,
        mr1_with_mock_process,
        user_text,
    ):
        mr1_instance, mock_process = mr1_with_mock_process
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "archive")
        mock_process.send.return_value = "Slack guidance."

        result = mr1_instance.step(user_text)

        assert result == "Slack guidance."
        assert mock_process.send.call_count == 1

    def test_direct_answer_includes_runtime_grounding_and_turn_artifact(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mock_process.send.return_value = "Grounded answer."
        mr1_instance._process = mock_process

        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        mr1_instance._message_store.create_message(
            from_agent_id=child.agent_id,
            to_agent_id=root.agent_id,
            kind="question",
            subject="Need input",
            body="Please confirm the next step.",
        )
        spec_path = tmp_path / "workflow.json"
        spec_path.write_text(json.dumps({
            "title": "Example workflow",
            "tasks": [
                {
                    "label": "read_notes",
                    "title": "Read notes",
                    "task_kind": "tool",
                    "tool_type": "read_file",
                    "tool_config": {"path": str(tmp_path / "notes.txt")},
                }
            ],
        }), encoding="utf-8")
        submit_result = mr1_instance._handle_builtin(f"/workflow submit {spec_path}")
        assert submit_result.startswith("submitted: wf-")
        original_request = CapabilityRequest(
            actor_id=child.agent_id,
            actor_type="mrn",
            actor_clearance=child.security_clearance,
            invocation_mode="direct",
            capability_name="read_file",
            args={"path": "README.md"},
            scope=build_scope_context(
                actor_id=child.agent_id,
                workspace_root=tmp_path,
                scoped_agent_store=mr1_instance._scoped_agents,
            ),
            step_id=f"{child.agent_id}:1",
        )
        approval = CapabilityApprovalRequest(
            approval_request_id="cap_approval_runtime",
            requesting_actor_id=child.agent_id,
            capability_name="read_file",
            invocation_mode="direct",
            args={"path": "README.md"},
            risk_score=0.1,
            reason="outside_actor_scope",
            scope_summary=original_request.scope.to_dict(),
            original_request=original_request,
            original_step_id=f"{child.agent_id}:1",
            designated_approver_id=root.agent_id,
        )
        mr1_instance._approval_store.save(approval)

        result = mr1_instance.step("what is the current runtime state?")

        assert result == "Grounded answer."
        sent = mock_process.send.call_args.args[0]
        assert "RUNTIME STATE IS SOURCE OF TRUTH." in sent
        assert "=== RUNTIME GROUNDING ===" in sent
        assert "=== END RUNTIME GROUNDING ===" in sent
        assert "=== ROUTING ADVICE ===" in sent
        assert "=== END ROUTING ADVICE ===" in sent
        assert '"agents": [' in sent
        assert '"workflows": [' in sent
        assert '"messages": [' in sent
        assert '"approvals": [' in sent
        assert '"events": [' in sent
        turn_dir = tmp_path / "mr1_turns"
        artifacts = sorted(turn_dir.glob("turn-*.json"))
        assert len(artifacts) == 1
        artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
        decision_event = mr1_instance._event_log.list_events()[-1]
        assert artifact["route"] == "direct_answer"
        assert artifact["route_advice"]["route"] == "direct_response"
        assert artifact["routing_decision"]["advisor_route"] == "direct_response"
        assert artifact["routing_decision"]["final_action"] == "direct_response"
        assert artifact["routing_decision"]["override_applied"] is False
        assert artifact["runtime_grounding"]["agents"]
        assert artifact["runtime_grounding"]["workflows"]
        assert artifact["runtime_grounding"]["messages"]
        assert artifact["runtime_grounding"]["approvals"]
        assert artifact["runtime_grounding"]["events"]
        assert artifact["brain_prompt"] == sent
        assert decision_event.event_type == "runtime_turn_decided"
        assert decision_event.target_id == artifact["turn_id"]
        assert decision_event.metadata["advisor_route"] == "direct_response"
        assert decision_event.metadata["final_action"] == "direct_response"
        assert decision_event.record_path == str(artifacts[0])

    def test_pause_agent_command_records_informative_override_reason(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        mr1_instance._state.set_reference_state("last_referenced_agent_id", child.agent_id)

        result = mr1_instance.step("pause that agent")

        artifacts = sorted((tmp_path / "mr1_turns").glob("turn-*.json"))
        artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
        decision = artifact["routing_decision"]
        assert "exact execute-now command details" in result
        assert artifact["route"] == "ask_clarification"
        assert artifact["route_advice"]["route"] == "run_commands"
        assert decision["override_applied"] is True
        assert "did not have enough concrete target or command detail" in artifact["route_advice_override_reason"]
        assert "did not have enough concrete target or command detail" in decision["override_cause"]
        assert "safe execute-now command details" in decision["missing_signals"]

    def test_runtime_grounding_includes_referenced_pending_parent_message_ids(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")

        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Sentinel")
        child.run_status = "waiting"
        child.last_action = {
            "action": "send_message",
            "reason": "Waiting for parent review of the long message body",
            "next_status": "waiting",
        }
        mr1_instance._scoped_agents.save_agent(child)
        long_body = "Parent review " + ("needed " * 900)
        message = mr1_instance._message_store.create_message(
            from_agent_id=child.agent_id,
            to_agent_id=root.agent_id,
            kind="question",
            subject="Need parent review",
            body=long_body,
        )
        mr1_instance._message_store.mark_read(message.message_id, actor_id=root.agent_id)

        grounding = mr1_instance.build_runtime_grounding()

        sentinel = next(item for item in grounding["agents"] if item["agent_id"] == child.agent_id)
        assert sentinel["pending_parent_messages"][0]["message_id"] == message.message_id
        assert grounding["messages"]
        grounded_message = next(item for item in grounding["messages"] if item["message_id"] == message.message_id)
        assert grounded_message["body_full_available"] is True
        assert grounded_message["body_truncated"] is True

    @pytest.mark.parametrize(
        "user_text",
        [
            "create a child responsible for tool creation",
            "delegate this domain to an agent",
            "have an agent own this area",
            "let that agent propose/create/test three tools",
        ],
    )
    def test_persistent_delegation_requests_are_classified(self, mr1_with_mock_process, user_text):
        mr1_instance, _mock_process = mr1_with_mock_process

        assert mr1_instance._classify_turn_route(user_text, None) == "persistent_delegation"

    def test_non_ownership_requests_preserve_existing_routing(self, mr1_with_mock_process):
        mr1_instance, _mock_process = mr1_with_mock_process

        assert mr1_instance._classify_turn_route("what is 2+2?", None) == "direct_answer"
        assert (
            mr1_instance._classify_turn_route(
                "Read a file, check Python version, and summarize",
                None,
            )
            == "create_workflow"
        )

    @pytest.mark.parametrize(
        ("user_text", "expected_route"),
        [
            (
                "Describe to me in detail in what situation you would use tools, create workflows, create an agent as an owner, pick from the list of agent children u have.",
                "direct_answer",
            ),
            ("when would you create an agent?", "direct_answer"),
            ("compare tools vs workflows vs agents", "direct_answer"),
            ("create an agent responsible for X", "persistent_delegation"),
            ("I want you to create an owner agent for X", "persistent_delegation"),
        ],
    )
    def test_meta_policy_questions_do_not_trigger_persistent_delegation(
        self,
        mr1_with_mock_process,
        user_text,
        expected_route,
    ):
        mr1_instance, _mock_process = mr1_with_mock_process

        assert mr1_instance._classify_turn_route(user_text, None) == expected_route

    def test_requested_persistent_child_title_is_preserved(self, mr1_with_mock_process):
        mr1_instance, _mock_process = mr1_with_mock_process

        assert mr1_instance._extract_requested_child_title(
            "Create a child of yours, MR3, to own README inspection",
        ) == "MR3"

    def test_named_persistent_child_title_is_preserved(self, mr1_with_mock_process):
        mr1_instance, _mock_process = mr1_with_mock_process

        assert mr1_instance._extract_requested_child_title(
            "Create a persistent agent named Sage to own README inspection",
        ) == "Sage"

    @pytest.mark.parametrize(
        ("user_input", "expected_title"),
        [
            ("create an agent called archivist that owns screenshots", "archivist"),
            ('create a persistent agent "researcher" to own literature review', "researcher"),
            ("create a persistent agent named librarian to own notes", "librarian"),
        ],
    )
    def test_explicit_persistent_child_names_are_preserved(
        self,
        mr1_with_mock_process,
        user_input,
        expected_title,
    ):
        mr1_instance, _mock_process = mr1_with_mock_process

        assert mr1_instance._extract_requested_child_title(user_input) == expected_title

    def test_special_name_request_uses_memorable_default_title(self, mr1_with_mock_process):
        mr1_instance, _mock_process = mr1_with_mock_process

        assert mr1_instance._extract_requested_child_title(
            "Create a persistent agent with a unique special name to own MR1 evolution",
        ) == "Sentinel"

    @patch("mr1.mr1.MRnRunRunner.run")
    def test_persistent_delegation_creates_child_and_runs_persistent_mrn(self, mock_run, tmp_path):
        compiler = MagicMock()
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        def _fake_run(agent_id, policy, caller_agent_id=None):
            return MRnRunResult(
                run_id="run-1",
                agent_id=agent_id,
                caller_agent_id=caller_agent_id or mr1_instance._root_agent_id,
                started_at="2026-01-01T00:00:00+00:00",
                finished_at="2026-01-01T00:00:02+00:00",
                policy=policy.to_dict(),
                steps=[{"iteration": 1, "action": "create_workflow"}],
                workflows_created=1,
                messages_created=0,
                stopped_reason="confirmation_required",
                status="stopped",
                final_run_status="reporting",
            )

        mock_run.side_effect = _fake_run
        mr1_instance._process.send.return_value = (
            "AGENT_TITLE: MR2\nOwn tool creation: propose, implement, and test new capabilities."
        )

        request = "create a child responsible for tool creation and let that agent propose/create/test"
        response = mr1_instance.step(request)

        assert response.startswith("delegated to persistent agent: ag-")
        assert "run_id=run-1" in response
        assert "workflows_created=1" in response
        assert mock_run.call_count == 1
        agent_id = mock_run.call_args.args[0]
        policy = mock_run.call_args.args[1]
        child = mr1_instance._scoped_agents.require_agent(agent_id)
        assert child.title == "MR2"
        assert child.assignment_packet is not None
        assert child.parent_request == request
        assert child.assignment_packet["full_parent_request"] == request
        assert child.assignment_packet["parent_agent_id"] == mr1_instance._root_agent_id
        assert child.assignment_packet["parent_level"] == 1
        assert child.assignment_packet["child_level"] == 2
        assert child.assignment_packet["assigned_clearance"] == child.security_clearance
        assert child.assignment_packet["child_title"] == "MR2"
        assert child.assignment_packet["relevant_context"]["agents"][0] == mr1_instance._root_agent_id
        assert "Own tool creation" in (child.mission or "")
        assert policy.max_steps == 3
        assert policy.max_workflows_created == 2
        assert policy.require_confirmation_for_workflows is True
        assert compiler.call_count == 0
        assert mr1_instance._process.send.call_count == 1

    @patch("mr1.mr1.MRnRunRunner.run")
    def test_persistent_delegation_uses_requested_child_title(self, mock_run, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True
        mr1_instance._process.send.return_value = (
            "AGENT_TITLE: MR3\nOwn README inspection and produce structured summaries."
        )
        mock_run.return_value = MRnRunResult(
            run_id="run-1",
            agent_id="ag-test",
            caller_agent_id=mr1_instance._root_agent_id,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            policy={},
            steps=[],
            workflows_created=0,
            messages_created=0,
            stopped_reason="waiting",
            status="stopped",
            final_run_status="waiting",
        )

        response = mr1_instance.step(
            "Create a child of yours, MR3, responsible for README inspection.",
        )

        assert response.startswith("delegated to persistent agent: ag-")
        child = mr1_instance._scoped_agents.require_agent(mock_run.call_args.args[0])
        assert child.title == "MR3"
        assert "README inspection" in (child.mission or "")
        assert child.assignment_packet is not None
        assert child.assignment_packet["child_level"] == 2

    @patch("mr1.mr1.MRnRunRunner.run")
    def test_persistent_delegation_prefers_explicit_user_title_over_brain_title(self, mock_run, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True
        mr1_instance._process.send.return_value = (
            "AGENT_TITLE: Sentinel\nOwn research synthesis and note indexing."
        )
        mock_run.return_value = MRnRunResult(
            run_id="run-1",
            agent_id="ag-test",
            caller_agent_id=mr1_instance._root_agent_id,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            policy={},
            steps=[],
            workflows_created=0,
            messages_created=0,
            stopped_reason="waiting",
            status="stopped",
            final_run_status="waiting",
        )

        response = mr1_instance.step(
            "Create a persistent agent called archivist to own my research notes.",
        )

        assert response.startswith("delegated to persistent agent: ag-")
        child = mr1_instance._scoped_agents.require_agent(mock_run.call_args.args[0])
        assert child.title == "archivist"

    @patch("mr1.mr1.MRnRunRunner.run")
    def test_persistent_delegation_named_agent_routes_without_workflow_authoring(self, mock_run, tmp_path):
        compiler = MagicMock()
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True
        mr1_instance._process.send.return_value = (
            "AGENT_TITLE: Sage\nOwn self-evolution: propose, test, and integrate system improvements."
        )
        mock_run.return_value = MRnRunResult(
            run_id="run-1",
            agent_id="ag-test",
            caller_agent_id=mr1_instance._root_agent_id,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            policy={},
            steps=[],
            workflows_created=0,
            messages_created=0,
            stopped_reason="waiting",
            status="stopped",
            final_run_status="waiting",
        )

        response = mr1_instance.step(
            "Create a persistent agent named Sage to own self-evolution.",
        )

        assert response.startswith("delegated to persistent agent: ag-")
        child = mr1_instance._scoped_agents.require_agent(mock_run.call_args.args[0])
        assert child.title == "Sage"
        assert "self-evolution" in (child.mission or "")
        assert compiler.call_count == 0
        assert mr1_instance._process.send.call_count == 1

    @patch("mr1.mr1.MRnRunRunner.run")
    def test_persistent_delegation_rejects_duplicate_existing_title(self, mock_run, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True
        mr1_instance._process.send.return_value = (
            "AGENT_TITLE: Builder\nOwn literature review and synthesis."
        )
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Alpha")

        response = mr1_instance.step(
            "Create a persistent agent named alpha to own literature review.",
        )

        assert response == "agent title already exists: Alpha"
        assert mock_run.call_count == 0
        assert mr1_instance._process.send.call_count == 1

    def test_existing_workflow_finish_question_routes_to_inspection(self, mr1_with_mock_process, tmp_path):
        mr1_instance, mock_process = mr1_with_mock_process
        workflow_id, _workflow = self._seed_completed_workflow(mr1_instance, tmp_path)

        result = mr1_instance.step(f"did workflow {workflow_id} finish?")

        assert f"workflow_id: {workflow_id}" in result
        assert "status:      succeeded" in result
        assert mock_process.send.call_count == 0

    def test_existing_workflow_findings_summary_uses_store_not_compiler(self, mr1_with_mock_process, tmp_path):
        mr1_instance, mock_process = mr1_with_mock_process
        workflow_id, _workflow = self._seed_completed_workflow(mr1_instance, tmp_path)

        with patch.object(mr1_instance._workflow_authoring, "author_request", side_effect=AssertionError("compiler route should not run")):
            result = mr1_instance.step(f"summarize the findings of {workflow_id}")

        assert result.startswith("findings:\nThe workflow found two regressions in inspection routing")
        assert "findings_task: synthesize_findings" in result
        assert mock_process.send.call_count == 0

    def test_failed_task_question_returns_failure_info_and_approval_id(self, mr1_with_mock_process, tmp_path):
        mr1_instance, mock_process = mr1_with_mock_process
        _workflow_id, task_id = self._seed_failed_task(mr1_instance, tmp_path)

        result = mr1_instance.step(f"why did {task_id} fail?")

        assert f"task_id:   {task_id}" in result
        assert "failure:   policy_block" in result
        assert "approval:  cap_approval_test" in result
        assert mock_process.send.call_count == 0

    def test_that_workflow_uses_last_referenced_workflow_id(self, mr1_with_mock_process, tmp_path):
        mr1_instance, mock_process = mr1_with_mock_process
        workflow_id, workflow = self._seed_completed_workflow(mr1_instance, tmp_path)
        mr1_instance._state.set_reference_state("last_referenced_workflow_id", workflow_id)
        mr1_instance._state.set_reference_alias("workflows", workflow.title, workflow_id)

        with patch.object(mr1_instance._workflow_authoring, "author_request", side_effect=AssertionError("compiler route should not run")):
            result = mr1_instance.step("did that workflow finish running? if so summarize its findings")

        assert result.startswith("findings:\nThe workflow found two regressions in inspection routing")
        assert f"workflow_id: {workflow_id}" in result
        assert mock_process.send.call_count == 0

    def test_findings_followup_with_it_uses_remembered_workflow(self, mr1_with_mock_process, tmp_path):
        mr1_instance, mock_process = mr1_with_mock_process
        workflow_id, workflow = self._seed_completed_workflow(mr1_instance, tmp_path)
        mr1_instance._state.set_reference_state("last_referenced_workflow_id", workflow_id)
        mr1_instance._state.set_reference_alias("workflows", workflow.title, workflow_id)

        with patch.object(mr1_instance._workflow_authoring, "author_request", side_effect=AssertionError("compiler route should not run")):
            result = mr1_instance.step("what were the findings of it?")

        assert result.startswith("findings:\nThe workflow found two regressions in inspection routing")
        assert f"workflow_id: {workflow_id}" in result
        assert mock_process.send.call_count == 0

    def test_workflow_inspection_surfaces_semantic_mismatch(self, mr1_with_mock_process, tmp_path):
        mr1_instance, mock_process = mr1_with_mock_process
        workflow_id = self._seed_semantic_mismatch_workflow(mr1_instance, tmp_path)

        result = mr1_instance.step(f"did workflow {workflow_id} finish?")

        assert "semantic_validation: FAILED" in result
        assert "semantic_mismatch" in result
        assert "status:      failed" in result
        assert mock_process.send.call_count == 0

    def test_ambiguous_workflow_followup_without_known_reference_asks_for_clarification(
        self,
        mr1_with_mock_process,
    ):
        mr1_instance, mock_process = mr1_with_mock_process

        result = mr1_instance.step("did the workflow finish running? if so summarize its findings")

        assert result == "Which workflow do you mean? Provide a workflow_id like wf-...."
        assert mock_process.send.call_count == 0

    def test_dont_create_workflow_phrase_suppresses_authoring_for_existing_workflow(
        self,
        mr1_with_mock_process,
        tmp_path,
    ):
        mr1_instance, mock_process = mr1_with_mock_process
        workflow_id, _workflow = self._seed_completed_workflow(mr1_instance, tmp_path)

        result = mr1_instance.step(
            f"no im not asking you to create a workflow lol. you made this workflow: /workflow {workflow_id} can you check the results of it?"
        )

        assert f"workflow_id: {workflow_id}" in result
        assert "task_statuses:" in result
        assert mock_process.send.call_count == 0

    def test_message_id_reply_routes_to_child_without_brain_or_compiler(self, tmp_path):
        compiler = MagicMock()
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mr1_instance._process = mock_process

        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        message = mr1_instance._message_store.create_message(
            from_agent_id=child.agent_id,
            to_agent_id=root.agent_id,
            kind="question",
            subject="Need clarification",
            body="Which path should I take?",
        )

        response = mr1_instance.step(
            f"{message.message_id} respond to this message and provide MR2 with Continue with the approved implementation.",
        )

        assert response == f"sent clarification to {child.agent_id} for {message.message_id}"
        inbox = mr1_instance._message_store.list_inbox(child.agent_id)
        assert inbox[0].subject == "Re: Need clarification"
        assert inbox[0].body == "Continue with the approved implementation."
        assert compiler.call_count == 0
        assert mock_process.send.call_count == 0

    def test_message_reply_can_be_synthesized_from_context(self, tmp_path):
        compiler = MagicMock()
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mock_process.send.return_value = (
            "Use the name Darwin. Build Discord send/read tools plus a Discord watcher. "
            "Assume two-way integration and proceed directly with implementation."
        )
        mr1_instance._process = mock_process

        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        message = mr1_instance._message_store.create_message(
            from_agent_id=child.agent_id,
            to_agent_id=root.agent_id,
            kind="question",
            subject="Need clarification",
            body="What should I name myself and should I proceed with the Discord tool/watcher work?",
        )
        mr1_instance._record_conversation("user", "We should use Discord, not Slack.")
        mr1_instance._record_conversation("mr1", "Discord is the better default for your setup.")

        response = mr1_instance.step(
            f"Respond to {message.message_id} with what you think is right based on the conversation.",
        )

        assert response == f"sent clarification to {child.agent_id} for {message.message_id}"
        inbox = mr1_instance._message_store.list_inbox(child.agent_id)
        assert "Darwin" in inbox[0].body
        assert "Discord watcher" in inbox[0].body
        assert compiler.call_count == 0
        assert mock_process.send.call_count == 1

    def test_freeform_approval_routes_through_run_commands_without_brain(self, tmp_path):
        compiler = MagicMock()
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mock_process = MagicMock(spec=MR1Process)
        mock_process.alive = True
        mr1_instance._process = mock_process

        child = mr1_instance._scoped_agents.create_child_agent(mr1_instance._root_agent_id, "research")
        original_request = CapabilityRequest(
            actor_id=child.agent_id,
            actor_type="mrn",
            actor_clearance=child.security_clearance,
            invocation_mode="direct",
            capability_name="read_file",
            args={"path": "README.md"},
            scope=build_scope_context(
                actor_id=child.agent_id,
                workspace_root=tmp_path,
                scoped_agent_store=mr1_instance._scoped_agents,
            ),
            step_id=f"{child.agent_id}:1",
        )
        approval = CapabilityApprovalRequest(
            approval_request_id="cap_approval_test",
            requesting_actor_id=child.agent_id,
            capability_name="read_file",
            invocation_mode="direct",
            args={"path": "README.md"},
            risk_score=0.1,
            reason="outside_actor_scope",
            scope_summary=original_request.scope.to_dict(),
            original_request=original_request,
            original_step_id=f"{child.agent_id}:1",
            designated_approver_id=mr1_instance._root_agent_id,
            workflow_id="wf-1",
            task_id="tk-1",
        )
        mr1_instance._approval_store.save(approval)

        response = mr1_instance.step("approve cap_approval_test because approved")

        assert "Approved." in response
        assert mr1_instance._approval_store.require("cap_approval_test").status == "approved"
        assert compiler.call_count == 0
        assert mock_process.send.call_count == 0

    def test_meta_command_question_routes_direct_answer(self, mr1_with_mock_process):
        mr1_instance, mock_process = mr1_with_mock_process
        mock_process.send.return_value = "You would clarify exact agent ids before termination."

        result = mr1_instance.step("how would you kill all research agents")

        assert result == "You would clarify exact agent ids before termination."
        assert mock_process.send.call_count == 1


class TestRuntimeReferenceResolution:
    def _mr1_instance(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True
        return mr1_instance

    def test_resolve_runtime_references_exact_agent_id(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")

        resolution = mr1_instance.resolve_runtime_references(
            f"ask {child.agent_id} to continue",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["resolved_references"][child.agent_id]["id"] == child.agent_id

    def test_resolve_runtime_references_unique_mr2_title(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")

        resolution = mr1_instance.resolve_runtime_references(
            "ask MR2 to continue",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["resolved_references"]["MR2"]["id"] == child.agent_id

    def test_resolve_runtime_references_that_mr2_prefers_latest_referenced_match(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        older = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        newer = _seed_legacy_agent(mr1_instance, parent_agent_id=root.agent_id, title="MR2")
        mr1_instance._state.set_reference_state("last_created_agent_id", newer.agent_id)

        resolution = mr1_instance.resolve_runtime_references(
            "that MR2 should continue",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["resolved_references"]["that MR2"]["id"] == newer.agent_id
        assert older.agent_id != newer.agent_id

    def test_resolve_runtime_references_that_agent_uses_last_referenced_agent(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        mr1_instance._state.set_reference_state("last_referenced_agent_id", child.agent_id)

        resolution = mr1_instance.resolve_runtime_references(
            "that agent should continue",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["resolved_references"]["that agent"]["id"] == child.agent_id

    def test_resolve_runtime_references_agent_we_just_created_uses_last_created_agent(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        mr1_instance._state.set_reference_state("last_created_agent_id", child.agent_id)

        resolution = mr1_instance.resolve_runtime_references(
            "why did the agent we just created get blocked?",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["resolved_references"]["the agent we just created"]["id"] == child.agent_id

    def test_resolve_runtime_references_ambiguous_mr2_flags_clarification(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        _seed_legacy_agent(mr1_instance, parent_agent_id=root.agent_id, title="MR2")
        mr1_instance._state.set_reference_state("last_created_agent_id", None)
        mr1_instance._state.set_reference_state("last_referenced_agent_id", None)

        resolution = mr1_instance.resolve_runtime_references(
            "MR2 should continue",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["ambiguities"]
        assert resolution["ambiguities"][0]["reference"] == "MR2"

    def test_resolve_runtime_references_plain_research_group_has_no_agent_ambiguity(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "archive")

        resolution = mr1_instance.resolve_runtime_references(
            "I use Slack because I'm part of a research group.",
            mr1_instance.build_runtime_grounding(),
        )

        assert not resolution["resolved_references"]
        assert not resolution["ambiguities"]

    def test_resolve_runtime_references_bulk_research_agents_avoids_single_agent_ambiguity(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        older = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
        newer = _seed_legacy_agent(mr1_instance, parent_agent_id=root.agent_id, title="research")

        resolution = mr1_instance.resolve_runtime_references(
            "kill all research agents",
            mr1_instance.build_runtime_grounding(),
        )

        assert not resolution["ambiguities"]
        assert resolution["bulk_targets"]
        candidate_ids = {item["id"] for item in resolution["bulk_targets"][0]["candidates"]}
        assert candidate_ids == {older.agent_id, newer.agent_id}

    def test_resolve_runtime_references_bulk_agents_applies_exclusions(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        genesis = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Genesis")
        research = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")

        resolution = mr1_instance.resolve_runtime_references(
            "kill all agents except Genesis",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["bulk_targets"]
        candidate_ids = {item["id"] for item in resolution["bulk_targets"][0]["candidates"]}
        excluded_ids = {item["id"] for item in resolution["bulk_targets"][0]["excluded_candidates"]}
        assert candidate_ids == {research.agent_id}
        assert excluded_ids == {genesis.agent_id}

    def test_resolve_runtime_references_research_doing_still_flags_single_agent_ambiguity(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
        _seed_legacy_agent(mr1_instance, parent_agent_id=root.agent_id, title="research")

        resolution = mr1_instance.resolve_runtime_references(
            "what is research doing?",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["ambiguities"]
        assert resolution["ambiguities"][0]["reference"] == "research"

    def test_bulk_research_agent_kill_terminates_all_matching_agents(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        older = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
        newer = _seed_legacy_agent(mr1_instance, parent_agent_id=root.agent_id, title="research")

        response = mr1_instance.step("kill all research agents")

        assert "Ambiguous reference 'research'" not in response
        assert "Terminated 2 agent(s)" in response
        assert older.agent_id in response
        assert newer.agent_id in response

    def test_bulk_agent_kill_with_exclusion_preserves_named_agent(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        genesis = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Genesis")
        research = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")

        response = mr1_instance.step("kill all agents except Genesis")

        assert f"Excluded 1 agent(s): Genesis ({genesis.agent_id})." in response
        assert research.agent_id in response
        assert mr1_instance._scoped_agents.require_agent(genesis.agent_id).status == "active"
        assert mr1_instance._scoped_agents.require_agent(research.agent_id).status == "terminated"

    def test_agent_kill_all_builtin_supports_exclusions(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        genesis = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "Genesis")
        research = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")

        response = mr1_instance._handle_builtin("/agent kill-all all --exclude Genesis")

        assert f"Excluded 1 agent(s): Genesis ({genesis.agent_id})." in response
        assert research.agent_id in response
        assert mr1_instance._scoped_agents.require_agent(genesis.agent_id).status == "active"
        assert mr1_instance._scoped_agents.require_agent(research.agent_id).status == "terminated"

    def test_agent_block_question_routes_to_runtime_inspection_without_brain(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        child = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        child.run_status = "blocked"
        child.last_action = {"action": "invalid", "reason": "invalid message kind: proposal"}
        mr1_instance._scoped_agents.save_agent(child)
        mr1_instance._state.set_reference_state("last_created_agent_id", child.agent_id)

        response = mr1_instance.step("why did the agent we just created get blocked after i ran it?")

        assert f"agent_id:     {child.agent_id}" in response
        assert "run_status:   blocked" in response
        assert "invalid message kind: proposal" in response
        assert mr1_instance._process.send.call_count == 0

    def test_approvals_builtin_lists_shows_and_approves_visible_requests(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        child = mr1_instance._scoped_agents.create_child_agent(mr1_instance._root_agent_id, "research")
        original_request = CapabilityRequest(
            actor_id=child.agent_id,
            actor_type="mrn",
            actor_clearance=child.security_clearance,
            invocation_mode="direct",
            capability_name="read_file",
            args={"path": "README.md"},
            scope=build_scope_context(
                actor_id=child.agent_id,
                workspace_root=tmp_path,
                scoped_agent_store=mr1_instance._scoped_agents,
            ),
            step_id=f"{child.agent_id}:1",
        )
        approval = CapabilityApprovalRequest(
            approval_request_id="cap_approval_test",
            requesting_actor_id=child.agent_id,
            capability_name="read_file",
            invocation_mode="direct",
            args={"path": "README.md"},
            risk_score=0.1,
            reason="outside_actor_scope",
            scope_summary=original_request.scope.to_dict(),
            original_request=original_request,
            original_step_id=f"{child.agent_id}:1",
            designated_approver_id=mr1_instance._root_agent_id,
            workflow_id="wf-1",
            task_id="tk-1",
        )
        mr1_instance._approval_store.save(approval)

        bare_output = mr1_instance._handle_builtin("/approvals")
        list_output = mr1_instance._handle_builtin("/approvals list")
        show_output = mr1_instance._handle_builtin("/approvals show cap_approval_test")
        approve_output = mr1_instance._handle_builtin(
            "/approvals approve cap_approval_test --reason approved",
        )

        assert bare_output == list_output
        assert "cap_approval_test" in list_output
        assert "capability_name:      read_file" in show_output
        assert "Approved." in approve_output
        assert "Blocked workflow task reopened automatically." in approve_output
        assert "Use --grant-scope to persist scope access if intended." in approve_output
        updated = mr1_instance._approval_store.require("cap_approval_test")
        assert updated.status == "approved"
        assert updated.decision is not None
        inbox = mr1_instance._message_store.list_inbox(child.agent_id)
        assert inbox[0].subject == "Capability approval approved: read_file"

    def test_orchestrator_prompt_includes_mr2_ownership_guidance(self):
        assert "PERSISTENT DELEGATION / OWNERSHIP" in _ORCHESTRATOR_PROMPT
        assert '{"agent": "mr2"' in _ORCHESTRATOR_PROMPT
        assert "If the user wants an agent to own an area" in _ORCHESTRATOR_PROMPT
        assert "INTERNAL OBSERVATIONS" in _ORCHESTRATOR_PROMPT
        assert '[OBSERVE]' in _ORCHESTRATOR_PROMPT
        assert "Do not claim you lack access to full runtime detail" in _ORCHESTRATOR_PROMPT

    def test_workflow_request_uses_compiler_first_not_brain(self, tmp_path):
        compiler = FakeCompiler(
            json.dumps(
                {
                    "title": "Read and summarize",
                    "tasks": [
                        {
                            "label": "read_notes",
                            "title": "Read notes",
                            "task_kind": "tool",
                            "tool_type": "read_file",
                            "tool_config": {"path": str(tmp_path / "notes.txt")},
                        },
                        {
                            "label": "python_version",
                            "title": "Python version",
                            "task_kind": "tool",
                            "tool_type": "shell_command",
                            "tool_config": {"argv": ["python3", "--version"], "cwd": str(tmp_path)},
                        },
                        {
                            "label": "summarize",
                            "title": "Summarize",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "depends_on": ["read_notes", "python_version"],
                            "inputs": [
                                {"name": "notes", "from": "read_notes.result.text"},
                                {"name": "version", "from": "python_version.result.data.stdout"},
                            ],
                            "prompt": "Summarize the notes and version.",
                        },
                    ],
                }
            )
        )
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        response = mr1_instance.step("Read a file, check Python version, and summarize")

        assert "This workflow will:" in response
        assert mr1_instance._state.pending_workflow is not None
        assert mr1_instance._process.send.call_count == 0

    def test_preview_persistence_json_and_confirmation(self, tmp_path):
        compiler = FakeCompiler(
            json.dumps(
                {
                    "title": "Complex",
                    "tasks": [
                        {
                            "label": "read_notes",
                            "title": "Read notes",
                            "task_kind": "tool",
                            "tool_type": "read_file",
                            "tool_config": {"path": str(tmp_path / "notes.txt")},
                        },
                        {
                            "label": "python_version",
                            "title": "Python version",
                            "task_kind": "tool",
                            "tool_type": "shell_command",
                            "tool_config": {"argv": ["python3", "--version"], "cwd": str(tmp_path)},
                        },
                        {
                            "label": "summarize",
                            "title": "Summarize",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "depends_on": ["read_notes", "python_version"],
                            "inputs": [
                                {"name": "notes", "from": "read_notes.result.text"},
                                {"name": "version", "from": "python_version.result.data.stdout"},
                            ],
                            "prompt": "Summarize.",
                        },
                        {
                            "label": "write_report",
                            "title": "Write report",
                            "task_kind": "tool",
                            "tool_type": "write_file",
                            "depends_on": ["summarize"],
                            "inputs": [{"name": "summary", "from": "summarize.result.text"}],
                            "tool_config": {
                                "path": str(tmp_path / "report.txt"),
                                "content": "placeholder",
                                "overwrite": True,
                            },
                        },
                    ],
                }
            )
        )
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        preview = mr1_instance.step("Read a file, check Python version, summarize, and write a report")
        assert "This workflow will:" in preview
        assert mr1_instance._state.pending_workflow is not None

        raw_json = mr1_instance.step("show json")
        assert raw_json == json.dumps(mr1_instance._state.pending_workflow["spec"], indent=2)

        submit = mr1_instance.step("execute")
        assert submit.startswith("submitted workflow: wf-")
        assert mr1_instance._state.pending_workflow is None

    @pytest.mark.parametrize("confirmation_input", ["yes", "yes, submit it", "ok"])
    def test_simple_workflow_requires_preview_then_confirmation(self, tmp_path, confirmation_input):
        compiler = FakeCompiler(
            json.dumps(
                {
                    "title": "Simple",
                    "tasks": [
                        {
                            "label": "read_notes",
                            "title": "Read notes",
                            "task_kind": "tool",
                            "tool_type": "read_file",
                            "tool_config": {"path": str(tmp_path / "notes.txt")},
                        },
                        {
                            "label": "summarize",
                            "title": "Summarize",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "depends_on": ["read_notes"],
                            "inputs": [{"name": "notes", "from": "read_notes.result.text"}],
                            "prompt": "Summarize the notes.",
                        },
                    ],
                }
            )
        )
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        preview = mr1_instance.step("Read a file and summarize it")

        assert "This workflow will:" in preview
        assert mr1_instance._state.pending_workflow is not None
        assert mr1_instance._runtime_access.list_workflows(caller_agent_id=mr1_instance._root_agent_id) == []

        submit = mr1_instance.step(confirmation_input)

        assert submit.startswith("submitted workflow: wf-")
        assert mr1_instance._state.pending_workflow is None

    @pytest.mark.parametrize("cancellation_input", ["no", "never mind", "abort", "actually nevermind, cancel that"])
    def test_pending_workflow_cancellation_phrases_cancel_draft(self, tmp_path, cancellation_input):
        compiler = FakeCompiler(
            json.dumps(
                {
                    "title": "Simple",
                    "tasks": [
                        {
                            "label": "read_notes",
                            "title": "Read notes",
                            "task_kind": "tool",
                            "tool_type": "read_file",
                            "tool_config": {"path": str(tmp_path / "notes.txt")},
                        },
                        {
                            "label": "summarize",
                            "title": "Summarize",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "depends_on": ["read_notes"],
                            "inputs": [{"name": "notes", "from": "read_notes.result.text"}],
                            "prompt": "Summarize the notes.",
                        },
                    ],
                }
            )
        )
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        mr1_instance.step("Read a file and summarize it")
        cancelled = mr1_instance.step(cancellation_input)

        assert cancelled == "Cancelled pending workflow draft."
        assert mr1_instance._state.pending_workflow is None
        assert mr1_instance._runtime_access.list_workflows(caller_agent_id=mr1_instance._root_agent_id) == []

    def test_conflicting_create_then_destroy_request_clarifies_without_execution(self, tmp_path):
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=lambda *_: "{}",
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        response = mr1_instance.step("create three agents and then kill them all")

        assert "conflicting agent lifecycle actions" in response.lower()
        visible_agents = mr1_instance._runtime_access.list_agents(caller_agent_id=mr1_instance._root_agent_id)
        child_agents = [
            agent for agent in visible_agents if agent["agent_id"] != mr1_instance._root_agent_id
        ]
        assert child_agents == []

    def test_cancel_pending_preview(self, tmp_path):
        compiler = FakeCompiler(
            json.dumps(
                {
                    "title": "Complex",
                    "tasks": [
                        {
                            "label": "read_notes",
                            "title": "Read notes",
                            "task_kind": "tool",
                            "tool_type": "read_file",
                            "tool_config": {"path": str(tmp_path / "notes.txt")},
                        },
                        {
                            "label": "python_version",
                            "title": "Python version",
                            "task_kind": "tool",
                            "tool_type": "shell_command",
                            "tool_config": {"argv": ["python3", "--version"], "cwd": str(tmp_path)},
                        },
                        {
                            "label": "summarize",
                            "title": "Summarize",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "depends_on": ["read_notes", "python_version"],
                            "inputs": [
                                {"name": "notes", "from": "read_notes.result.text"},
                                {"name": "version", "from": "python_version.result.data.stdout"},
                            ],
                            "prompt": "Summarize.",
                        },
                        {
                            "label": "write_report",
                            "title": "Write report",
                            "task_kind": "tool",
                            "tool_type": "write_file",
                            "depends_on": ["summarize"],
                            "inputs": [{"name": "summary", "from": "summarize.result.text"}],
                            "tool_config": {
                                "path": str(tmp_path / "report.txt"),
                                "content": "placeholder",
                                "overwrite": True,
                            },
                        },
                    ],
                }
            )
        )
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        mr1_instance.step("Read a file, check Python version, summarize, and write a report")
        cancelled = mr1_instance.step("cancel")

        assert cancelled == "Cancelled pending workflow draft."
        assert mr1_instance._state.pending_workflow is None

    def test_submission_receipt_contains_inspection_commands(self, tmp_path):
        compiler = FakeCompiler(
            json.dumps(
                {
                    "title": "Read and summarize",
                    "tasks": [
                        {
                            "label": "read_notes",
                            "title": "Read notes",
                            "task_kind": "tool",
                            "tool_type": "read_file",
                            "tool_config": {"path": str(tmp_path / "notes.txt")},
                        },
                        {
                            "label": "python_version",
                            "title": "Python version",
                            "task_kind": "tool",
                            "tool_type": "shell_command",
                            "tool_config": {"argv": ["python3", "--version"], "cwd": str(tmp_path)},
                        },
                        {
                            "label": "summarize",
                            "title": "Summarize",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "depends_on": ["read_notes", "python_version"],
                            "inputs": [
                                {"name": "notes", "from": "read_notes.result.text"},
                                {"name": "version", "from": "python_version.result.data.stdout"},
                            ],
                            "prompt": "Summarize the notes and version.",
                        },
                    ],
                }
            )
        )
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        preview = mr1_instance.step("Read a file, check Python version, and summarize")
        response = mr1_instance.step("submit")

        assert "This workflow will:" in preview
        assert "/workflow wf-" in response
        assert "/jobs" in response
        assert "/events wf-" in response
        assert "/artifacts wf-" in response
        assert "/result tk-" in response
        assert "/inputs tk-" in response

    def test_compiler_agent_backend_uses_preview_and_auto_submit_gate(self, tmp_path):
        compiler = FakeCompiler(
            json.dumps(
                {
                    "preview": "Read the notes, then summarize them.",
                    "spec": {
                        "title": "Read and summarize",
                        "tasks": [
                            {
                                "label": "read_notes",
                                "title": "Read notes",
                                "task_kind": "tool",
                                "tool_type": "read_file",
                                "tool_config": {"path": str(tmp_path / "notes.txt")},
                            },
                            {
                                "label": "summarize",
                                "title": "Summarize",
                                "task_kind": "agent",
                                "agent_type": "kazi",
                                "depends_on": ["read_notes"],
                                "inputs": [
                                    {"name": "notes", "from": "read_notes.result.text"},
                                ],
                                "prompt": "Summarize the notes.",
                            },
                        ],
                    },
                    "assumptions": [],
                    "risks": [],
                    "needs_confirmation": True,
                    "confidence": "high",
                    "memory_refs_used": [],
                }
            )
        )
        mr1_instance = MR1(
            workflow_store=WorkflowStore(root=tmp_path / "workflows"),
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
            workflow_authoring_backend="compiler_agent",
            workflow_compiler=compiler,
        )
        mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
        mr1_instance._process = MagicMock(spec=MR1Process)
        mr1_instance._process.alive = True

        response = mr1_instance.step("Read a file and summarize it")

        assert response.startswith("Read the notes, then summarize them.")
        assert mr1_instance._state.pending_workflow is not None
        assert mr1_instance._process.send.call_count == 0

    def test_explicit_agent_id_bypasses_title_ambiguity(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        agent_a = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        agent_b = _seed_legacy_agent(mr1_instance, parent_agent_id=root.agent_id, title="MR2")

        resolution = mr1_instance.resolve_runtime_references(
            f"message {agent_a.agent_id} to report status",
            mr1_instance.build_runtime_grounding(),
        )

        assert not resolution["ambiguities"]
        assert resolution["resolved_references"][agent_a.agent_id]["id"] == agent_a.agent_id
        assert agent_b.agent_id not in resolution["resolved_references"]

    def test_single_active_with_terminated_duplicate_resolves_directly(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        alive = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        dead = _seed_legacy_agent(
            mr1_instance,
            parent_agent_id=root.agent_id,
            title="MR2",
            status="terminated",
            run_status="terminated",
        )

        resolution = mr1_instance.resolve_runtime_references(
            "what is MR2 doing?",
            mr1_instance.build_runtime_grounding(),
        )

        assert not resolution["ambiguities"]
        assert resolution["resolved_references"]["MR2"]["id"] == alive.agent_id

    def test_single_active_with_legacy_active_terminated_duplicate_resolves_directly(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        alive = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        legacy_dead = _seed_legacy_agent(
            mr1_instance,
            parent_agent_id=root.agent_id,
            title="MR2",
            run_status="terminated",
        )

        resolution = mr1_instance.resolve_runtime_references(
            "what is MR2 doing?",
            mr1_instance.build_runtime_grounding(),
        )

        assert not resolution["ambiguities"]
        assert resolution["resolved_references"]["MR2"]["id"] == alive.agent_id

    def test_multiple_active_same_title_asks_clarification(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        _seed_legacy_agent(mr1_instance, parent_agent_id=root.agent_id, title="MR2")
        mr1_instance._state.set_reference_state("last_created_agent_id", None)
        mr1_instance._state.set_reference_state("last_referenced_agent_id", None)

        resolution = mr1_instance.resolve_runtime_references(
            "kill MR2",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["ambiguities"]
        assert resolution["ambiguities"][0]["reference"] == "MR2"

    def test_default_child_title_matches_tree_level(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        mr1_instance._scoped_agents.ensure_root_agent()

        # MRn denotes tree level, not a unique sequence — all level-2 children default to "MR2"
        title_a = mr1_instance._extract_requested_child_title("create a persistent agent to own discord")
        title_b = mr1_instance._extract_requested_child_title("create a persistent agent to own github")
        title_c = mr1_instance._extract_requested_child_title("create a persistent agent to own reporting")

        assert title_a == "MR2"
        assert title_b == "MR2"
        assert title_c == "MR2"
