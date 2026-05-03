"""Tests for mr1.mr1 — orchestrator logic (no subprocess spawning)."""

import json
import tempfile
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
from mr1.workflow_models import TaskStatus, WorkflowStatus
from mr1.mr1 import (
    MR1,
    MR1Process,
    StateManager,
    _ORCHESTRATOR_PROMPT,
    _load_agent_config,
    _generate_task_id,
)
from mr1.kazi_runner import MockRunner
from mr1.mrn_run import MRnRunResult
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

    def test_agent_builtin_shows_mission_and_iteration(self, mr1_instance):
        child = mr1_instance._scoped_agents.create_child_agent(mr1_instance._root_agent_id, "research")
        child.mission = "Investigate the repo"
        child.run_status = "working"
        child.current_iteration = 2
        mr1_instance._scoped_agents.save_agent(child)

        result = mr1_instance._handle_builtin(f"/agent {child.agent_id}")

        assert "mission:      Investigate the repo" in result
        assert "run_status:   working" in result
        assert "iteration:    2" in result

    def test_agent_builtin_shows_runtime_activity_clarity(self, mr1_instance, tmp_path):
        child = mr1_instance._scoped_agents.create_child_agent(mr1_instance._root_agent_id, "research")
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
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
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
        assert '"agents": [' in sent
        assert '"workflows": [' in sent
        assert '"messages": [' in sent
        assert '"approvals": [' in sent
        assert '"events": [' in sent
        turn_dir = tmp_path / "mr1_turns"
        artifacts = sorted(turn_dir.glob("turn-*.json"))
        assert len(artifacts) == 1
        artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert artifact["route"] == "direct_answer"
        assert artifact["runtime_grounding"]["agents"]
        assert artifact["runtime_grounding"]["workflows"]
        assert artifact["runtime_grounding"]["messages"]
        assert artifact["runtime_grounding"]["approvals"]
        assert artifact["runtime_grounding"]["events"]
        assert artifact["brain_prompt"] == sent

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
        assert request in (child.mission or "")
        assert "Own the requested domain/responsibility" in (child.mission or "")
        assert "do not create another child agent" in (child.mission or "")
        assert "Prefer creating workflows for execution when appropriate." in (child.mission or "")
        assert "Keep responsibility for proposal quality, safety review, creation, and testing." in (child.mission or "")
        assert "Escalate to MR1/user when clarification or confirmation is needed." in (child.mission or "")
        assert policy.max_steps == 3
        assert policy.max_workflows_created == 2
        assert policy.require_confirmation_for_workflows is True
        assert compiler.call_count == 0
        assert mr1_instance._process.send.call_count == 0

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
        assert "persistent MR3-style child agent" in (child.mission or "")

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
        newer = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
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

    def test_resolve_runtime_references_ambiguous_mr2_flags_clarification(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "MR2")
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
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")

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
        newer = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")

        resolution = mr1_instance.resolve_runtime_references(
            "kill all research agents",
            mr1_instance.build_runtime_grounding(),
        )

        assert not resolution["ambiguities"]
        assert resolution["bulk_targets"]
        candidate_ids = {item["id"] for item in resolution["bulk_targets"][0]["candidates"]}
        assert candidate_ids == {older.agent_id, newer.agent_id}

    def test_resolve_runtime_references_research_doing_still_flags_single_agent_ambiguity(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
        mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")

        resolution = mr1_instance.resolve_runtime_references(
            "what is research doing?",
            mr1_instance.build_runtime_grounding(),
        )

        assert resolution["ambiguities"]
        assert resolution["ambiguities"][0]["reference"] == "research"

    def test_bulk_research_agent_kill_routes_to_bulk_clarification(self, tmp_path):
        mr1_instance = self._mr1_instance(tmp_path)
        root = mr1_instance._scoped_agents.ensure_root_agent()
        older = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")
        newer = mr1_instance._scoped_agents.create_child_agent(root.agent_id, "research")

        response = mr1_instance.step("kill all research agents")

        assert "Ambiguous reference 'research'" not in response
        assert "Confirm the exact agent_ids you want me to terminate" in response
        assert older.agent_id in response
        assert newer.agent_id in response
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

        list_output = mr1_instance._handle_builtin("/approvals list")
        show_output = mr1_instance._handle_builtin("/approvals show cap_approval_test")
        approve_output = mr1_instance._handle_builtin(
            "/approvals approve cap_approval_test --reason approved",
        )

        assert "cap_approval_test" in list_output
        assert "capability_name:      read_file" in show_output
        assert "Approved." in approve_output
        assert "/workflow rerun wf-1 tk-1" in approve_output
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

        assert response.startswith("submitted workflow: wf-")
        assert "/workflow wf-" in response
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

        response = mr1_instance.step("Read a file, check Python version, and summarize")

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
