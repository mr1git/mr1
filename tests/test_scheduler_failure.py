"""
When a parent task fails, its descendants must transition to `blocked`
with `blocked_reason`, `blocked_by`, and `blocked_at` populated.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from mr1 import workflow_events as ev
from mr1.core import Logger
from mr1.worker_runner import WorkerAsyncRunner, MockRunner, RunHandle, RunResult, RunStatus
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus, WorkflowStatus
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "Failure cascade",
    "tasks": [
        {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "worker", "prompt": "x"},
        {"label": "b", "title": "B", "task_kind": "agent", "agent_type": "worker",
         "prompt": "x", "depends_on": ["a"]},
        {"label": "c", "title": "C", "task_kind": "agent", "agent_type": "worker",
         "prompt": "x", "depends_on": ["a"]},
    ],
}


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def runner():
    return MockRunner()


@pytest.fixture
def scheduler(store, runner):
    s = Scheduler(store, runner, auto_tick=False)
    yield s
    s.shutdown()


def _load_attempt_result(store: WorkflowStore, workflow_id: str, task_id: str, attempt_id: int = 1) -> dict:
    path = store.task_attempt_result_path(workflow_id, task_id, attempt_id)
    return json.loads(path.read_text(encoding="utf-8"))


class TestFailureCascade:
    def test_failed_parent_blocks_both_dependents(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]
        c_id = wf.label_to_task_id["c"]

        runner.complete(a_id, RunStatus.FAILED, error="boom")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.FAILED
        assert wf.tasks[b_id].status is TaskStatus.BLOCKED
        assert wf.tasks[c_id].status is TaskStatus.BLOCKED

        b = wf.tasks[b_id]
        assert b.blocked_by == [a_id]
        assert "failed" in (b.blocked_reason or "")
        assert b.blocked_at is not None

    def test_timed_out_parent_also_blocks(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]

        runner.complete(a_id, RunStatus.TIMED_OUT, error="timeout")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.TIMED_OUT
        assert wf.tasks[b_id].status is TaskStatus.BLOCKED
        assert "timed_out" in (wf.tasks[b_id].blocked_reason or "")

    def test_workflow_marked_failed(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        runner.complete(wf.label_to_task_id["a"], RunStatus.FAILED, error="boom")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.FAILED
        assert wf.finished_at is not None

    def test_blocked_event_emitted(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        runner.complete(wf.label_to_task_id["a"], RunStatus.FAILED, error="boom")
        scheduler.tick()

        events = store.load_events(wf_id)
        types = [e.event_type for e in events]
        assert ev.TASK_FAILED in types
        assert types.count(ev.TASK_BLOCKED) == 2
        assert ev.WORKFLOW_FAILED == types[-1]

    def test_cancel_workflow_marks_live_tasks_cancelled(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        ok = scheduler.cancel_workflow(wf_id)
        assert ok

        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.CANCELLED
        for t in wf.tasks.values():
            assert t.is_terminal()

    def test_task_finalization_exception_is_persisted_as_failed(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]

        runner.complete(a_id, RunStatus.SUCCEEDED, summary="ok", result_payload={"summary": "ok"})
        with patch("mr1.scheduler.build_agent_task_output", side_effect=RuntimeError("boom")):
            scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = wf.tasks[a_id]
        assert task.status is TaskStatus.FAILED
        assert "task finalization failed: RuntimeError: boom" == task.last_error
        assert task.last_error_type == "internal_error"

    def test_kill_agents_semantic_mismatch_fails_when_output_is_unrelated(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = MockRunner()
        scheduler = Scheduler(store, runner, auto_tick=False)
        root = scheduler._scoped_agents.ensure_root_agent()
        child = scheduler._scoped_agents.create_child_agent(root.agent_id, "MR2")
        spec = {
            "title": "Kill all running agents",
            "tasks": [
                {
                    "label": "kill_agents",
                    "title": "Kill agents",
                    "task_kind": "agent",
                    "agent_type": "worker",
                    "prompt": f"Terminate orchestrator {child.agent_id}.",
                },
            ],
        }
        try:
            wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            task_id = wf.label_to_task_id["kill_agents"]
            runner.complete(
                task_id,
                RunStatus.SUCCEEDED,
                summary="Checked background jobs.",
                result_payload={
                    "summary": "Checked background jobs.",
                    "text": "I checked cron jobs and background services. Which agent should I terminate?",
                },
            )
            scheduler.tick()

            wf = store.load_workflow(wf_id)
            task = wf.tasks[task_id]
            assert task.status is TaskStatus.FAILED
            assert task.last_error_type == "semantic_mismatch"
            assert _load_attempt_result(store, wf_id, task_id)["failure_type"] == "semantic_mismatch"
            assert wf.status is WorkflowStatus.FAILED
        finally:
            scheduler.shutdown()

    def test_kill_agents_semantic_validation_passes_when_target_is_terminated(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = MockRunner()
        scheduler = Scheduler(store, runner, auto_tick=False)
        root = scheduler._scoped_agents.ensure_root_agent()
        child = scheduler._scoped_agents.create_child_agent(root.agent_id, "MR2")
        spec = {
            "title": "Kill all running agents",
            "tasks": [
                {
                    "label": "kill_agents",
                    "title": "Kill agents",
                    "task_kind": "agent",
                    "agent_type": "worker",
                    "prompt": f"Terminate orchestrator {child.agent_id}.",
                },
            ],
        }
        try:
            wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            task_id = wf.label_to_task_id["kill_agents"]
            scheduler._scoped_agents.terminate_agent(root.agent_id, child.agent_id)
            runner.complete(
                task_id,
                RunStatus.SUCCEEDED,
                summary=f"Terminated orchestrator {child.agent_id}.",
                result_payload={
                    "summary": f"Terminated orchestrator {child.agent_id}.",
                    "text": f"Terminated orchestrator {child.agent_id}.",
                },
            )
            scheduler.tick()

            wf = store.load_workflow(wf_id)
            task = wf.tasks[task_id]
            assert task.status is TaskStatus.SUCCEEDED
            assert task.last_error_type is None
            assert wf.status is WorkflowStatus.SUCCEEDED
        finally:
            scheduler.shutdown()

    def test_run_tests_semantic_mismatch_fails_without_test_evidence(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = MockRunner()
        scheduler = Scheduler(store, runner, auto_tick=False)
        spec = {
            "title": "Run tests",
            "tasks": [
                {
                    "label": "run_tests",
                    "title": "Run tests",
                    "task_kind": "agent",
                    "agent_type": "worker",
                    "prompt": "Run pytest -q and report the result.",
                },
            ],
        }
        try:
            wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            task_id = wf.label_to_task_id["run_tests"]
            runner.complete(
                task_id,
                RunStatus.SUCCEEDED,
                summary="Reviewed the repository.",
                result_payload={
                    "summary": "Reviewed the repository.",
                    "text": "I reviewed the repository structure and may need clarification before changing anything.",
                },
            )
            scheduler.tick()

            wf = store.load_workflow(wf_id)
            task = wf.tasks[task_id]
            assert task.status is TaskStatus.FAILED
            assert task.last_error_type == "semantic_mismatch"
            assert wf.status is WorkflowStatus.FAILED
        finally:
            scheduler.shutdown()


class TestHandleLoss:
    """Regression tests for the run handle lost bug."""

    def test_handle_stored_atomically_with_running_status(self, tmp_path):
        """After _begin_attempt, handle is present iff task is RUNNING on disk."""
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = MockRunner()
        scheduler = Scheduler(store, runner, auto_tick=False)
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]

        # Task must be RUNNING on disk.
        assert wf.tasks[a_id].status is TaskStatus.RUNNING
        # Handle must be present in the scheduler's in-memory map.
        assert a_id in scheduler._handles
        # Invariant: every handle in _handles belongs to a RUNNING task.
        for task in wf.tasks.values():
            if task.task_id in scheduler._handles:
                assert task.status is TaskStatus.RUNNING

        scheduler.shutdown()

    def test_scheduler_restart_fails_running_task_deterministically(self, tmp_path):
        """New scheduler instance sees a RUNNING task with no handle and fails it."""
        store = WorkflowStore(root=tmp_path / "workflows")
        runner1 = MockRunner()
        sched1 = Scheduler(store, runner1, auto_tick=False)
        wf_id = sched1.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        sched1.tick()  # task a starts
        sched1.shutdown()  # no cancel_running → handle orphaned

        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        assert wf.tasks[a_id].status is TaskStatus.RUNNING

        # Simulate restart: new scheduler, empty _handles.
        sched2 = Scheduler(store, MockRunner(), auto_tick=False)
        sched2.tick()
        sched2.shutdown()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.FAILED
        assert "run handle lost" in (wf.tasks[a_id].last_error or "")
        assert wf.tasks[a_id].last_error_type == "infrastructure_failure"
        assert wf.is_terminal()

    def test_scheduler_restart_recovers_completed_result_via_mock(self, tmp_path):
        """Recovery via MockRunner.set_recoverable avoids the failure path."""
        single_task_spec = {
            "title": "single",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "worker", "prompt": "x"},
            ],
        }
        store = WorkflowStore(root=tmp_path / "workflows")
        logger = Logger(tasks_dir=str(tmp_path / "task-logs"))
        runner1 = MockRunner()
        sched1 = Scheduler(store, runner1, auto_tick=False, logger=logger)
        wf_id = sched1.submit_workflow(single_task_spec, Provenance(type="agent", id="MR1"))
        sched1.tick()  # task a starts, log paths set on disk
        sched1.shutdown()

        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        assert wf.tasks[a_id].status is TaskStatus.RUNNING

        # Stage a recoverable result on the new runner.
        runner2 = MockRunner()
        runner2.set_recoverable(
            a_id,
            RunResult(
                status=RunStatus.SUCCEEDED,
                exit_code=0,
                summary="recovered via file",
                result_payload={"status": "succeeded", "summary": "recovered via file"},
            ),
        )
        sched2 = Scheduler(store, runner2, auto_tick=False, logger=logger)
        sched2.tick()
        sched2.shutdown()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.SUCCEEDED
        assert "run handle lost" not in (wf.tasks[a_id].last_error or "")
        assert wf.is_terminal()
        actions = [entry["action"] for entry in logger.read_logs(a_id, "scheduler")]
        assert "handle_missing" in actions
        assert "handle_recovered" in actions

    def test_worker_async_runner_recover_result_reads_stdout_log(self, tmp_path):
        """WorkerAsyncRunner.recover_result parses the stdout log when present."""
        from pathlib import Path
        from mr1.worker_runner import WorkerAsyncRunner
        from mr1.workflow_models import Task, TaskStatus

        store = WorkflowStore(root=tmp_path / "workflows")
        runner = WorkerAsyncRunner(store)

        # Build a minimal Task with a stdout log path.
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        stdout_log = log_dir / "stdout.log"
        stdout_log.write_text(
            json.dumps({"result": "done", "is_error": False, "usage": {}}),
            encoding="utf-8",
        )

        task = Task(
            task_id="t1",
            workflow_id="wf1",
            label="a",
            title="A",
            task_kind="agent",
            agent_type="worker",
            prompt="",
            log_stdout_path=str(stdout_log),
        )

        result = runner.recover_result(task)
        assert result is not None
        assert result.status is RunStatus.SUCCEEDED
        assert result.summary == "done"

    def test_worker_async_runner_recover_result_returns_none_for_empty_log(self, tmp_path):
        """WorkerAsyncRunner.recover_result returns None when log is absent or empty."""
        from mr1.worker_runner import WorkerAsyncRunner
        from mr1.workflow_models import Task

        store = WorkflowStore(root=tmp_path / "workflows")
        runner = WorkerAsyncRunner(store)

        _kw = dict(title="A", task_kind="agent", agent_type="worker", prompt="")

        # No log path set.
        assert runner.recover_result(Task(task_id="t1", workflow_id="wf1", label="a", **_kw)) is None

        # Log path set but file does not exist.
        assert runner.recover_result(Task(
            task_id="t2", workflow_id="wf1", label="b",
            log_stdout_path=str(tmp_path / "no_such_file.log"), **_kw,
        )) is None

        # Log exists but is empty.
        empty_log = tmp_path / "empty.log"
        empty_log.write_text("", encoding="utf-8")
        assert runner.recover_result(Task(
            task_id="t3", workflow_id="wf1", label="c",
            log_stdout_path=str(empty_log), **_kw,
        )) is None

        # Log exists but contains invalid JSON.
        bad_log = tmp_path / "bad.log"
        bad_log.write_text("not json", encoding="utf-8")
        assert runner.recover_result(Task(
            task_id="t4", workflow_id="wf1", label="d",
            log_stdout_path=str(bad_log), **_kw,
        )) is None

    def test_worker_task_completes_without_handle_loss(self, tmp_path):
        """Worker agent task started and completed on the same scheduler has no handle loss."""
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = MockRunner()
        scheduler = Scheduler(store, runner, auto_tick=False)
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))

        scheduler.tick()  # task a starts
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        assert wf.tasks[a_id].status is TaskStatus.RUNNING
        assert a_id in scheduler._handles

        runner.complete(a_id, RunStatus.SUCCEEDED, summary="done")
        scheduler.tick()  # task a finishes

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.SUCCEEDED
        assert "run handle lost" not in (wf.tasks[a_id].last_error or "")
        assert a_id not in scheduler._handles  # handle cleaned up after finalization
        scheduler.shutdown()

    def test_recoverable_result_staged_via_mock_runner(self, tmp_path):
        """MockRunner.set_recoverable lets tests exercise the recovery code path."""
        store = WorkflowStore(root=tmp_path / "workflows")
        runner1 = MockRunner()
        sched1 = Scheduler(store, runner1, auto_tick=False)
        wf_id = sched1.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        sched1.tick()  # task a starts
        sched1.shutdown()

        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]

        # Stage a recoverable result on runner2 (no disk file needed).
        runner2 = MockRunner()
        runner2.set_recoverable(
            a_id,
            RunResult(
                status=RunStatus.SUCCEEDED,
                exit_code=0,
                summary="mock recovered",
                result_payload={"status": "succeeded", "summary": "mock recovered"},
            ),
        )
        sched2 = Scheduler(store, runner2, auto_tick=False)
        sched2.tick()
        sched2.shutdown()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.SUCCEEDED
        assert wf.tasks[a_id].result_summary == "mock recovered"

    @patch("mr1.worker_runner.subprocess.Popen")
    def test_worker_fast_exit_finalized_in_launch_tick(self, mock_popen, tmp_path):
        """A fast-exiting real Worker subprocess is finalized on the launch tick."""
        store = WorkflowStore(root=tmp_path / "workflows")
        logger = Logger(tasks_dir=str(tmp_path / "task-logs"))
        runner = WorkerAsyncRunner(store, logger=logger)
        scheduler = Scheduler(store, runner, auto_tick=False, logger=logger)
        single_task_spec = {
            "title": "single",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "worker", "prompt": "x"},
            ],
        }

        def _fake_popen(*_args, stdout=None, stderr=None, **_kwargs):
            assert stdout is not None
            stdout.write(
                json.dumps({"result": "fast done", "is_error": False, "usage": {"output_tokens": 1}}).encode("utf-8")
            )
            stdout.flush()
            if stderr is not None:
                stderr.flush()
            proc = MagicMock()
            proc.pid = 4242
            proc.poll.return_value = 0
            proc.returncode = 0
            return proc

        mock_popen.side_effect = _fake_popen

        wf_id = scheduler.submit_workflow(single_task_spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        task = wf.tasks[a_id]
        assert task.status is TaskStatus.SUCCEEDED
        assert a_id not in scheduler._handles
        assert "run handle lost" not in (task.last_error or "")
        assert task.result_path is not None
        result_payload = store.read_result(wf_id, a_id)
        assert result_payload is not None
        assert result_payload["summary"] == "fast done"

        scheduler_actions = [entry["action"] for entry in logger.read_logs(a_id, "scheduler")]
        assert "launch_request" in scheduler_actions
        assert "handle_registered" in scheduler_actions
        assert "immediate_poll_terminal" in scheduler_actions
        assert "handle_removed" in scheduler_actions
        assert "handle_missing" not in scheduler_actions

        worker_actions = [entry["action"] for entry in logger.read_logs(a_id, "worker")]
        assert "spawn" in worker_actions
        assert "exit" in worker_actions
        scheduler.shutdown()

    @patch("mr1.worker_runner.subprocess.Popen")
    def test_fast_exit_worker_result_survives_restart_without_handle_loss(self, mock_popen, tmp_path):
        """A fast-exit task persists its terminal result before any restart can lose the handle."""
        store = WorkflowStore(root=tmp_path / "workflows")
        logger = Logger(tasks_dir=str(tmp_path / "task-logs"))
        single_task_spec = {
            "title": "single",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "worker", "prompt": "x"},
            ],
        }

        def _fake_popen(*_args, stdout=None, stderr=None, **_kwargs):
            assert stdout is not None
            stdout.write(
                json.dumps({"result": "restart safe", "is_error": False, "usage": {"output_tokens": 1}}).encode("utf-8")
            )
            stdout.flush()
            if stderr is not None:
                stderr.flush()
            proc = MagicMock()
            proc.pid = 4343
            proc.poll.return_value = 0
            proc.returncode = 0
            return proc

        mock_popen.side_effect = _fake_popen

        runner1 = WorkerAsyncRunner(store, logger=logger)
        sched1 = Scheduler(store, runner1, auto_tick=False, logger=logger)
        wf_id = sched1.submit_workflow(single_task_spec, Provenance(type="agent", id="MR1"))
        sched1.tick()
        sched1.shutdown()

        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        assert wf.tasks[a_id].status is TaskStatus.SUCCEEDED

        sched2 = Scheduler(store, MockRunner(), auto_tick=False, logger=logger)
        sched2.tick()
        sched2.shutdown()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.SUCCEEDED
        actions = [entry["action"] for entry in logger.read_logs(a_id, "scheduler")]
        assert "immediate_poll_terminal" in actions
        assert "handle_missing" not in actions
