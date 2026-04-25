from __future__ import annotations

from pathlib import Path

import pytest

from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "Attempt tracking",
    "tasks": [
        {
            "label": "a",
            "title": "A",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "x",
        }
    ],
}


@pytest.fixture
def store(tmp_path: Path) -> WorkflowStore:
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def runner() -> MockRunner:
    return MockRunner()


@pytest.fixture
def scheduler(store: WorkflowStore, runner: MockRunner) -> Scheduler:
    sched = Scheduler(store, runner, auto_tick=False)
    yield sched
    sched.shutdown()


class TestAttempts:
    def test_attempt_ids_are_monotonic_and_match_directory_names(
        self,
        scheduler: Scheduler,
        store: WorkflowStore,
        runner: MockRunner,
    ) -> None:
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = wf.task_by_label("a")
        assert task.status is TaskStatus.RUNNING
        assert task.attempt_count == 1
        assert task.current_attempt == 1
        assert [attempt.attempt_id for attempt in task.attempts] == [1]
        assert "/attempts/1/stdout.log" in (task.log_stdout_path or "")
        assert "/attempts/1/stderr.log" in (task.log_stderr_path or "")

        runner.complete(task.task_id, RunStatus.SUCCEEDED, summary="first")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = wf.task_by_label("a")
        first_attempt = task.attempts[0]
        assert first_attempt.status is TaskStatus.SUCCEEDED
        assert first_attempt.result_path is not None
        assert Path(first_attempt.result_path).name == "result.json"
        assert Path(first_attempt.result_path).parent.name == "1"
        assert Path(first_attempt.result_path).exists()

        scheduler.rerun_task(wf_id, "a")
        wf = store.load_workflow(wf_id)
        task = wf.task_by_label("a")
        assert task.attempt_count == 1
        assert task.status is TaskStatus.READY

        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = wf.task_by_label("a")
        assert task.status is TaskStatus.RUNNING
        assert task.attempt_count == 2
        assert task.current_attempt == 2
        assert [attempt.attempt_id for attempt in task.attempts] == [1, 2]
        assert Path(task.log_stdout_path).parent.name == "2"

        runner.complete(task.task_id, RunStatus.SUCCEEDED, summary="second")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = wf.task_by_label("a")
        assert task.status is TaskStatus.SUCCEEDED
        assert task.attempt_count == 2
        assert len(task.attempts) == 2
        assert [attempt.attempt_id for attempt in task.attempts] == [1, 2]
        assert task.attempts[1].status is TaskStatus.SUCCEEDED
        assert Path(task.attempts[1].result_path).parent.name == "2"

    def test_rerun_and_cancel_without_launch_do_not_consume_attempt_ids(
        self,
        scheduler: Scheduler,
        store: WorkflowStore,
        runner: MockRunner,
    ) -> None:
        spec = {
            "title": "No-gap attempts",
            "tasks": [
                {
                    "label": "a",
                    "title": "A",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "x",
                },
                {
                    "label": "b",
                    "title": "B",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "y",
                    "depends_on": ["a"],
                },
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]

        runner.complete(a_id, RunStatus.FAILED, error="boom")
        scheduler.tick()
        scheduler.rerun_task(wf_id, "a")
        scheduler.cancel_task(b_id)

        wf = store.load_workflow(wf_id)
        a = wf.task_by_label("a")
        b = wf.task_by_label("b")
        assert a.attempt_count == 1
        assert b.attempt_count == 0
        assert b.status is TaskStatus.CANCELLED

        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a = wf.task_by_label("a")
        assert a.attempt_count == 2
        assert [attempt.attempt_id for attempt in a.attempts] == [1, 2]
