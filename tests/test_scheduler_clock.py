"""A0 — the scheduler reads time through the injected clock."""

from __future__ import annotations

from datetime import datetime, timezone

from mr1.clock import VirtualClock
from mr1.kazi_runner import MockRunner, RunResult, RunStatus
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


def _spec() -> dict:
    return {
        "title": "clock seam",
        "tasks": [{"label": "one", "prompt": "do the thing"}],
    }


def _always_succeeds(_handle) -> RunResult:
    return RunResult(status=RunStatus.SUCCEEDED, exit_code=0, summary="ok")


def _scheduler(tmp_path, clock):
    store = WorkflowStore(root=tmp_path / "workflows")
    return Scheduler(
        store,
        MockRunner(on_poll=_always_succeeds),
        auto_tick=False,
        clock=clock,
        workspace_root=tmp_path,
    ), store


def test_task_timestamps_come_from_the_injected_clock(tmp_path):
    clock = VirtualClock(start=datetime(2030, 6, 1, tzinfo=timezone.utc))
    scheduler, store = _scheduler(tmp_path, clock)
    workflow_id = scheduler.submit_workflow(_spec(), Provenance(type="user", id="test"))

    scheduler.tick()
    scheduler.tick()

    workflow = store.load_workflow(workflow_id)
    task = next(iter(workflow.tasks.values()))
    assert task.status is TaskStatus.SUCCEEDED
    assert task.started_at.startswith("2030-06-01")
    assert task.finished_at.startswith("2030-06-01")
    assert workflow.finished_at.startswith("2030-06-01")


def test_advancing_the_clock_moves_recorded_timestamps(tmp_path):
    clock = VirtualClock(start=datetime(2030, 6, 1, tzinfo=timezone.utc))
    store = WorkflowStore(root=tmp_path / "workflows")
    runner = MockRunner()
    scheduler = Scheduler(
        store,
        runner,
        auto_tick=False,
        clock=clock,
        workspace_root=tmp_path,
    )
    workflow_id = scheduler.submit_workflow(_spec(), Provenance(type="user", id="test"))

    scheduler.tick()
    scheduler.tick()
    task_id = next(iter(store.load_workflow(workflow_id).tasks))
    clock.advance(86_400)
    runner.complete(task_id, RunStatus.SUCCEEDED, exit_code=0, summary="ok")
    scheduler.tick()

    workflow = store.load_workflow(workflow_id)
    task = workflow.tasks[task_id]
    assert task.status is TaskStatus.SUCCEEDED
    assert task.started_at.startswith("2030-06-01")
    assert task.finished_at.startswith("2030-06-02")


def test_default_scheduler_still_uses_real_time(tmp_path):
    store = WorkflowStore(root=tmp_path / "workflows")
    scheduler = Scheduler(
        store,
        MockRunner(on_poll=_always_succeeds),
        auto_tick=False,
        workspace_root=tmp_path,
    )
    workflow_id = scheduler.submit_workflow(_spec(), Provenance(type="user", id="test"))

    scheduler.tick()
    scheduler.tick()

    workflow = store.load_workflow(workflow_id)
    task = next(iter(workflow.tasks.values()))
    started = datetime.fromisoformat(task.started_at)
    assert abs((datetime.now(timezone.utc) - started).total_seconds()) < 60
