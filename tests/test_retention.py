"""
B1 — retention, archival, and full-history correctness.

Two defects and one absence:

  * `list_events()` read from a 50 000-event cache and silently returned a
    window as though it were history.
  * `events.jsonl`, workflow directories, audit records, and snapshots grew
    without bound and nothing ever reclaimed them.
  * There was no way to see what cleanup *would* do before it did it.

The rules these tests hold the implementation to: history queries are complete
or they fail loudly; nothing live is ever archived; nothing is deleted merely
for being old; and every run is auditable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mr1.autonomy.retention import RetentionManager, RetentionPolicy
from mr1.clock import VirtualClock
from mr1.event_archive import EventArchive, EventArchiveError
from mr1.event_log import EventLog
from mr1.worker_runner import MockRunner, RunStatus
from mr1.scheduler import Scheduler
from mr1.scoped_agents import AgentStore
from mr1.workflow_models import Provenance, WorkflowStatus
from mr1.workflow_store import WorkflowStore


CREATED_BY = Provenance(type="user", id="test")

SPEC = {
    "title": "Retention subject",
    "tasks": [
        {
            "label": "only",
            "title": "One task",
            "task_kind": "agent",
            "agent_type": "worker",
            "prompt": "do it",
        }
    ],
}


def _clock(days_in: int = 0) -> VirtualClock:
    return VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=days_in))


def _emit(log: EventLog, n: int, *, start: int = 0) -> None:
    for i in range(start, start + n):
        log.emit(
            event_type="workflow_created",
            actor_id="MR1",
            actor_type="root_orchestrator",
            target_id=f"wf-{i}",
            target_type="workflow",
            status="pending",
            summary=f"workflow {i}",
            workflow_id=f"wf-{i}",
        )


# ---------------------------------------------------------------------------
# 1. Event history — never silently truncated
# ---------------------------------------------------------------------------


def test_a_large_history_is_returned_complete_not_windowed(tmp_path, monkeypatch):
    """
    The headline defect. Past the cache limit, a history query returned the tail.

    Nothing raised, nothing warned — recovery and planning logic reading history
    would simply have been reasoning about a partial past.
    """
    monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", 100)
    log = EventLog(tmp_path / "events")
    _emit(log, 250)

    window = log.recent_events()
    assert len(window) == 100, "the cache is still bounded — memory has not regressed"

    history = log.list_events()
    assert len(history) == 250, "but history is complete"
    assert [event.event_index for event in history] == list(range(1, 251))
    assert log.cache_is_complete is False


def test_filters_and_traces_see_the_whole_log(tmp_path, monkeypatch):
    """Everything built on list_events() inherits completeness, not the window."""
    monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", 50)
    log = EventLog(tmp_path / "events")
    _emit(log, 200)

    # wf-3 is long evicted from the cache.
    assert len(log.filter_events(workflow_id="wf-3")) == 1
    assert len(log.workflow_trace("wf-3")) == 1
    assert len(log.agent_activity("MR1")) == 200


def test_rotation_preserves_order_ids_and_indices(tmp_path):
    log = EventLog(tmp_path / "events")
    _emit(log, 120)
    before = log.list_events()

    segment = log.rotate(keep_recent=20)
    assert segment is not None
    assert segment.count == 100
    assert segment.first_index == 1
    assert segment.last_index == 100

    after = log.list_events()
    assert len(after) == 120, "rotation must not lose a single event"
    assert [event.event_id for event in after] == [event.event_id for event in before]
    assert [event.event_index for event in after] == list(range(1, 121))


def test_indices_continue_across_rotation(tmp_path):
    """
    An index that restarts at 1 after rotation forks the log.

    The live file is short or empty post-rotation, so the next index cannot come
    from it — it comes from the manifest.
    """
    log = EventLog(tmp_path / "events")
    _emit(log, 50)
    log.rotate(keep_recent=0)

    assert log.recent_events() == [], "the live file was fully sealed"

    _emit(log, 5, start=50)
    history = log.list_events()
    assert [event.event_index for event in history] == list(range(1, 56))
    assert len(history) == 55


def test_archived_history_survives_restart(tmp_path):
    log = EventLog(tmp_path / "events")
    _emit(log, 80)
    log.rotate(keep_recent=10)

    reopened = EventLog(tmp_path / "events")
    history = reopened.list_events()
    assert len(history) == 80
    assert [event.event_index for event in history] == list(range(1, 81))

    _emit(reopened, 3, start=80)
    assert len(reopened.list_events()) == 83


def test_an_archived_event_is_still_findable_by_id(tmp_path):
    log = EventLog(tmp_path / "events")
    _emit(log, 60)
    archived_id = log.list_events()[0].event_id
    log.rotate(keep_recent=5)

    found = log.get_event(archived_id)
    assert found is not None, "an archived event still happened"
    assert found.event_index == 1


def test_a_causal_parent_survives_rotation(tmp_path):
    """
    Causal links are the reason rotation keeps a tail.

    `emit()` resolves a parent by searching backwards. If rotation swept the
    whole live file out mid-chain, the next event would be orphaned — and an
    explicitly-parented event would have been *rejected* outright.
    """
    log = EventLog(tmp_path / "events")
    parent = log.emit(
        event_type="capability_requested",
        actor_id="MR1",
        actor_type="root_orchestrator",
        target_id="shell_command",
        target_type="capability",
        status="requested",
        summary="requested",
        workflow_id="wf-x",
        task_id="tk-x",
    )
    _emit(log, 40)
    log.rotate(keep_recent=0)  # the parent is now archived, and the cache is empty

    child = log.emit(
        event_type="capability_allowed",
        actor_id="MR1",
        actor_type="root_orchestrator",
        target_id="shell_command",
        target_type="capability",
        status="allowed",
        summary="allowed",
        parent_event_id=parent.event_id,
        workflow_id="wf-x",
        task_id="tk-x",
    )
    assert child.parent_event_id == parent.event_id
    assert log.get_event(parent.event_id) is not None


def test_a_missing_segment_is_an_error_not_an_empty_answer(tmp_path):
    """A short answer to a full-history question is worse than no answer."""
    log = EventLog(tmp_path / "events")
    _emit(log, 40)
    segment = log.rotate(keep_recent=5)

    (tmp_path / "events" / "archive" / segment.name).unlink()

    with pytest.raises(EventArchiveError, match="missing"):
        EventLog(tmp_path / "events").list_events()


def test_a_corrupt_manifest_does_not_read_as_no_history(tmp_path):
    log = EventLog(tmp_path / "events")
    _emit(log, 30)
    log.rotate(keep_recent=5)

    (tmp_path / "events" / "segments.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(EventArchiveError):
        EventArchive(tmp_path / "events").archived_last_index()


# ---------------------------------------------------------------------------
# 2. Workflows — archive the finished, never the live
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime"
    store = WorkflowStore(root=root / "workflows")
    agents = AgentStore(root=root / "agents")
    return root, store, agents


def _finish_workflow(store, scheduler, runner, *, finished_at: str) -> str:
    workflow_id = scheduler.submit_workflow(SPEC, CREATED_BY)
    scheduler.tick()
    task_id = next(iter(store.load_workflow(workflow_id).tasks))
    runner.complete(task_id, RunStatus.SUCCEEDED, summary="done")
    scheduler.tick()

    with store.locked():
        workflow = store.load_workflow(workflow_id)
        assert workflow.status is WorkflowStatus.SUCCEEDED
        workflow.finished_at = finished_at
        store.save_workflow(workflow)
    return workflow_id


def test_terminal_workflows_are_archived_and_active_ones_are_not(runtime):
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)

    old_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")
    live_id = scheduler.submit_workflow(SPEC, CREATED_BY)
    scheduler.tick()  # live_id is now RUNNING

    later = _clock(days_in=60)
    manager = RetentionManager(
        root,
        policy=RetentionPolicy(workflow_archive_after_days=30, workflow_keep_recent=0),
        clock=later,
    )
    report = manager.run()

    assert report.workflows_archived == [old_id]
    assert report.workflows_kept.get("active") == 1

    assert not (root / "workflows" / old_id).exists()
    assert (root / "archive" / "workflows" / old_id / "workflow.json").exists()
    assert (root / "workflows" / live_id).exists(), "live work must never be archived"


def test_recent_terminal_workflows_are_kept(runtime):
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    workflow_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")

    manager = RetentionManager(
        root,
        policy=RetentionPolicy(workflow_archive_after_days=30),
        clock=_clock(days_in=5),  # only 5 days old
    )
    report = manager.run()

    assert report.workflows_archived == []
    assert report.workflows_kept.get("too_recent") == 1
    assert (root / "workflows" / workflow_id).exists()


def test_the_keep_recent_floor_beats_the_age_rule(runtime):
    """A quiet month must not empty out the history an operator reads."""
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)

    ids = [
        _finish_workflow(store, scheduler, runner, finished_at=f"2026-01-0{i + 1}T00:00:00+00:00")
        for i in range(4)
    ]

    manager = RetentionManager(
        root,
        policy=RetentionPolicy(workflow_archive_after_days=1, workflow_keep_recent=2),
        clock=_clock(days_in=90),
    )
    report = manager.run()

    assert len(report.workflows_archived) == 2
    assert report.workflows_kept.get("keep_recent_floor") == 2
    # The two newest survive.
    assert set(report.workflows_archived) == set(ids[:2])


def test_a_workflow_an_objective_still_references_is_never_archived(runtime):
    """
    Cleanup must not break objective history.

    An objective's attempt history points at the workflows it ran. Archiving one
    out from under it turns `mr1 objective show` and the recovery ladder's own
    history into dangling references.
    """
    from mr1.autonomy.objectives import Attempt, ObjectiveStore

    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    workflow_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")

    objectives = ObjectiveStore(root, clock=clock)
    objective = objectives.create(
        title="Genesis",
        statement="run the weekly cycle",
        kind="recurring",
        trigger={"type": "interval", "interval_s": 3600},
        owner_agent_id=agents.root_agent_id,
    )
    objective.record_attempt(
        Attempt(workflow_id=workflow_id, outcome="succeeded", at=clock.now_iso())
    )
    objectives.save(objective)

    manager = RetentionManager(
        root,
        policy=RetentionPolicy(workflow_archive_after_days=1, workflow_keep_recent=0),
        clock=_clock(days_in=90),
    )
    report = manager.run()

    assert report.workflows_archived == []
    assert report.workflows_kept.get("referenced") == 1
    assert (root / "workflows" / workflow_id).exists()


def test_a_workflow_with_a_pending_approval_is_never_archived(runtime, monkeypatch):
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    workflow_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")

    # A pending approval pinned to the workflow.
    from mr1.capability_policy import (
        CapabilityApprovalStore,
        CapabilityRequest,
        PolicyEngine,
        ScopeContext,
        build_approval_request,
        maybe_route_approval_request,
        metadata_for_capability,
    )
    from mr1.messages import MessageStore

    approvals = CapabilityApprovalStore(root / "capability_approvals", clock=clock)
    request = CapabilityRequest(
        actor_id=agents.root_agent_id,
        actor_type="root_orchestrator",
        actor_clearance=0.99,
        invocation_mode="workflow",
        capability_name="shell_command",
        args={"argv": ["echo", "hi"], "cwd": str(root)},
        scope=ScopeContext(allowed_roots=[root], workspace_root=root),
        workflow_id=workflow_id,
        task_id="tk-1",
    )
    metadata = metadata_for_capability("shell_command", "tool")
    decision = PolicyEngine().evaluate(request, metadata)
    maybe_route_approval_request(
        build_approval_request(request, metadata, decision),
        approval_store=approvals,
        message_store=MessageStore(root=root / "messages", scoped_agent_store=agents),
        scoped_agent_store=agents,
    )

    manager = RetentionManager(
        root,
        policy=RetentionPolicy(workflow_archive_after_days=1, workflow_keep_recent=0),
        clock=_clock(days_in=90),
    )
    report = manager.run()

    assert report.workflows_archived == []
    assert report.workflows_kept.get("referenced") == 1


# ---------------------------------------------------------------------------
# 3. Operational controls
# ---------------------------------------------------------------------------


def test_dry_run_changes_nothing_and_reports_everything(runtime):
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    workflow_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")

    later = _clock(days_in=90)
    policy = RetentionPolicy(workflow_archive_after_days=1, workflow_keep_recent=0)
    manager = RetentionManager(root, policy=policy, clock=later)

    dry = manager.run(dry_run=True)
    assert dry.dry_run is True
    assert dry.workflows_archived == [workflow_id]
    assert dry.bytes_reclaimed > 0
    assert (root / "workflows" / workflow_id).exists(), "a dry run must not move anything"
    assert not (root / "archive").exists()

    wet = RetentionManager(root, policy=policy, clock=later).run()
    assert wet.workflows_archived == dry.workflows_archived, (
        "the dry run must predict exactly what the real run does"
    )
    assert not (root / "workflows" / workflow_id).exists()


def test_every_run_leaves_a_persisted_report_and_a_timeline_event(runtime):
    root, store, agents = runtime
    manager = RetentionManager(root, clock=_clock())
    report = manager.run()

    reports = list((root / "retention" / "reports").glob("retention-*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert payload["dry_run"] is False
    assert "policy" in payload, "the report records the thresholds it ran under"

    events = EventLog(root / "events").list_events()
    assert [event.event_type for event in events] == ["retention_run"]
    assert events[0].metadata["dry_run"] is False


def test_nothing_is_deleted_merely_for_being_old(runtime):
    """
    Archival is not deletion. The purge is opt-in and never touches live state.
    """
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    workflow_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")

    policy = RetentionPolicy(workflow_archive_after_days=1, workflow_keep_recent=0)
    assert policy.purge_archives_after_days is None, "purging is off by default"

    report = RetentionManager(root, policy=policy, clock=_clock(days_in=90)).run()
    assert report.archives_purged == []
    # Archived — which means still there, still readable.
    archived = root / "archive" / "workflows" / workflow_id / "workflow.json"
    assert archived.exists()
    assert json.loads(archived.read_text())["workflow_id"] == workflow_id


def test_the_purge_is_explicit_and_confined_to_the_archive(runtime):
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    archived_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")

    RetentionManager(
        root,
        policy=RetentionPolicy(workflow_archive_after_days=1, workflow_keep_recent=0),
        clock=_clock(days_in=90),
    ).run()
    assert (root / "archive" / "workflows" / archived_id).exists()

    # A live workflow that is *also* old enough to be purged by age alone.
    live_id = scheduler.submit_workflow(SPEC, CREATED_BY)
    scheduler.tick()

    report = RetentionManager(
        root,
        policy=RetentionPolicy(
            workflow_archive_after_days=1,
            workflow_keep_recent=0,
            purge_archives_after_days=0.0001,
        ),
        clock=_clock(days_in=120),
    ).run()

    assert report.archives_purged, "the archived workflow is gone"
    assert not (root / "archive" / "workflows" / archived_id).exists()
    assert (root / "workflows" / live_id).exists(), (
        "the purge must never be able to reach live state"
    )


def test_events_rotate_when_the_live_log_exceeds_its_limit(runtime):
    root, _store, _agents = runtime
    log = EventLog(root / "events")
    _emit(log, 300)
    live_bytes = log.history_stats()["live_bytes"]

    manager = RetentionManager(
        root,
        policy=RetentionPolicy(
            events_max_live_bytes=live_bytes // 2,
            events_keep_recent=50,
            workflow_archive_after_days=999,
        ),
        clock=_clock(),
    )

    dry = manager.run(dry_run=True)
    assert dry.events_rotated is True
    assert dry.events_archived_count == 250
    assert EventLog(root / "events").history_stats()["archive_segments"] == 0

    report = manager.run()
    assert report.events_rotated is True
    # 251, not 250: the dry run recorded its own `retention_run` event on the
    # very log it was measuring. That is the audit trail working, not drift.
    assert report.events_archived_count == 251
    assert report.events_live_bytes_after < report.events_live_bytes_before

    reopened = EventLog(root / "events")
    # +1 and +2 for the retention_run events the sweeps themselves emit.
    history = reopened.list_events()
    assert len([e for e in history if e.event_type == "workflow_created"]) == 300
    assert reopened.history_stats()["archive_segments"] == 1


def test_the_runtime_restarts_cleanly_after_retention(runtime):
    """Retention must leave a runtime that still works, not just a smaller one."""
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")
    scheduler.shutdown()

    RetentionManager(
        root,
        policy=RetentionPolicy(
            workflow_archive_after_days=1,
            workflow_keep_recent=0,
            events_max_live_bytes=1,
            events_keep_recent=2,
        ),
        clock=_clock(days_in=90),
    ).run()

    # A fresh runtime on the archived root: submits, ticks, and completes.
    store2 = WorkflowStore(root=root / "workflows")
    runner2 = MockRunner()
    scheduler2 = Scheduler(store2, runner2, auto_tick=False, scoped_agent_store=agents)
    workflow_id = scheduler2.submit_workflow(SPEC, CREATED_BY)
    scheduler2.tick()
    task_id = next(iter(store2.load_workflow(workflow_id).tasks))
    runner2.complete(task_id, RunStatus.SUCCEEDED, summary="ok")
    scheduler2.tick()

    assert store2.load_workflow(workflow_id).status is WorkflowStatus.SUCCEEDED
    assert EventLog(root / "events").list_events(), "the timeline still works"
    scheduler2.shutdown()


def test_audits_and_snapshots_are_archived_by_age(runtime):
    """
    Execution artifacts are judged by file mtime — the only timestamp an
    arbitrary artifact carries — so the test sets mtimes rather than pretending
    the clock can age a file it did not write.
    """
    import os

    root, _store, _agents = runtime
    now = _clock()
    now_ts = now.now().timestamp()

    audit = root / "agents" / "ag-root" / "logs" / "capability_audits" / "aud-1.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({"capability_name": "shell_command"}), encoding="utf-8")

    fresh = root / "agents" / "ag-root" / "logs" / "capability_audits" / "aud-2.json"
    fresh.write_text(json.dumps({"capability_name": "read_file"}), encoding="utf-8")

    snapshot = root / "snapshots" / "snap-old"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "snapshot_manifest.json").write_text("{}", encoding="utf-8")

    old_ts = now_ts - 200 * 86_400
    os.utime(audit, (old_ts, old_ts))
    os.utime(snapshot, (old_ts, old_ts))
    os.utime(fresh, (now_ts, now_ts))

    report = RetentionManager(
        root,
        policy=RetentionPolicy(
            audit_archive_after_days=90,
            snapshot_archive_after_days=90,
        ),
        clock=now,
    ).run()

    assert report.audits_archived == 1
    assert report.snapshots_archived == 1
    assert not audit.exists()
    assert fresh.exists(), "a recent audit record stays where it is"
    assert (root / "archive" / "capability_audits" / "ag-root" / "aud-1.json").exists()
    assert (root / "archive" / "snapshots" / "snap-old").exists()


def test_the_archive_records_when_it_archived_each_item(runtime):
    """
    `shutil.move` carries the original mtime across, so the archive cannot ask
    the filesystem when it took custody of something. It writes it down.
    """
    root, store, agents = runtime
    clock = _clock()
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False, scoped_agent_store=agents, clock=clock)
    workflow_id = _finish_workflow(store, scheduler, runner, finished_at="2026-01-01T00:00:00+00:00")

    archived_at_clock = _clock(days_in=90)
    RetentionManager(
        root,
        policy=RetentionPolicy(workflow_archive_after_days=1, workflow_keep_recent=0),
        clock=archived_at_clock,
    ).run()

    ledger = json.loads(
        (root / "archive" / "workflows" / ".archive_index.json").read_text(encoding="utf-8")
    )
    assert workflow_id in ledger
    assert ledger[workflow_id]["archived_at"] == archived_at_clock.now_iso()

    # One day after archival, a 30-day purge must not fire.
    report = RetentionManager(
        root,
        policy=RetentionPolicy(
            workflow_archive_after_days=1,
            workflow_keep_recent=0,
            purge_archives_after_days=30,
        ),
        clock=_clock(days_in=91),
    ).run()
    assert report.archives_purged == []
    assert (root / "archive" / "workflows" / workflow_id).exists()


def test_the_supervisor_runs_retention_on_its_own_cadence_without_a_brain(runtime):
    """
    Retention must be automatic to be useful — an operator who has to remember
    to run it is an operator whose disk fills.

    It is deliberately outside the planning gate (a paused supervisor should
    still reclaim disk) and it must never reach the planner: cleanup that
    depends on an LLM is cleanup that stops working when the LLM does.
    """
    from mr1.autonomy.service import Supervisor, SupervisorConfig

    root, _store, _agents = runtime
    clock = _clock()

    brain_calls = []

    class ExplodingPlanner:
        def plan(self, *args, **kwargs):
            brain_calls.append(1)
            raise AssertionError("retention must never call the brain")

    supervisor = Supervisor(
        root,
        config=SupervisorConfig(retention_interval_s=3600, tick_interval_s=60),
        clock=clock,
        runner=MockRunner(),
        auto_scheduler_tick=False,
        planner=ExplodingPlanner(),
        retention_policy=RetentionPolicy(events_max_live_bytes=1, events_keep_recent=5),
    )

    _emit(EventLog(root / "events"), 40)

    first = supervisor.tick()
    assert "retention" in first, "the first tick runs retention"
    assert EventLog(root / "events").history_stats()["archive_segments"] == 1

    # Well inside the interval: it must not run again.
    clock.advance(60)
    second = supervisor.tick()
    assert "retention" not in second

    # Past the interval: it runs again.
    clock.advance(3600)
    _emit(EventLog(root / "events"), 40, start=40)
    third = supervisor.tick()
    assert "retention" in third

    assert brain_calls == [], "no idle tick, and no retention run, may call the brain"
    assert len(EventLog(root / "events").list_events(limit=None)) >= 80
    supervisor.shutdown()


def test_status_reports_where_history_lives(runtime):
    root, _store, _agents = runtime
    log = EventLog(root / "events")
    _emit(log, 40)
    log.rotate(keep_recent=10)

    status = RetentionManager(root, clock=_clock()).status()
    assert status["events"]["archive_segments"] == 1
    assert status["events"]["archived_events"] == 30
    assert status["events"]["cache_is_complete"] is False
    assert status["events"]["total_events"] == 40
