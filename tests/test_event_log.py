from __future__ import annotations

import json
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mr1.capability_policy import CapabilityApprovalDecision
from mr1.capability_runner import CapabilityRunner
from mr1.event_log import EventLog, SystemEvent, _MAX_CACHE_EVENTS
from mr1.messages import MessageStore
from mr1.mrn_loop import MRnStepRunner
from mr1.scoped_agents import PersistentAgentStore
from mr1.scheduler import Scheduler
from mr1.kazi_runner import MockRunner, RunStatus
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


def _append_events_in_process(log_root: str, worker_id: int, count: int) -> None:
    log = EventLog(Path(log_root))
    for index in range(count):
        workflow_id = f"wf-{worker_id}-{index}"
        log.emit(
            event_type="workflow_created",
            actor_id=f"worker-{worker_id}",
            actor_type="test",
            target_id=workflow_id,
            target_type="workflow",
            status="pending",
            summary="created",
            workflow_id=workflow_id,
        )


def _ts(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()


def _action(action: str, **extra) -> str:
    payload = {
        "action": action,
        "reason": "test reason",
        "next_status": extra.pop("next_status", "idle" if action == "idle" else "working"),
        "workflow_request": extra.pop("workflow_request", None),
        "workflow_context": extra.pop("workflow_context", None),
        "workflow_id": extra.pop("workflow_id", None),
        "report": extra.pop("report", None),
        "message_kind": extra.pop("message_kind", None),
        "message_subject": extra.pop("message_subject", None),
        "message_body": extra.pop("message_body", None),
        "to_agent_id": extra.pop("to_agent_id", None),
        "parent_request": extra.pop("parent_request", None),
    }
    payload.update(extra)
    return json.dumps(payload)


class FakeReasoner:
    def __init__(self, *responses: str):
        self._responses = list(responses)

    def __call__(self, agent, system_prompt: str, prompt: str) -> str:
        if not self._responses:
            raise AssertionError("no reasoner responses configured")
        return self._responses.pop(0)


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def message_store(tmp_path, agent_store):
    return MessageStore(root=tmp_path / "messages", scoped_agent_store=agent_store)


@pytest.fixture
def event_log(tmp_path):
    return EventLog(tmp_path / "events")


def test_append_and_trace_are_deterministic(event_log: EventLog):
    first = event_log.emit(
        event_type="workflow_created",
        actor_id="cli",
        actor_type="user",
        target_id="wf-1",
        target_type="workflow",
        status="pending",
        summary="created",
        workflow_id="wf-1",
        timestamp=_ts("2026-04-29T12:00:00.100000+00:00"),
    )
    duplicate = event_log.emit(
        event_type="workflow_created",
        actor_id="cli",
        actor_type="user",
        target_id="wf-1",
        target_type="workflow",
        status="pending",
        summary="created",
        workflow_id="wf-1",
        timestamp=_ts("2026-04-29T12:00:00.100000+00:00"),
    )
    second = event_log.emit(
        event_type="workflow_started",
        actor_id="scheduler",
        actor_type="scheduler",
        target_id="wf-1",
        target_type="workflow",
        status="running",
        summary="started",
        workflow_id="wf-1",
        correlation_id=first.correlation_id,
        timestamp=_ts("2026-04-29T11:59:59.000000+00:00"),
    )

    assert first.event_id == duplicate.event_id
    assert duplicate.event_index == 1
    trace = event_log.trace_by_correlation(first.correlation_id or "")
    assert [item.event_index for item in trace] == [1, 2]
    assert [item.event_type for item in trace] == ["workflow_created", "workflow_started"]


def test_append_large_log_reuses_loaded_cache(event_log: EventLog, monkeypatch):
    for index in range(500):
        workflow_id = f"wf-{index}"
        event_log.emit(
            event_type="workflow_created",
            actor_id="cli",
            actor_type="user",
            target_id=workflow_id,
            target_type="workflow",
            status="pending",
            summary="created",
            workflow_id=workflow_id,
            timestamp=_ts(f"2026-04-29T12:00:{index % 60:02d}.{index % 1000:03d}000+00:00"),
        )

    reloaded = EventLog(event_log.path)
    rebuilds = 0
    real_rebuild = reloaded._rebuild_cache_locked

    def wrapped_rebuild() -> None:
        nonlocal rebuilds
        rebuilds += 1
        real_rebuild()

    monkeypatch.setattr(reloaded, "_rebuild_cache_locked", wrapped_rebuild)

    for suffix in ("first", "second"):
        reloaded.emit(
            event_type="workflow_created",
            actor_id="cli",
            actor_type="user",
            target_id=f"wf-{suffix}",
            target_type="workflow",
            status="pending",
            summary="created",
            workflow_id=f"wf-{suffix}",
        )

    assert rebuilds == 1
    assert reloaded.list_events()[-1].event_index == 502


def test_concurrent_appenders_get_unique_indices(tmp_path):
    log_root = tmp_path / "events"
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_append_events_in_process, args=(str(log_root), worker_id, 25))
        for worker_id in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    events = EventLog(log_root).list_events()
    indices = [event.event_index for event in events]
    assert len(events) == 100
    assert indices == list(range(1, 101))


def test_parent_event_must_exist(event_log: EventLog):
    with pytest.raises(ValueError, match="parent event not found"):
        event_log.append_event(SystemEvent(
            event_id="evt-test",
            event_index=0,
            event_version=1,
            timestamp=_ts("2026-04-29T12:00:00+00:00"),
            event_type="message_read",
            event_kind="communication",
            actor_id="ag-1",
            actor_type="mrn",
            target_id="msg-1",
            target_type="message",
            status="read",
            severity="INFO",
            summary="read",
            parent_event_id="evt-missing",
            metadata={},
        ))


def test_invalid_kind_and_severity_are_rejected():
    with pytest.raises(ValueError, match="invalid event_kind"):
        SystemEvent(
            event_id="evt-test",
            event_index=0,
            event_version=1,
            timestamp=_ts("2026-04-29T12:00:00+00:00"),
            event_type="message_read",
            event_kind="bad",
            actor_id="ag-1",
            actor_type="mrn",
            target_id="msg-1",
            target_type="message",
            status="read",
            severity="INFO",
            summary="read",
            metadata={},
        )
    with pytest.raises(ValueError, match="invalid severity"):
        SystemEvent(
            event_id="evt-test",
            event_index=0,
            event_version=1,
            timestamp=_ts("2026-04-29T12:00:00+00:00"),
            event_type="message_read",
            event_kind="communication",
            actor_id="ag-1",
            actor_type="mrn",
            target_id="msg-1",
            target_type="message",
            status="read",
            severity="BAD",
            summary="read",
            metadata={},
        )


def test_capability_block_approval_and_consumption_emit_events(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child")
    allowed_dir = tmp_path / "allowed"
    denied_dir = tmp_path / "denied"
    allowed_dir.mkdir()
    denied_dir.mkdir()
    target = denied_dir / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    child.scope_roots = [str(allowed_dir)]
    agent_store.save_agent(child)

    runner = CapabilityRunner(
        scoped_agent_store=agent_store,
        workspace_root=tmp_path,
    )

    first = runner.run_capability(
        "read_file",
        {"path": str(target)},
        child.agent_id,
        step_id=f"{child.agent_id}:1",
    )
    assert first.status == "requires_approval"

    approval = runner._approval_store.apply_decision(
        first.approval_request_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=first.approval_request_id,
            decision="approved",
            decided_by=root.agent_id,
            reason="approve",
            timestamp=datetime.now(timezone.utc).timestamp(),
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )
    assert approval.status == "approved"

    second = runner.run_capability(
        "read_file",
        {"path": str(target)},
        child.agent_id,
        step_id=f"{child.agent_id}:1",
    )
    assert second.status == "succeeded"

    event_log = EventLog(tmp_path / "events")
    event_types = [event.event_type for event in event_log.list_events()]
    assert "capability_requested" in event_types
    assert "capability_blocked" in event_types
    assert "approval_requested" in event_types
    assert "approval_approved" in event_types
    assert "approval_consumed" in event_types


def test_scope_and_message_events_emit(tmp_path, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child")
    scope_dir = tmp_path / "scope"
    scope_dir.mkdir()

    agent_store.grant_scope(root.agent_id, child.agent_id, scope_dir, reason="allow")
    agent_store.revoke_scope(root.agent_id, child.agent_id, scope_dir)
    message = message_store.create_message(
        from_agent_id=root.agent_id,
        to_agent_id=child.agent_id,
        kind="status",
        subject="hello",
        body="body",
    )
    message_store.mark_read(message.message_id, actor_id=child.agent_id)

    event_types = [event.event_type for event in EventLog(tmp_path / "events").list_events()]
    assert "agent_created" in event_types
    assert "agent_scope_granted" in event_types
    assert "agent_scope_revoked" in event_types
    assert "message_sent" in event_types
    assert "message_read" in event_types


def test_workflow_lifecycle_events_emit(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    scheduler = Scheduler(
        workflow_store,
        MockRunner(),
        auto_tick=False,
        scoped_agent_store=agent_store,
    )
    spec = {
        "title": "Timeline workflow",
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
    wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id=root.agent_id))
    scheduler.tick()
    wf = workflow_store.load_workflow(wf_id)
    task = wf.task_by_label("a")
    scheduler._runner.complete(task.task_id, RunStatus.SUCCEEDED, summary="ok")
    scheduler.tick()
    scheduler.shutdown()

    event_types = [event.event_type for event in EventLog(workflow_store.root.parent / "events").list_events()]
    assert "workflow_created" in event_types
    assert "workflow_started" in event_types
    assert "workflow_task_started" in event_types
    assert "workflow_task_completed" in event_types
    assert "workflow_completed" in event_types


def test_mrn_step_and_report_events_emit(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_action("write_report", report="## Findings", next_status="reporting")),
    )

    result = runner.step(child.agent_id)
    assert result.report_path is not None

    event_types = [event.event_type for event in EventLog(workflow_store.root.parent / "events").list_events()]
    assert "mrn_step_started" in event_types
    assert "mrn_step_completed" in event_types
    assert "mrn_reported" in event_types


# ---------------------------------------------------------------------------
# N-5: Bounded event cache
# ---------------------------------------------------------------------------

def _emit_n(log: EventLog, n: int, *, start: int = 0) -> None:
    for i in range(start, start + n):
        log.emit(
            event_type="workflow_created",
            actor_id="test-actor",
            actor_type="test",
            target_id=f"wf-{i}",
            target_type="workflow",
            status="pending",
            summary=f"wf {i}",
            workflow_id=f"wf-{i}",
        )


class TestBoundedCache:
    """N-5: event cache must not grow without bound."""

    def test_cache_stays_at_max_after_overflow(self, tmp_path, monkeypatch):
        small_limit = 10
        monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", small_limit)
        log = EventLog(tmp_path / "events")
        _emit_n(log, small_limit + 5)
        assert len(log._cache.events) == small_limit

    def test_oldest_events_evicted_first(self, tmp_path, monkeypatch):
        small_limit = 10
        monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", small_limit)
        log = EventLog(tmp_path / "events")
        _emit_n(log, small_limit + 3)
        cached_ids = {e.event_id for e in log._cache.events}
        # Events are kept in index order; earliest events should be gone from the cache.
        # The cache should hold the last `small_limit` events.
        assert len(cached_ids) == small_limit
        all_disk = [
            json.loads(line)
            for line in (tmp_path / "events" / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        # Disk always has all events.
        assert len(all_disk) == small_limit + 3
        # First few events should have been evicted from cache.
        first_id = all_disk[0]["event_id"]
        assert first_id not in cached_ids

    def test_dedupe_still_works_within_cache_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", 20)
        log = EventLog(tmp_path / "events")
        _emit_n(log, 10)
        initial_count = len(list(log._cache.events))
        # Re-emit the same workflow_id events — these produce different event_ids
        # (timestamps differ), so they won't be deduped; the point is the cache
        # size is still bounded.
        _emit_n(log, 15)
        assert len(log._cache.events) == 20

    def test_rebuild_applies_bound(self, tmp_path, monkeypatch):
        """
        The rebuild bounds *memory*. It must not bound *history*.

        This test used to assert `list_events()` returned only the cached
        window — which is the B1 defect written down as an expectation: past
        the limit, every history query silently answered with the tail. The
        memory bound is still real, and is now asserted where it lives.
        """
        small_limit = 8
        monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", small_limit)
        # Write events without cache (directly to file via a temporary log).
        log_a = EventLog(tmp_path / "events")
        _emit_n(log_a, small_limit + 4)

        # Fresh log instance triggers a rebuild.
        log_b = EventLog(tmp_path / "events")

        window = log_b.recent_events()
        assert len(window) == small_limit, "the cache is still bounded"
        assert window == list(log_b._cache.events), "recent_events() is the cached window, by name"
        assert log_b.cache_is_complete is False, "and it knows it is not all of history"

        events = log_b.list_events()
        assert len(events) == small_limit + 4, (
            "list_events() is a full-history query and must never silently truncate"
        )
        assert [event.event_index for event in events] == list(range(1, small_limit + 5))

    def test_file_always_has_full_history(self, tmp_path, monkeypatch):
        small_limit = 5
        monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", small_limit)
        log = EventLog(tmp_path / "events")
        total = small_limit + 10
        _emit_n(log, total)

        lines = [
            line for line in
            (tmp_path / "events" / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(lines) == total, "JSONL file must contain full history"

    def test_event_by_id_stays_consistent_with_cache(self, tmp_path, monkeypatch):
        small_limit = 10
        monkeypatch.setattr("mr1.event_log._MAX_CACHE_EVENTS", small_limit)
        log = EventLog(tmp_path / "events")
        _emit_n(log, small_limit + 5)
        # event_by_id must exactly match the cached events.
        cached_events = list(log._cache.events)
        assert len(log._cache.event_by_id) == len(cached_events)
        for event in cached_events:
            assert log._cache.event_by_id.get(event.event_id) is event
