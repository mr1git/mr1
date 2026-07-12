"""A5 — the objective store, triggers, and the persisted lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mr1.autonomy.objectives import (
    KIND_ONCE,
    KIND_RECURRING,
    KIND_STANDING,
    STATUS_ACTIVE,
    STATUS_PAUSED,
    STATUS_QUARANTINED,
    STATUS_SATISFIED,
    Attempt,
    Objective,
    ObjectiveError,
    ObjectiveStore,
    trigger_is_ready,
)
from mr1.autonomy.recovery import FailurePolicy
from mr1.clock import VirtualClock
from mr1.watchers import default_watcher_registry


def _store(tmp_path):
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return ObjectiveStore(tmp_path / "runtime", clock=clock), clock


def _create(store, **overrides):
    payload = {
        "title": "Genesis",
        "statement": "run the weekly genesis cycle",
        "owner_agent_id": "agent-root",
        "kind": KIND_ONCE,
    }
    payload.update(overrides)
    return store.create(**payload)


def test_create_and_reload(tmp_path):
    store, _clock = _store(tmp_path)
    objective = _create(store)

    reloaded = ObjectiveStore(tmp_path / "runtime").require(objective.objective_id)
    assert reloaded.statement == "run the weekly genesis cycle"
    assert reloaded.status == STATUS_ACTIVE
    assert reloaded.created_at.startswith("2026-01-01")


def test_an_objective_must_want_something(tmp_path):
    store, _clock = _store(tmp_path)
    with pytest.raises(ObjectiveError):
        _create(store, statement="   ")


def test_unknown_kind_and_status_are_rejected(tmp_path):
    store, _clock = _store(tmp_path)
    with pytest.raises(ObjectiveError):
        _create(store, kind="whenever")
    objective = _create(store)
    with pytest.raises(ObjectiveError):
        store.set_status(objective.objective_id, "vibing")


def test_update_is_read_modify_write(tmp_path):
    store, _clock = _store(tmp_path)
    objective = _create(store)

    store.update(objective.objective_id, consecutive_failures=3)
    store.update(objective.objective_id, success_count=1)

    reloaded = store.require(objective.objective_id)
    assert reloaded.consecutive_failures == 3
    assert reloaded.success_count == 1


def test_update_rejects_unknown_fields(tmp_path):
    store, _clock = _store(tmp_path)
    objective = _create(store)
    with pytest.raises(ObjectiveError):
        store.update(objective.objective_id, nonsense=True)


def test_history_is_bounded(tmp_path):
    store, _clock = _store(tmp_path)
    objective = _create(store)
    for index in range(40):
        objective.record_attempt(Attempt(
            workflow_id=f"wf-{index}",
            outcome="failed",
            at="2026-01-01T00:00:00+00:00",
        ))
    store.save(objective)

    assert len(store.require(objective.objective_id).history) == 25


def test_live_terminal_and_parked(tmp_path):
    store, _clock = _store(tmp_path)
    active = _create(store)
    satisfied = _create(store, title="done")
    quarantined = _create(store, title="stuck")

    store.set_status(satisfied.objective_id, STATUS_SATISFIED)
    store.set_status(quarantined.objective_id, STATUS_QUARANTINED)

    live_ids = [item.objective_id for item in store.list_live()]
    assert live_ids == [active.objective_id]
    assert store.require(satisfied.objective_id).is_terminal
    assert store.require(quarantined.objective_id).is_parked


def test_pause_all_stops_every_live_objective(tmp_path):
    store, _clock = _store(tmp_path)
    first = _create(store)
    second = _create(store, title="second")
    done = _create(store, title="done")
    store.set_status(done.objective_id, STATUS_SATISFIED)

    paused = store.pause_all(reason="halt")

    assert sorted(paused) == sorted([first.objective_id, second.objective_id])
    assert store.require(done.objective_id).status == STATUS_SATISFIED
    assert store.require(first.objective_id).status == STATUS_PAUSED


def test_counts_by_status(tmp_path):
    store, _clock = _store(tmp_path)
    _create(store)
    second = _create(store, title="two")
    store.set_status(second.objective_id, STATUS_QUARANTINED)

    assert store.counts_by_status() == {STATUS_ACTIVE: 1, STATUS_QUARANTINED: 1}


def test_a_corrupt_objective_file_is_skipped_not_fatal(tmp_path):
    store, _clock = _store(tmp_path)
    good = _create(store)
    bad = _create(store, title="bad")
    store.objective_path(bad.objective_id).write_text("{ nope", encoding="utf-8")

    ids = [item.objective_id for item in store.list_objectives()]
    assert ids == [good.objective_id]
    assert store.load(bad.objective_id) is None


def test_failure_policy_round_trips_through_the_store(tmp_path):
    store, _clock = _store(tmp_path)
    objective = _create(store, failure_policy=FailurePolicy(max_retries=1, max_replans=0))

    reloaded = store.require(objective.objective_id)
    assert reloaded.failure_policy.max_retries == 1
    assert reloaded.failure_policy.max_replans == 0


# -- triggers --------------------------------------------------------------


def test_immediate_trigger_fires_once_for_a_once_objective(tmp_path):
    store, clock = _store(tmp_path)
    objective = _create(store, kind=KIND_ONCE, trigger={"type": "immediate"})

    ready, _why = trigger_is_ready(objective, now=clock.now())
    assert ready is True

    objective.last_completed_at = clock.now_iso()
    ready, why = trigger_is_ready(objective, now=clock.now())
    assert ready is False
    assert "completed" in why


def test_interval_trigger_waits_for_the_interval(tmp_path):
    store, clock = _store(tmp_path)
    objective = _create(
        store,
        kind=KIND_RECURRING,
        trigger={"type": "interval", "interval_s": 604_800},
    )

    ready, why = trigger_is_ready(objective, now=clock.now())
    assert ready is True and "never run" in why

    objective.last_completed_at = clock.now_iso()
    clock.advance(604_799)
    assert trigger_is_ready(objective, now=clock.now())[0] is False

    clock.advance(2)
    assert trigger_is_ready(objective, now=clock.now())[0] is True


def test_manual_trigger_never_fires_on_its_own(tmp_path):
    store, clock = _store(tmp_path)
    objective = _create(store, trigger={"type": "manual"})

    ready, why = trigger_is_ready(objective, now=clock.now())
    assert ready is False
    assert "manual" in why


def test_watcher_trigger_reuses_the_watcher_registry(tmp_path):
    store, clock = _store(tmp_path)
    target = tmp_path / "trigger.txt"
    objective = _create(
        store,
        kind=KIND_STANDING,
        trigger={
            "type": "watcher",
            "watcher_type": "file_exists",
            "watch_config": {"path": str(target)},
        },
    )
    registry = default_watcher_registry()

    ready, _why = trigger_is_ready(objective, now=clock.now(), watcher_registry=registry)
    assert ready is False

    target.write_text("go", encoding="utf-8")
    ready, why = trigger_is_ready(objective, now=clock.now(), watcher_registry=registry)
    assert ready is True
    assert "satisfied" in why


def test_an_unknown_trigger_type_never_fires(tmp_path):
    store, clock = _store(tmp_path)
    objective = _create(store, trigger={"type": "telepathy"})

    ready, why = trigger_is_ready(objective, now=clock.now())
    assert ready is False
    assert "unknown trigger" in why


def test_recovery_state_is_derived_from_the_objective(tmp_path):
    store, clock = _store(tmp_path)
    objective = _create(store)
    objective.first_attempt_at = clock.now_iso()
    objective.retries_used = 2
    objective.consecutive_failures = 3
    objective.fallback_statement = "do the simple thing instead"
    objective.record_attempt(Attempt(
        workflow_id="wf-1",
        outcome="failed",
        at=clock.now_iso(),
        signature="abc123",
    ))
    clock.advance(3600)

    state = objective.recovery_state(clock.now())

    assert state.retries_used == 2
    assert state.consecutive_failures == 3
    assert state.elapsed_s == 3600.0
    assert state.recent_signatures == ["abc123"]
    assert state.has_fallback is True


def test_ready_to_retry_respects_the_backoff_deadline(tmp_path):
    store, clock = _store(tmp_path)
    objective = _create(store)
    objective.status = "recovering"
    objective.next_attempt_at = (clock.now()).isoformat()

    assert objective.ready_to_retry(clock.now()) is True

    objective.next_attempt_at = "2026-01-02T00:00:00+00:00"
    assert objective.ready_to_retry(clock.now()) is False
    clock.advance(86_401)
    assert objective.ready_to_retry(clock.now()) is True
