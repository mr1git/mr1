"""
B2 — recurring triggers: interval, cron, restart, and downtime.

The case that matters is not "does an interval fire". Phase A did that. It is
what happens when MR1 comes back after being off for a week and a daily
objective has seven runs pending. A scheduler with no opinion fires seven
workflows in one tick — seven plans, seven consent-authorized executions, for
work whose windows have all passed.

So the load-bearing tests here are the ones about time MR1 did *not* observe:
restart without duplicate firing, and downtime without unbounded catch-up.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mr1.autonomy.objectives import (
    KIND_RECURRING,
    ObjectiveStore,
    evaluate_trigger,
)
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.autonomy.triggers import (
    POLICY_BOUNDED,
    POLICY_CATCH_UP_ONCE,
    POLICY_SKIP,
    CronSpec,
    TriggerError,
    evaluate_recurrence,
    occurrences_due,
    validate_trigger,
)
from mr1.clock import VirtualClock
from mr1.worker_runner import MockRunner


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _at(**delta) -> datetime:
    return START + timedelta(**delta)


# ---------------------------------------------------------------------------
# Cron
# ---------------------------------------------------------------------------


def test_cron_parses_the_fields_people_actually_write():
    spec = CronSpec.parse("0 9 * * 1")  # 09:00 every Monday
    assert spec.matches(datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc))  # a Monday
    assert not spec.matches(datetime(2026, 1, 6, 9, 0, tzinfo=timezone.utc))  # Tuesday
    assert not spec.matches(datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc))


def test_cron_supports_steps_ranges_and_lists():
    spec = CronSpec.parse("*/15 9-17 * * *")
    assert spec.matches(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))
    assert spec.matches(datetime(2026, 1, 1, 17, 45, tzinfo=timezone.utc))
    assert not spec.matches(datetime(2026, 1, 1, 9, 7, tzinfo=timezone.utc))
    assert not spec.matches(datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc))

    weekend = CronSpec.parse("30 6 * * 0,6")
    assert weekend.matches(datetime(2026, 1, 3, 6, 30, tzinfo=timezone.utc))  # Saturday
    assert weekend.matches(datetime(2026, 1, 4, 6, 30, tzinfo=timezone.utc))  # Sunday
    assert not weekend.matches(datetime(2026, 1, 5, 6, 30, tzinfo=timezone.utc))


def test_sunday_is_both_0_and_7():
    assert CronSpec.parse("0 0 * * 7").matches(datetime(2026, 1, 4, tzinfo=timezone.utc))


def test_a_bad_cron_is_rejected_at_creation_not_at_3am():
    for bad in ("", "0 9 * *", "60 9 * * *", "0 25 * * *", "x 9 * * *", "0 9 * * */0"):
        with pytest.raises(TriggerError):
            CronSpec.parse(bad)


def test_cron_fields_are_read_in_the_objectives_timezone():
    """
    "09:00 Monday" means 09:00 *there*. Computing it in UTC and hoping is how a
    calendar objective silently slides an hour twice a year.
    """
    trigger = {
        "type": "cron",
        "expression": "0 9 * * *",
        "timezone": "America/New_York",
    }
    # 2026-06-01 is EDT (UTC-4), so 09:00 local == 13:00 UTC.
    anchor = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    _due, next_due = occurrences_due(trigger, anchor=anchor, now=anchor)
    assert next_due == datetime(2026, 6, 1, 13, 0, tzinfo=timezone.utc)

    # 2026-01-01 is EST (UTC-5), so 09:00 local == 14:00 UTC. Same expression,
    # different UTC instant — which is the whole point of naming a timezone.
    winter = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    _due, winter_next = occurrences_due(trigger, anchor=winter, now=winter)
    assert winter_next == datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)


def test_a_cron_objective_does_not_fire_the_moment_it_is_created():
    """"Every Monday at 9" does not mean "right now, and then every Monday at 9"."""
    trigger = {"type": "cron", "expression": "0 9 * * 1"}
    decision = evaluate_recurrence(trigger, anchor=None, now=_at(hours=1))
    assert decision.ready is False
    assert decision.next_due_at is not None


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------


def test_an_interval_fires_once_per_period():
    trigger = {"type": "interval", "interval_s": 3600}

    assert evaluate_recurrence(trigger, anchor=None, now=START).ready is True

    on_time = evaluate_recurrence(trigger, anchor=START, now=_at(minutes=59))
    assert on_time.ready is False

    due = evaluate_recurrence(trigger, anchor=START, now=_at(hours=1))
    assert due.ready is True
    assert due.due == 1
    assert due.missed == 0
    assert due.next_due_at == _at(hours=2).isoformat()


def test_a_slightly_late_tick_is_not_a_missed_run():
    """A 60s supervisor tick and an hourly objective will always be a little late."""
    trigger = {"type": "interval", "interval_s": 3600}
    decision = evaluate_recurrence(trigger, anchor=START, now=_at(hours=1, seconds=45))
    assert decision.ready is True
    assert decision.due == 1
    assert decision.missed == 0


# ---------------------------------------------------------------------------
# Downtime — the case the whole checkpoint exists for
# ---------------------------------------------------------------------------


def test_a_week_of_downtime_never_produces_a_week_of_workflows():
    """
    The invariant, stated directly: whatever the outage, the backlog is bounded
    by configuration, never by the length of the outage.
    """
    hourly = {"type": "interval", "interval_s": 3600}
    after_a_week = _at(days=7)  # 168 occurrences elapsed

    for policy, expected_runs_now, expected_owed in (
        ({"missed_run_policy": POLICY_SKIP}, False, 0),
        ({"missed_run_policy": POLICY_CATCH_UP_ONCE}, True, 0),
        ({"missed_run_policy": POLICY_BOUNDED, "max_catch_up_runs": 3}, True, 2),
    ):
        trigger = {**hourly, **policy}
        decision = evaluate_recurrence(trigger, anchor=START, now=after_a_week)

        assert decision.missed == 167
        assert decision.ready is expected_runs_now
        assert decision.catch_up_remaining == expected_owed, (
            f"policy {policy} must owe at most its allowance, not 167"
        )


def test_skip_realigns_and_runs_nothing():
    trigger = {
        "type": "interval",
        "interval_s": 3600,
        "missed_run_policy": POLICY_SKIP,
    }
    decision = evaluate_recurrence(trigger, anchor=START, now=_at(days=1))
    assert decision.ready is False
    assert "skipped" in decision.reason
    # Realigned forward: the next boundary is in the future, not in the backlog.
    assert decision.next_due_at > _at(days=1).isoformat()


def test_bounded_catch_up_works_the_backlog_off_one_run_at_a_time():
    trigger = {
        "type": "interval",
        "interval_s": 3600,
        "missed_run_policy": POLICY_BOUNDED,
        "max_catch_up_runs": 3,
    }
    # Down for 10 hours: 9 missed.
    first = evaluate_recurrence(trigger, anchor=START, now=_at(hours=10))
    assert first.ready is True
    assert first.catch_up_remaining == 2

    # The next two ticks fire the owed runs and nothing more.
    second = evaluate_recurrence(
        trigger,
        anchor=_at(hours=10),
        now=_at(hours=10, minutes=1),
        catch_up_remaining=first.catch_up_remaining,
    )
    assert second.ready is True
    assert second.catch_up_remaining == 1

    third = evaluate_recurrence(
        trigger,
        anchor=_at(hours=10, minutes=1),
        now=_at(hours=10, minutes=2),
        catch_up_remaining=second.catch_up_remaining,
    )
    assert third.ready is True
    assert third.catch_up_remaining == 0

    # Backlog spent: back to the ordinary schedule.
    fourth = evaluate_recurrence(
        trigger,
        anchor=_at(hours=10, minutes=2),
        now=_at(hours=10, minutes=3),
        catch_up_remaining=third.catch_up_remaining,
    )
    assert fourth.ready is False


def test_a_cron_backlog_is_bounded_too():
    trigger = {
        "type": "cron",
        "expression": "0 * * * *",  # hourly
        "missed_run_policy": POLICY_CATCH_UP_ONCE,
    }
    decision = evaluate_recurrence(trigger, anchor=START, now=_at(days=3))
    assert decision.missed > 0
    assert decision.ready is True
    assert decision.catch_up_remaining == 0


def test_validate_trigger_rejects_an_unknown_policy():
    with pytest.raises(TriggerError, match="missed_run_policy"):
        validate_trigger({
            "type": "interval",
            "interval_s": 60,
            "missed_run_policy": "run_them_all",
        })


# ---------------------------------------------------------------------------
# Restart — through the real supervisor
# ---------------------------------------------------------------------------


class _RecordingPlanner:
    """Plans a trivial workflow and counts every call."""

    def __init__(self):
        self.calls = 0

    def plan(self, *_args, **_kwargs):
        self.calls += 1
        return {
            "title": "recurring work",
            "tasks": [{
                "label": "t",
                "title": "T",
                "task_kind": "agent",
                "agent_type": "worker",
                "prompt": "work",
            }],
        }


def _supervisor(root, clock, planner, runner=None):
    return Supervisor(
        root,
        config=SupervisorConfig(tick_interval_s=60, retention_interval_s=0),
        clock=clock,
        runner=runner or MockRunner(),
        auto_scheduler_tick=False,
        planner=planner,
    )


def _drive_to_completion(supervisor, runner) -> None:
    """Run the planned workflow through to SUCCEEDED and reconcile it."""
    from mr1.worker_runner import RunStatus

    scheduler = supervisor.scheduler
    scheduler.tick()
    for workflow in scheduler.list_workflows():
        for task_id in workflow.tasks:
            runner.complete(task_id, RunStatus.SUCCEEDED, summary="ok")
    scheduler.tick()
    supervisor.tick()  # RECONCILE records the success and re-arms the objective


def _recurring(root, clock, *, trigger) -> str:
    from mr1.scoped_agents import AgentStore

    objectives = ObjectiveStore(root, clock=clock)
    objective = objectives.create(
        title="Genesis",
        statement="run the weekly cycle",
        kind=KIND_RECURRING,
        trigger=trigger,
        owner_agent_id=AgentStore(root=root / "agents").root_agent_id,
    )
    return objective.objective_id


def test_a_restart_does_not_fire_the_same_occurrence_twice(tmp_path):
    """
    The Phase-A anchor was the last *completion*. A supervisor that fired at
    09:00 and was killed at 09:01 had recorded nothing — so on restart the
    occurrence was still pending and fired again. `last_fired_at` is written
    when the decision is made, which is what makes the recurrence at-most-once.
    """
    root = tmp_path / "runtime"
    clock = VirtualClock(start=START)
    planner = _RecordingPlanner()

    trigger = {"type": "interval", "interval_s": 3600}
    objective_id = _recurring(root, clock, trigger=trigger)

    first = _supervisor(root, clock, planner)
    outcome = first.tick()
    assert "error" not in outcome, outcome
    assert planner.calls == 1, "it fires once when first due"

    objectives = ObjectiveStore(root, clock=clock)
    fired_at = objectives.require(objective_id).last_fired_at
    assert fired_at is not None, "the fire is recorded at the moment of firing"

    # Kill the supervisor mid-flight: the workflow never completes, so
    # `last_completed_at` is still None.
    first.shutdown()
    assert objectives.require(objective_id).last_completed_at is None

    # Restart a minute later. The occurrence is spent; it must not fire again.
    clock.advance(60)
    second = _supervisor(root, clock, planner)
    second.tick()
    assert planner.calls == 1, "a restart must not re-fire a spent occurrence"

    assert objectives.require(objective_id).last_fired_at == fired_at
    second.shutdown()


def test_downtime_through_the_supervisor_produces_one_workflow_not_many(tmp_path):
    """
    End-to-end, through the real store and the real tick: an hourly objective,
    MR1 off for two days, one workflow on return.
    """
    root = tmp_path / "runtime"
    clock = VirtualClock(start=START)
    planner = _RecordingPlanner()
    runner = MockRunner()

    objective_id = _recurring(
        root,
        clock,
        trigger={
            "type": "interval",
            "interval_s": 3600,
            "missed_run_policy": POLICY_CATCH_UP_ONCE,
        },
    )

    supervisor = _supervisor(root, clock, planner, runner)
    supervisor.tick()
    assert planner.calls == 1
    _drive_to_completion(supervisor, runner)
    supervisor.shutdown()

    # Two days of downtime: 48 occurrences elapse unobserved.
    clock.advance(2 * 86_400)

    revived = _supervisor(root, clock, planner, MockRunner())
    outcome = revived.tick()

    assert planner.calls == 2, (
        "48 missed occurrences must coalesce into exactly one make-up run, "
        "not 48 plans and 48 consent-authorized workflows"
    )
    assert outcome.get("missed_runs", 0) > 0, "and it must say it fell behind"

    objectives = ObjectiveStore(root, clock=clock)
    assert objectives.require(objective_id).catch_up_remaining == 0
    revived.shutdown()


def test_the_next_due_time_is_persisted_and_survives_restart(tmp_path):
    root = tmp_path / "runtime"
    clock = VirtualClock(start=START)
    objective_id = _recurring(
        root,
        clock,
        trigger={"type": "interval", "interval_s": 3600},
    )

    supervisor = _supervisor(root, clock, _RecordingPlanner())
    supervisor.tick()
    supervisor.shutdown()

    objectives = ObjectiveStore(root, clock=clock)
    next_due = objectives.require(objective_id).next_due_at
    assert next_due is not None

    # A fresh process reads the same schedule off disk.
    reopened = ObjectiveStore(root, clock=VirtualClock(start=START))
    assert reopened.require(objective_id).next_due_at == next_due


def test_an_ordinary_tick_still_costs_nothing(tmp_path):
    """The Phase-A invariant, re-pinned: recurrence must not make ticks expensive."""
    root = tmp_path / "runtime"
    clock = VirtualClock(start=START)
    planner = _RecordingPlanner()

    _recurring(root, clock, trigger={"type": "interval", "interval_s": 86_400})

    supervisor = _supervisor(root, clock, planner)
    first = supervisor.tick()  # fires once: never run before
    assert "error" not in first, first
    assert planner.calls == 1

    for _ in range(50):
        clock.advance(60)
        outcome = supervisor.tick()
        assert "error" not in outcome, outcome

    assert planner.calls == 1, "50 ticks inside the interval must call the brain zero times"
    supervisor.shutdown()


def test_evaluate_trigger_fails_closed_on_a_broken_spec(tmp_path):
    """An unschedulable trigger creates no work. It does not crash the tick."""
    from mr1.autonomy.objectives import Objective

    objective = Objective(
        objective_id="obj-x",
        title="broken",
        statement="do a thing",
        owner_agent_id="ag-root",
        kind=KIND_RECURRING,
        trigger={"type": "cron", "expression": "not a cron"},
    )
    decision = evaluate_trigger(objective, now=START)
    assert decision.ready is False
    assert "invalid trigger" in decision.reason
