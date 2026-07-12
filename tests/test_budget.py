"""A5/A9 — the shared budget ledger."""

from __future__ import annotations

from datetime import datetime, timezone

from mr1.autonomy.budget import BudgetLedger, BudgetLimits
from mr1.clock import VirtualClock


def _ledger(tmp_path, **limits):
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return BudgetLedger(tmp_path, clock=clock, limits=BudgetLimits(**limits)), clock


def test_plans_are_capped_per_hour(tmp_path):
    ledger, _clock = _ledger(tmp_path, max_plans_per_hour=2)

    assert ledger.try_consume_plan().allowed is True
    assert ledger.try_consume_plan().allowed is True

    denied = ledger.try_consume_plan()
    assert denied.allowed is False
    assert denied.reason == "plan_rate_exhausted"
    assert denied.used == 2


def test_the_plan_window_rolls_forward(tmp_path):
    ledger, clock = _ledger(tmp_path, max_plans_per_hour=1)
    ledger.try_consume_plan()
    assert ledger.try_consume_plan().allowed is False

    clock.advance(3601)

    assert ledger.try_consume_plan().allowed is True


def test_per_objective_daily_workflow_cap(tmp_path):
    ledger, clock = _ledger(
        tmp_path,
        max_plans_per_hour=100,
        max_workflows_per_objective_per_day=2,
    )

    assert ledger.try_consume_plan("obj-a").allowed is True
    assert ledger.try_consume_plan("obj-a").allowed is True

    denied = ledger.try_consume_plan("obj-a")
    assert denied.allowed is False
    assert denied.reason == "objective_daily_workflow_limit"

    # Another objective is unaffected.
    assert ledger.try_consume_plan("obj-b").allowed is True

    clock.advance(86_401)
    assert ledger.try_consume_plan("obj-a").allowed is True


def test_a_denied_plan_does_not_spend_anything(tmp_path):
    """All-or-nothing: a plan that cannot submit must not burn a token first."""
    ledger, _clock = _ledger(
        tmp_path,
        max_plans_per_hour=10,
        max_workflows_per_objective_per_day=1,
    )
    ledger.try_consume_plan("obj-a")
    before = ledger.plans_this_hour()

    denied = ledger.try_consume_plan("obj-a")

    assert denied.allowed is False
    assert ledger.plans_this_hour() == before


def test_a_retry_consumes_a_workflow_slot_but_no_plan(tmp_path):
    ledger, _clock = _ledger(tmp_path, max_workflows_per_objective_per_day=2)

    assert ledger.try_consume_objective_workflow("obj-a").allowed is True

    assert ledger.plans_this_hour() == 0
    assert ledger.snapshot()["workflows_today_by_objective"]["obj-a"] == 1


def test_actions_are_capped_per_hour(tmp_path):
    ledger, _clock = _ledger(tmp_path, max_actions_per_hour=3)

    assert ledger.try_consume_action(2).allowed is True

    denied = ledger.try_consume_action(2)
    assert denied.allowed is False
    assert denied.reason == "action_rate_exhausted"

    assert ledger.try_consume_action(1).allowed is True


def test_the_ledger_is_shared_across_processes(tmp_path):
    first, clock = _ledger(tmp_path, max_plans_per_hour=2)
    second = BudgetLedger(
        tmp_path,
        clock=clock,
        limits=BudgetLimits(max_plans_per_hour=2),
    )

    first.try_consume_plan()
    second.try_consume_plan()

    assert first.try_consume_plan().allowed is False
    assert second.plans_this_hour() == 2


def test_the_ledger_file_stays_bounded(tmp_path):
    ledger, clock = _ledger(tmp_path, max_plans_per_hour=1_000_000)
    for _ in range(50):
        ledger.try_consume_plan()
        clock.advance(120)

    # 50 stamps written 2 minutes apart, but only the ones inside the rolling
    # hour survive the prune — the file cannot grow without bound.
    assert ledger.plans_this_hour() == 29


def test_a_corrupt_ledger_fails_closed(tmp_path):
    ledger, _clock = _ledger(tmp_path, max_plans_per_hour=5)
    ledger.try_consume_plan()
    ledger.path.write_text("{ not json", encoding="utf-8")

    # A ledger it cannot read must not be treated as "nothing has been spent".
    assert ledger.try_consume_plan().allowed is False


def test_snapshot_reports_limits_and_usage(tmp_path):
    ledger, _clock = _ledger(tmp_path, max_plans_per_hour=9, max_actions_per_hour=8)
    ledger.try_consume_plan("obj-a")
    ledger.try_consume_action(2)

    snapshot = ledger.snapshot()

    assert snapshot["plans_this_hour"] == 1
    assert snapshot["actions_this_hour"] == 2
    assert snapshot["max_plans_per_hour"] == 9
    assert snapshot["max_actions_per_hour"] == 8
    assert snapshot["workflows_today_by_objective"] == {"obj-a": 1}
