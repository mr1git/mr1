"""
A8 — the 10 000-tick soak.

The one that must exist before autonomy ships. Ten thousand supervisor ticks at
a 60s cadence is about a week of simulated uptime, driven by a recurring
objective so the system is never idle in the trivial sense.

What it proves:
  * no stuck objectives
  * no infinite retry
  * the heartbeat never goes stale
  * budgets stay enforced
  * memory/state growth stays bounded
  * tick cost does not creep with tick number (no O(n) in the loop)
"""

from __future__ import annotations

import statistics

import pytest

from mr1.autonomy.budget import BudgetLimits
from mr1.autonomy.health import heartbeat_is_stale
from mr1.autonomy.objectives import KIND_RECURRING
from mr1.autonomy.service import SupervisorConfig
from tests.soak.harness import SoakRuntime

TICKS = 10_000
TICK_INTERVAL_S = 60.0


@pytest.fixture(scope="module")
def soak(tmp_path_factory):
    """Ten thousand ticks, run once, asserted many ways."""
    tmp_path = tmp_path_factory.mktemp("soak10k")
    runtime = SoakRuntime(
        tmp_path / "runtime",
        workspace_root=tmp_path,
        config=SupervisorConfig(
            tick_interval_s=TICK_INTERVAL_S,
            doctor_interval_s=86_400.0,
            max_concurrent_workflows=4,
            max_plans_per_hour=1_000,
            max_workflows_per_objective_per_day=1_000,
            tick_event_interval=60,
        ),
        limits=BudgetLimits(
            max_plans_per_hour=1_000,
            max_actions_per_hour=1_000,
            max_workflows_per_objective_per_day=1_000,
        ),
    )
    # A recurring objective that comes due every 30 simulated minutes, so the
    # loop is doing real work — planning, submitting, reconciling — throughout.
    runtime.create_objective(
        title="genesis",
        statement="run the genesis cycle",
        kind=KIND_RECURRING,
        trigger={"type": "interval", "interval_s": 1_800},
    )
    runtime.run(TICKS)
    return runtime


def test_it_actually_ran_ten_thousand_ticks(soak):
    assert len(soak.samples) == TICKS
    assert soak.supervisor.metrics()["supervisor_tick_count"] == TICKS
    assert soak.supervisor.metrics()["supervisor_tick_errors"] == 0


def test_the_objective_kept_making_progress(soak):
    objective = soak.objectives.list_objectives()[0]
    # ~10 000 ticks * 60s = ~166h; one run per 30min => a lot of completed runs.
    assert objective.success_count > 100, f"only {objective.success_count} runs completed"
    assert objective.failure_count == 0
    assert soak.planner.call_count == objective.success_count


def test_no_stuck_objectives(soak):
    assert soak.stuck_objectives() == []


def test_no_infinite_retry(soak):
    objective = soak.objectives.list_objectives()[0]
    assert objective.consecutive_failures == 0
    assert objective.retries_used == 0
    assert objective.replans_used == 0


def test_the_heartbeat_is_never_stale(soak):
    payload = soak.health()
    assert payload["supervisor_heartbeat_at"]
    assert heartbeat_is_stale(
        payload,
        tick_interval_s=TICK_INTERVAL_S,
        clock=soak.clock,
    ) is False


def test_the_budget_stayed_enforced(soak):
    snapshot = soak.supervisor.budget.snapshot()
    assert snapshot["plans_this_hour"] <= snapshot["max_plans_per_hour"]
    assert snapshot["actions_this_hour"] <= snapshot["max_actions_per_hour"]
    for count in snapshot["workflows_today_by_objective"].values():
        assert count <= snapshot["max_workflows_per_objective_per_day"]


def test_objective_history_is_bounded(soak):
    """Hundreds of runs, but the objective file cannot grow without limit."""
    objective = soak.objectives.list_objectives()[0]
    assert len(objective.history) <= 25


def test_the_timeline_does_not_grow_per_tick(soak):
    """
    The per-tick pulse lives in health.json, which is rewritten — not appended.

    If every tick emitted an event, events.jsonl would carry 10 000 rows of
    "nothing happened" and every tick would get slower than the last.
    """
    tick_events = soak.events(event_type="supervisor_tick")
    assert len(tick_events) < TICKS / 10


def test_state_growth_is_bounded(soak):
    """Disk grows with *work done*, not with ticks."""
    objective_files = list((soak.runtime_root / "objectives").glob("obj-*.json"))
    assert len(objective_files) == 1

    health_bytes = (soak.runtime_root / "health.json").stat().st_size
    assert health_bytes < 8_000, "health.json is rewritten, so it must stay small"

    budget_bytes = (soak.runtime_root / "autonomy_budget.json").stat().st_size
    assert budget_bytes < 200_000, "the budget ledger prunes to its window"

    # The event log grows with completed workflows, which is expected and
    # bounded per unit of work — but it must not be growing per idle tick.
    objective = soak.objectives.list_objectives()[0]
    bytes_per_run = soak.event_bytes() / max(1, objective.success_count)
    assert bytes_per_run < 20_000, f"{bytes_per_run:.0f} bytes of events per run"


def test_tick_cost_does_not_grow_with_tick_number(soak):
    """
    The O(n) creep detector.

    Compares the first 500 ticks against the last 500. A loop that re-reads a
    growing event log, or re-scans every workflow ever created, shows up here as
    the last ticks costing several times the first.
    """
    first = [sample.wall_ms for sample in soak.samples[:500]]
    last = [sample.wall_ms for sample in soak.samples[-500:]]

    first_median = statistics.median(first)
    last_median = statistics.median(last)

    assert last_median < max(first_median * 3.0, first_median + 5.0), (
        f"tick cost crept: first 500 median {first_median:.2f}ms, "
        f"last 500 median {last_median:.2f}ms"
    )


def test_the_supervisor_is_still_healthy_at_the_end(soak):
    gauges = soak.gauges()
    assert gauges["supervisor_tick_errors"] == 0
    assert gauges["scheduler_tick_errors"] == 0
    assert gauges["objectives_quarantined"] == 0
    assert gauges["objectives_waiting_human"] == 0
    assert soak.supervisor.runtime_errors == []
