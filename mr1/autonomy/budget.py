"""
Shared budgets.

Two things in MR1 can spend tokens and create work without a human in the
loop: the supervisor's PLAN phase, and the inbox-triage loop. They must draw
on the *same* ledger, or "paused" is not paused and a runaway loop in one is
invisible to the cap on the other.

The ledger is a file under a `flock`, so a `mr1 serve` process and an open
REPL share one budget rather than two.

Windows:
  * plans   — per hour   (the token/cost ceiling on brain calls)
  * actions — per hour   (triage actions, agent runs, anything unattended)
  * objective workflows — per day, per objective
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mr1.clock import Clock, default_clock, parse_iso


BUDGET_FILE_NAME = "autonomy_budget.json"

_HOUR_S = 3600.0
_DAY_S = 86_400.0


@dataclass(frozen=True)
class BudgetLimits:
    max_plans_per_hour: int = 20
    max_actions_per_hour: int = 60
    max_workflows_per_objective_per_day: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_plans_per_hour": self.max_plans_per_hour,
            "max_actions_per_hour": self.max_actions_per_hour,
            "max_workflows_per_objective_per_day": self.max_workflows_per_objective_per_day,
        }


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str = ""
    used: int = 0
    limit: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "used": self.used,
            "limit": self.limit,
        }


class BudgetLedger:
    def __init__(
        self,
        runtime_root: Path,
        *,
        clock: Optional[Clock] = None,
        limits: Optional[BudgetLimits] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or default_clock()
        self._limits = limits or BudgetLimits()
        self._lock_path = self._runtime_root / ".autonomy_budget.lock"

    @property
    def path(self) -> Path:
        return self._runtime_root / BUDGET_FILE_NAME

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    # -- spending ------------------------------------------------------

    def try_consume_plan(self, objective_id: Optional[str] = None) -> BudgetDecision:
        """
        Reserve one brain call, and (if an objective is named) one of that
        objective's daily workflow slots. All-or-nothing: a plan that cannot
        also afford to submit its workflow must not burn a token first.
        """
        with self._locked():
            state = self._read()
            now = self._clock.now_iso()
            plans = self._within(state.get("plans", []), _HOUR_S)

            if len(plans) >= self._limits.max_plans_per_hour:
                return BudgetDecision(
                    allowed=False,
                    reason="plan_rate_exhausted",
                    used=len(plans),
                    limit=self._limits.max_plans_per_hour,
                )

            per_objective = dict(state.get("objective_workflows", {}))
            if objective_id:
                stamps = self._within(per_objective.get(objective_id, []), _DAY_S)
                if len(stamps) >= self._limits.max_workflows_per_objective_per_day:
                    return BudgetDecision(
                        allowed=False,
                        reason="objective_daily_workflow_limit",
                        used=len(stamps),
                        limit=self._limits.max_workflows_per_objective_per_day,
                    )
                per_objective[objective_id] = stamps + [now]

            state["plans"] = plans + [now]
            state["objective_workflows"] = per_objective
            self._write(state)
            return BudgetDecision(
                allowed=True,
                used=len(plans) + 1,
                limit=self._limits.max_plans_per_hour,
            )

    def try_consume_action(self, count: int = 1) -> BudgetDecision:
        """Reserve `count` unattended actions (triage, agent runs, replays)."""
        with self._locked():
            state = self._read()
            actions = self._within(state.get("actions", []), _HOUR_S)
            if len(actions) + count > self._limits.max_actions_per_hour:
                return BudgetDecision(
                    allowed=False,
                    reason="action_rate_exhausted",
                    used=len(actions),
                    limit=self._limits.max_actions_per_hour,
                )
            now = self._clock.now_iso()
            state["actions"] = actions + [now] * count
            self._write(state)
            return BudgetDecision(
                allowed=True,
                used=len(actions) + count,
                limit=self._limits.max_actions_per_hour,
            )

    def try_consume_objective_workflow(self, objective_id: str) -> BudgetDecision:
        """A workflow submission that did not need a brain call (a retry)."""
        with self._locked():
            state = self._read()
            per_objective = dict(state.get("objective_workflows", {}))
            stamps = self._within(per_objective.get(objective_id, []), _DAY_S)
            if len(stamps) >= self._limits.max_workflows_per_objective_per_day:
                return BudgetDecision(
                    allowed=False,
                    reason="objective_daily_workflow_limit",
                    used=len(stamps),
                    limit=self._limits.max_workflows_per_objective_per_day,
                )
            per_objective[objective_id] = stamps + [self._clock.now_iso()]
            state["objective_workflows"] = per_objective
            self._write(state)
            return BudgetDecision(
                allowed=True,
                used=len(stamps) + 1,
                limit=self._limits.max_workflows_per_objective_per_day,
            )

    # -- reads ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        state = self._read()
        plans = self._within(state.get("plans", []), _HOUR_S)
        actions = self._within(state.get("actions", []), _HOUR_S)
        per_objective = {
            objective_id: len(self._within(stamps, _DAY_S))
            for objective_id, stamps in dict(state.get("objective_workflows", {})).items()
        }
        return {
            "plans_this_hour": len(plans),
            "actions_this_hour": len(actions),
            "workflows_today_by_objective": per_objective,
            **self._limits.to_dict(),
        }

    def plans_this_hour(self) -> int:
        return len(self._within(self._read().get("plans", []), _HOUR_S))

    def actions_this_hour(self) -> int:
        return len(self._within(self._read().get("actions", []), _HOUR_S))

    # -- internals -----------------------------------------------------

    def _within(self, stamps: Any, window_s: float) -> list[str]:
        """Prune to the live window. This is also what keeps the file bounded."""
        now = self._clock.now()
        kept: list[str] = []
        for item in list(stamps or []):
            parsed = parse_iso(item) if isinstance(item, str) else None
            if parsed is None:
                continue
            if (now - parsed).total_seconds() < window_s:
                kept.append(item)
        return kept

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"plans": [], "actions": [], "objective_workflows": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            # A corrupt ledger must not be read as "no spending has happened".
            # Fail closed by reporting a full hour of plans until it is rewritten.
            return {
                "plans": [self._clock.now_iso()] * self._limits.max_plans_per_hour,
                "actions": [],
                "objective_workflows": {},
            }
        if not isinstance(payload, dict):
            return {"plans": [], "actions": [], "objective_workflows": {}}
        return payload

    def _write(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)

    def _locked(self):
        return _FileLock(self._lock_path)


class _FileLock:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._handle = None

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a+b")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
