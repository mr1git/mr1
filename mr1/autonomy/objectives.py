"""
Objectives — the thing that makes MR1 *want* something.

Nothing in today's runtime creates work: a human types, or nothing happens. An
objective is a persisted long-lived goal that the supervisor reconciles. It is
not an execution engine; it holds intent, authority, budgets, and history, and
the scheduler still does all the work.

Kinds
-----
once      a mission with a completion criterion; runs until satisfied
recurring fires on a trigger (Genesis's weekly cycle is this)
standing  a continuously-reconciled responsibility

Unattended safety
-----------------
`idempotent` is a claim the operator makes about the mission, and the recovery
ladder relies on it: a transient failure resubmits the same workflow, which is
only safe if re-running it is harmless. It is recorded on the objective rather
than assumed.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from mr1.autonomy.recovery import FailurePolicy, RecoveryState
from mr1.autonomy.triggers import (
    TriggerDecision,
    TriggerError,
    evaluate_recurrence,
    validate_trigger,
)
from mr1.clock import Clock, default_clock, parse_iso


OBJECTIVES_DIR_NAME = "objectives"

KIND_ONCE = "once"
KIND_RECURRING = "recurring"
KIND_STANDING = "standing"
OBJECTIVE_KINDS = (KIND_ONCE, KIND_RECURRING, KIND_STANDING)

STATUS_ACTIVE = "active"
STATUS_PLANNING = "planning"
STATUS_EXECUTING = "executing"
STATUS_RECOVERING = "recovering"
STATUS_WAITING_HUMAN = "waiting_human"
STATUS_SATISFIED = "satisfied"
STATUS_QUARANTINED = "quarantined"
STATUS_ABANDONED = "abandoned"
STATUS_PAUSED = "paused"

OBJECTIVE_STATUSES = (
    STATUS_ACTIVE,
    STATUS_PLANNING,
    STATUS_EXECUTING,
    STATUS_RECOVERING,
    STATUS_WAITING_HUMAN,
    STATUS_SATISFIED,
    STATUS_QUARANTINED,
    STATUS_ABANDONED,
    STATUS_PAUSED,
)

# States MR1 will not leave on its own. Every one of them was reached by
# telling a human why (A7) — none of them is a silent stop.
TERMINAL_STATUSES = frozenset({STATUS_SATISFIED, STATUS_ABANDONED})
PARKED_STATUSES = frozenset({STATUS_WAITING_HUMAN, STATUS_QUARANTINED, STATUS_PAUSED})

_MAX_HISTORY = 25


class ObjectiveError(ValueError):
    pass


def new_objective_id() -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"obj-{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class Attempt:
    workflow_id: Optional[str]
    outcome: str
    at: str
    classification: Optional[str] = None
    signature: Optional[str] = None
    action: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "outcome": self.outcome,
            "at": self.at,
            "classification": self.classification,
            "signature": self.signature,
            "action": self.action,
            "detail": self.detail[:300],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attempt":
        return cls(
            workflow_id=data.get("workflow_id"),
            outcome=data.get("outcome", "unknown"),
            at=data.get("at", ""),
            classification=data.get("classification"),
            signature=data.get("signature"),
            action=data.get("action"),
            detail=str(data.get("detail") or ""),
        )


@dataclass
class Objective:
    objective_id: str
    title: str
    statement: str
    owner_agent_id: str
    kind: str = KIND_ONCE
    trigger: dict[str, Any] = field(default_factory=lambda: {"type": "immediate"})
    status: str = STATUS_ACTIVE
    consent_grant_ids: list[str] = field(default_factory=list)
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)
    current_workflow_id: Optional[str] = None
    last_spec: Optional[dict[str, Any]] = None
    fallback_statement: Optional[str] = None
    idempotent: bool = False
    consecutive_failures: int = 0
    retries_used: int = 0
    replans_used: int = 0
    fallbacks_used: int = 0
    success_count: int = 0
    failure_count: int = 0
    history: list[Attempt] = field(default_factory=list)
    last_failure: Optional[dict[str, Any]] = None
    last_escalation: Optional[dict[str, Any]] = None
    status_reason: str = ""
    created_at: str = ""
    updated_at: str = ""
    first_attempt_at: Optional[str] = None
    last_planned_at: Optional[str] = None
    last_completed_at: Optional[str] = None
    next_attempt_at: Optional[str] = None
    use_fallback_next: bool = False
    # B2. Recurrence state, persisted so it survives restart. `next_due_at` is
    # when this objective is next scheduled to fire; `catch_up_remaining` is the
    # bounded backlog of missed runs still owed after downtime — the counter
    # that stops a week-long outage from firing a week of workflows at once.
    next_due_at: Optional[str] = None
    catch_up_remaining: int = 0
    last_fired_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in OBJECTIVE_KINDS:
            raise ObjectiveError(f"unknown objective kind '{self.kind}'")
        if self.status not in OBJECTIVE_STATUSES:
            raise ObjectiveError(f"unknown objective status '{self.status}'")
        if not self.statement.strip():
            raise ObjectiveError("an objective must state what it wants")
        if not isinstance(self.trigger, dict) or not self.trigger.get("type"):
            raise ObjectiveError("an objective must carry a trigger with a type")

    # -- lifecycle predicates ------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_parked(self) -> bool:
        return self.status in PARKED_STATUSES

    @property
    def is_live(self) -> bool:
        """Still MR1's problem: not parked, not finished."""
        return not self.is_terminal and not self.is_parked

    def recovery_state(self, now: datetime) -> RecoveryState:
        started = parse_iso(self.first_attempt_at)
        elapsed = (now - started).total_seconds() if started else 0.0
        return RecoveryState(
            retries_used=self.retries_used,
            replans_used=self.replans_used,
            fallbacks_used=self.fallbacks_used,
            consecutive_failures=self.consecutive_failures,
            elapsed_s=max(0.0, elapsed),
            recent_signatures=[
                attempt.signature
                for attempt in self.history
                if attempt.signature
            ],
            has_fallback=bool(self.fallback_statement),
        )

    def ready_to_retry(self, now: datetime) -> bool:
        if self.status != STATUS_RECOVERING:
            return False
        deadline = parse_iso(self.next_attempt_at)
        return deadline is None or now >= deadline

    def record_attempt(self, attempt: Attempt) -> None:
        self.history.append(attempt)
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_MAX_HISTORY:]

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "title": self.title,
            "statement": self.statement,
            "owner_agent_id": self.owner_agent_id,
            "kind": self.kind,
            "trigger": dict(self.trigger),
            "status": self.status,
            "consent_grant_ids": list(self.consent_grant_ids),
            "failure_policy": self.failure_policy.to_dict(),
            "current_workflow_id": self.current_workflow_id,
            "last_spec": dict(self.last_spec) if self.last_spec else None,
            "fallback_statement": self.fallback_statement,
            "idempotent": bool(self.idempotent),
            "consecutive_failures": int(self.consecutive_failures),
            "retries_used": int(self.retries_used),
            "replans_used": int(self.replans_used),
            "fallbacks_used": int(self.fallbacks_used),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "history": [item.to_dict() for item in self.history],
            "last_failure": dict(self.last_failure) if self.last_failure else None,
            "last_escalation": dict(self.last_escalation) if self.last_escalation else None,
            "status_reason": self.status_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "first_attempt_at": self.first_attempt_at,
            "last_planned_at": self.last_planned_at,
            "last_completed_at": self.last_completed_at,
            "next_attempt_at": self.next_attempt_at,
            "use_fallback_next": bool(self.use_fallback_next),
            "next_due_at": self.next_due_at,
            "catch_up_remaining": int(self.catch_up_remaining),
            "last_fired_at": self.last_fired_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Objective":
        return cls(
            objective_id=data["objective_id"],
            title=data.get("title", ""),
            statement=data["statement"],
            owner_agent_id=data["owner_agent_id"],
            kind=data.get("kind", KIND_ONCE),
            trigger=dict(data.get("trigger") or {"type": "immediate"}),
            status=data.get("status", STATUS_ACTIVE),
            consent_grant_ids=list(data.get("consent_grant_ids", [])),
            failure_policy=FailurePolicy.from_dict(data.get("failure_policy")),
            current_workflow_id=data.get("current_workflow_id"),
            last_spec=dict(data["last_spec"]) if data.get("last_spec") else None,
            fallback_statement=data.get("fallback_statement"),
            idempotent=bool(data.get("idempotent", False)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            retries_used=int(data.get("retries_used", 0)),
            replans_used=int(data.get("replans_used", 0)),
            fallbacks_used=int(data.get("fallbacks_used", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            history=[Attempt.from_dict(item) for item in data.get("history", [])],
            last_failure=dict(data["last_failure"]) if data.get("last_failure") else None,
            last_escalation=(
                dict(data["last_escalation"]) if data.get("last_escalation") else None
            ),
            status_reason=str(data.get("status_reason") or ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            first_attempt_at=data.get("first_attempt_at"),
            last_planned_at=data.get("last_planned_at"),
            last_completed_at=data.get("last_completed_at"),
            next_attempt_at=data.get("next_attempt_at"),
            use_fallback_next=bool(data.get("use_fallback_next", False)),
            next_due_at=data.get("next_due_at"),
            catch_up_remaining=int(data.get("catch_up_remaining", 0)),
            last_fired_at=data.get("last_fired_at"),
        )


class ObjectiveStore:
    """
    One JSON file per objective, `flock`-serialised, atomically replaced.

    Same discipline as `WorkflowStore`: a supervisor and a REPL may both be
    touching this, and a torn objective is an objective whose authority and
    failure counters cannot be trusted.
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        clock: Optional[Clock] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._root = self._runtime_root / OBJECTIVES_DIR_NAME
        self._root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or default_clock()
        self._lock_path = self._root / ".objectives.lock"

    @property
    def root(self) -> Path:
        return self._root

    def objective_path(self, objective_id: str) -> Path:
        return self._root / f"{objective_id}.json"

    # -- creation ------------------------------------------------------

    def create(
        self,
        *,
        title: str,
        statement: str,
        owner_agent_id: str,
        kind: str = KIND_ONCE,
        trigger: Optional[dict[str, Any]] = None,
        failure_policy: Optional[FailurePolicy] = None,
        consent_grant_ids: Optional[list[str]] = None,
        fallback_statement: Optional[str] = None,
        idempotent: bool = False,
        objective_id: Optional[str] = None,
    ) -> Objective:
        now = self._clock.now_iso()
        # Reject an unschedulable trigger here, where a human is watching, rather
        # than on the first tick of an unattended supervisor at 3am.
        resolved_trigger = validate_trigger(dict(trigger or _default_trigger(kind)))
        objective = Objective(
            objective_id=objective_id or new_objective_id(),
            title=title or statement[:60],
            statement=statement,
            owner_agent_id=owner_agent_id,
            kind=kind,
            trigger=resolved_trigger,
            failure_policy=(failure_policy or FailurePolicy()).validate(),
            consent_grant_ids=list(consent_grant_ids or []),
            fallback_statement=fallback_statement,
            idempotent=idempotent,
            created_at=now,
            updated_at=now,
        )
        with self._locked():
            if self.objective_path(objective.objective_id).exists():
                raise ObjectiveError(f"objective already exists: {objective.objective_id}")
            self._write(objective)
        return objective

    # -- reads ---------------------------------------------------------

    def load(self, objective_id: str) -> Optional[Objective]:
        path = self.objective_path(objective_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return Objective.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def require(self, objective_id: str) -> Objective:
        objective = self.load(objective_id)
        if objective is None:
            raise ObjectiveError(f"objective not found: {objective_id}")
        return objective

    def list_objectives(self) -> list[Objective]:
        objectives: list[Objective] = []
        for path in sorted(self._root.glob("obj-*.json")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    objectives.append(Objective.from_dict(json.load(handle)))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        objectives.sort(key=lambda item: (item.created_at, item.objective_id))
        return objectives

    def list_live(self) -> list[Objective]:
        return [item for item in self.list_objectives() if item.is_live]

    def counts_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for objective in self.list_objectives():
            counts[objective.status] = counts.get(objective.status, 0) + 1
        return counts

    # -- writes --------------------------------------------------------

    def save(self, objective: Objective) -> Objective:
        with self._locked():
            objective.updated_at = self._clock.now_iso()
            self._write(objective)
        return objective

    def update(self, objective_id: str, **changes: Any) -> Objective:
        """Read-modify-write under the cross-process lock."""
        with self._locked():
            objective = self.require(objective_id)
            for key, value in changes.items():
                if not hasattr(objective, key):
                    raise ObjectiveError(f"unknown objective field: {key}")
                setattr(objective, key, value)
            objective.updated_at = self._clock.now_iso()
            self._write(objective)
            return objective

    def set_status(
        self,
        objective_id: str,
        status: str,
        *,
        reason: str = "",
    ) -> Objective:
        if status not in OBJECTIVE_STATUSES:
            raise ObjectiveError(f"unknown objective status '{status}'")
        return self.update(objective_id, status=status, status_reason=reason)

    def pause_all(self, *, reason: str = "paused") -> list[str]:
        """Used by `halt`: every live objective stops wanting things."""
        paused: list[str] = []
        for objective in self.list_objectives():
            if objective.is_terminal or objective.status == STATUS_PAUSED:
                continue
            self.set_status(objective.objective_id, STATUS_PAUSED, reason=reason)
            paused.append(objective.objective_id)
        return paused

    def delete(self, objective_id: str) -> bool:
        with self._locked():
            path = self.objective_path(objective_id)
            if not path.exists():
                return False
            path.unlink()
            return True

    # -- internals -----------------------------------------------------

    def _write(self, objective: Objective) -> None:
        path = self.objective_path(objective.objective_id)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(objective.to_dict(), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)

    def _locked(self):
        return _FileLock(self._lock_path)


def _default_trigger(kind: str) -> dict[str, Any]:
    if kind == KIND_RECURRING:
        return {"type": "interval", "interval_s": 7 * 86_400.0}
    if kind == KIND_STANDING:
        return {"type": "interval", "interval_s": 3600.0}
    return {"type": "immediate"}


def recurrence_anchor(objective: Objective) -> Optional[datetime]:
    """
    When this objective's recurrence last fired.

    `last_fired_at` is written the moment the supervisor decides to plan, which
    is what makes a recurrence idempotent across a restart: an objective that
    fired at 09:00 and crashed at 09:01 does not fire again at 09:02, because
    the fire is recorded when it happens, not when it finishes.

    Objectives created before B2 have no `last_fired_at`, so fall back to the
    Phase-A anchors. Completion is preferred over planning: an interval means
    "this long after the last run finished".
    """
    return (
        parse_iso(objective.last_fired_at)
        or parse_iso(objective.last_completed_at)
        or parse_iso(objective.last_planned_at)
    )


def evaluate_trigger(
    objective: Objective,
    *,
    now: datetime,
    watcher_registry: Any = None,
) -> TriggerDecision:
    """
    Should this objective produce work right now, and what carries forward?

    Deterministic and brain-free — this runs on every tick, so it may never
    cost a token. Watcher-backed triggers reuse `WatcherRegistry` rather than
    inventing a second trigger engine; recurrence (interval, cron, and the
    missed-run policy) lives in `mr1.autonomy.triggers`.
    """
    trigger = dict(objective.trigger or {})
    kind = trigger.get("type")

    if kind == "manual":
        return TriggerDecision(
            ready=False,
            reason="manual trigger: waiting for `mr1 objective run`",
        )

    if kind == "immediate":
        if objective.last_completed_at and objective.kind == KIND_ONCE:
            return TriggerDecision(ready=False, reason="already completed")
        return TriggerDecision(ready=True, reason="immediate")

    if kind in {"interval", "cron"}:
        try:
            return evaluate_recurrence(
                trigger,
                anchor=recurrence_anchor(objective),
                now=now,
                catch_up_remaining=objective.catch_up_remaining,
            )
        except TriggerError as exc:
            # Fail closed: an unschedulable trigger creates no work.
            return TriggerDecision(ready=False, reason=f"invalid trigger: {exc}")

    if kind == "watcher":
        if watcher_registry is None:
            return TriggerDecision(
                ready=False,
                reason="watcher trigger requires a watcher registry",
            )
        from mr1.workflow_models import Task, TaskStatus

        probe = Task(
            task_id=f"trigger-{objective.objective_id}",
            workflow_id=objective.objective_id,
            label="trigger",
            title="objective trigger",
            task_kind="watcher",
            agent_type=None,
            prompt="",
            watcher_type=trigger.get("watcher_type"),
            watch_config=dict(trigger.get("watch_config") or {}),
            status=TaskStatus.RUNNING,
        )
        try:
            evaluation = watcher_registry.evaluate(probe, now)
        except Exception as exc:
            return TriggerDecision(ready=False, reason=f"watcher trigger error: {exc}")
        if evaluation.state == "satisfied":
            return TriggerDecision(ready=True, reason=f"watcher satisfied: {evaluation.message}")
        return TriggerDecision(
            ready=False,
            reason=f"watcher {evaluation.state}: {evaluation.message}",
        )

    return TriggerDecision(ready=False, reason=f"unknown trigger type '{kind}'")


def trigger_is_ready(
    objective: Objective,
    *,
    now: datetime,
    watcher_registry: Any = None,
) -> tuple[bool, str]:
    """The Phase-A boolean view of `evaluate_trigger`, for callers that only ask."""
    decision = evaluate_trigger(
        objective,
        now=now,
        watcher_registry=watcher_registry,
    )
    return decision.ready, decision.reason


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
