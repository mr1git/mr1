"""
The failure ladder.

One rule governs everything here:

    A failure escalates exactly one level per exhausted budget.
    Every level terminates. Nothing retries forever; nothing fails silently.

This module is pure. `classify()` is a function of a failed workflow, and
`decide()` is a function of (classification, budgets, history). No stores, no
clock reads, no LLM — which is what makes the recovery behaviour of an
autonomous system something you can actually unit-test.

Classes
-------
transient  timeout, infrastructure failure, tool unavailable, auth blip
           -> back off and resubmit the same workflow
planning   the task failed on its merits (bad command, wrong assumption)
           -> replan: the brain drafts a new workflow given the failure
blocked    approval required, consent missing/expired, scope denied, cancelled
           -> escalate. MR1 cannot self-authorize. Ever.
fatal      budget exhausted, the same failure repeated, spec invalid
           -> quarantine + escalate + stop trying
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from mr1.workflow_models import FAILED_TASK_STATUSES, TaskStatus, Workflow, WorkflowStatus


class FailureClass(str, Enum):
    TRANSIENT = "transient"
    PLANNING = "planning"
    BLOCKED = "blocked"
    FATAL = "fatal"


class RecoveryKind(str, Enum):
    RETRY = "retry"
    REPLAN = "replan"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"


# An in-flight task that died with the runtime (crash, restart, OOM) is
# reported as `infrastructure_failure`. Treating that as a real defect turns a
# restart into permanent failure, so it is transient — bounded by the retry
# budget like anything else.
_TRANSIENT_ERROR_TYPES = frozenset({
    "timeout",
    "infrastructure_failure",
    "internal_error",
    "auth_error",
    "tool_unavailable",
})

# MR1 cannot resolve any of these by trying again or by thinking harder. They
# need a human decision, so they escalate rather than burn budget.
_BLOCKED_ERROR_TYPES = frozenset({
    "approval_required",
    "policy_block",
    "consent_missing",
    "consent_expired",
    "scope_denied",
})


@dataclass(frozen=True)
class FailurePolicy:
    """Budgets. Every one of them terminates."""

    max_retries: int = 3
    max_replans: int = 2
    max_fallbacks: int = 1
    max_consecutive_failures: int = 5
    max_identical_failures: int = 3
    max_elapsed_s: float = 7 * 86_400.0
    backoff_base_s: float = 30.0
    backoff_factor: float = 4.0
    backoff_max_s: float = 3600.0
    backoff_jitter_ratio: float = 0.1

    def validate(self) -> "FailurePolicy":
        if self.max_retries < 0 or self.max_replans < 0 or self.max_fallbacks < 0:
            raise ValueError("recovery budgets must be >= 0")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if self.max_identical_failures < 1:
            raise ValueError("max_identical_failures must be >= 1")
        if self.backoff_base_s < 0:
            raise ValueError("backoff_base_s must be >= 0")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "max_replans": self.max_replans,
            "max_fallbacks": self.max_fallbacks,
            "max_consecutive_failures": self.max_consecutive_failures,
            "max_identical_failures": self.max_identical_failures,
            "max_elapsed_s": self.max_elapsed_s,
            "backoff_base_s": self.backoff_base_s,
            "backoff_factor": self.backoff_factor,
            "backoff_max_s": self.backoff_max_s,
            "backoff_jitter_ratio": self.backoff_jitter_ratio,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "FailurePolicy":
        payload = dict(data or {})
        defaults = cls()
        return cls(
            max_retries=int(payload.get("max_retries", defaults.max_retries)),
            max_replans=int(payload.get("max_replans", defaults.max_replans)),
            max_fallbacks=int(payload.get("max_fallbacks", defaults.max_fallbacks)),
            max_consecutive_failures=int(
                payload.get("max_consecutive_failures", defaults.max_consecutive_failures)
            ),
            max_identical_failures=int(
                payload.get("max_identical_failures", defaults.max_identical_failures)
            ),
            max_elapsed_s=float(payload.get("max_elapsed_s", defaults.max_elapsed_s)),
            backoff_base_s=float(payload.get("backoff_base_s", defaults.backoff_base_s)),
            backoff_factor=float(payload.get("backoff_factor", defaults.backoff_factor)),
            backoff_max_s=float(payload.get("backoff_max_s", defaults.backoff_max_s)),
            backoff_jitter_ratio=float(
                payload.get("backoff_jitter_ratio", defaults.backoff_jitter_ratio)
            ),
        ).validate()


@dataclass(frozen=True)
class FailureSignal:
    """What a failed workflow tells us, reduced to what recovery needs."""

    classification: FailureClass
    reason: str
    error_type: Optional[str] = None
    task_id: Optional[str] = None
    detail: str = ""

    @property
    def signature(self) -> str:
        """Identity of *this kind of failure*, for repeat detection."""
        raw = "|".join([
            self.classification.value,
            self.error_type or "",
            _normalize_detail(self.detail),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "reason": self.reason,
            "error_type": self.error_type,
            "task_id": self.task_id,
            "detail": self.detail[:500],
            "signature": self.signature,
        }


@dataclass(frozen=True)
class RecoveryState:
    """
    Everything `decide()` is allowed to look at.

    Deliberately not the `Objective` itself: recovery is a pure function of
    counters and history, so it can be tested exhaustively without a store.
    """

    retries_used: int = 0
    replans_used: int = 0
    fallbacks_used: int = 0
    consecutive_failures: int = 0
    elapsed_s: float = 0.0
    recent_signatures: list[str] = field(default_factory=list)
    has_fallback: bool = False


@dataclass(frozen=True)
class RecoveryAction:
    kind: RecoveryKind
    reason: str
    classification: FailureClass
    delay_s: float = 0.0
    escalate: bool = False
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "classification": self.classification.value,
            "delay_s": round(self.delay_s, 3),
            "escalate": self.escalate,
            "terminal": self.terminal,
        }


class _Rng(Protocol):
    def random(self) -> float: ...


def _normalize_detail(detail: str) -> str:
    """
    Strip the parts of an error message that differ between two runs of the
    *same* failure — ids, paths, timestamps, numbers — so "it failed the same
    way again" is detectable.
    """
    if not detail:
        return ""
    out: list[str] = []
    for token in detail.split():
        if any(char.isdigit() for char in token):
            continue
        out.append(token.strip(".,:;()[]'\"").lower())
    return " ".join(out[:20])


def classify(workflow: Workflow) -> FailureSignal:
    """
    Reduce a terminal workflow to one failure signal. Pure.

    Blocked wins over everything: if any task needs a human, no amount of
    retrying or replanning by MR1 can help, and pretending otherwise is how an
    autonomous system burns a budget going nowhere.
    """
    if workflow.status is WorkflowStatus.CANCELLED:
        return FailureSignal(
            classification=FailureClass.BLOCKED,
            reason="workflow_cancelled",
            error_type="cancelled",
            detail="workflow was cancelled",
        )
    if workflow.status is WorkflowStatus.SUCCEEDED:
        raise ValueError("classify() expects a failed workflow")

    failed = [
        task for task in workflow.tasks.values()
        if task.status in FAILED_TASK_STATUSES
    ]
    if not failed:
        return FailureSignal(
            classification=FailureClass.PLANNING,
            reason="workflow_failed_without_a_failed_task",
            detail="workflow reported failure with no failed task",
        )

    for task in failed:
        if task.status is TaskStatus.BLOCKED or task.last_error_type in _BLOCKED_ERROR_TYPES:
            return FailureSignal(
                classification=FailureClass.BLOCKED,
                reason=task.last_error_type or "blocked",
                error_type=task.last_error_type or "blocked",
                task_id=task.task_id,
                detail=task.last_error or task.blocked_reason or "task is blocked",
            )

    for task in failed:
        if task.status is TaskStatus.CANCELLED:
            return FailureSignal(
                classification=FailureClass.BLOCKED,
                reason="task_cancelled",
                error_type="cancelled",
                task_id=task.task_id,
                detail=task.last_error or "task was cancelled",
            )

    transient = [
        task for task in failed
        if task.status is TaskStatus.TIMED_OUT
        or (task.last_error_type or "") in _TRANSIENT_ERROR_TYPES
    ]
    if transient and len(transient) == len(failed):
        task = transient[0]
        return FailureSignal(
            classification=FailureClass.TRANSIENT,
            reason=task.last_error_type or "timeout",
            error_type=task.last_error_type or "timeout",
            task_id=task.task_id,
            detail=task.last_error or "transient failure",
        )

    merit_failure = next(
        (task for task in failed if task not in transient),
        failed[0],
    )
    return FailureSignal(
        classification=FailureClass.PLANNING,
        reason=merit_failure.last_error_type or "task_failed",
        error_type=merit_failure.last_error_type or "unknown",
        task_id=merit_failure.task_id,
        detail=merit_failure.last_error or merit_failure.result_summary or "task failed",
    )


def backoff_for(
    attempt: int,
    policy: FailurePolicy,
    *,
    rng: Optional[_Rng] = None,
) -> float:
    """30s, 2m, 8m … capped, optionally jittered. `rng=None` is deterministic."""
    if attempt < 1:
        attempt = 1
    delay = policy.backoff_base_s * (policy.backoff_factor ** (attempt - 1))
    delay = min(delay, policy.backoff_max_s)
    if rng is not None and policy.backoff_jitter_ratio > 0:
        jitter = delay * policy.backoff_jitter_ratio * rng.random()
        delay += jitter
    return float(delay)


def decide(
    signal: FailureSignal,
    state: RecoveryState,
    policy: FailurePolicy,
    *,
    rng: Optional[_Rng] = None,
) -> RecoveryAction:
    """
    The ladder. Every branch terminates.

    Order matters: the unconditional stops (consecutive failures, elapsed
    runtime, the same failure over and over) are checked before the per-class
    budgets, so no class of failure can outrun the global bounds.
    """
    if state.consecutive_failures >= policy.max_consecutive_failures:
        return RecoveryAction(
            kind=RecoveryKind.QUARANTINE,
            reason=(
                f"{state.consecutive_failures} consecutive failures "
                f"(limit {policy.max_consecutive_failures})"
            ),
            classification=FailureClass.FATAL,
            escalate=True,
            terminal=True,
        )

    if state.elapsed_s >= policy.max_elapsed_s:
        return RecoveryAction(
            kind=RecoveryKind.QUARANTINE,
            reason=(
                f"objective exceeded its {policy.max_elapsed_s:.0f}s runtime budget"
            ),
            classification=FailureClass.FATAL,
            escalate=True,
            terminal=True,
        )

    if signal.classification is FailureClass.BLOCKED:
        # MR1 cannot self-authorize. Not once, not ever, not under any budget.
        return RecoveryAction(
            kind=RecoveryKind.ESCALATE,
            reason=f"blocked: {signal.reason}",
            classification=FailureClass.BLOCKED,
            escalate=True,
            terminal=True,
        )

    identical = _identical_run_length(state.recent_signatures, signal.signature)
    if identical >= policy.max_identical_failures:
        return RecoveryAction(
            kind=RecoveryKind.QUARANTINE,
            reason=(
                f"the same failure repeated {identical} times "
                f"(limit {policy.max_identical_failures})"
            ),
            classification=FailureClass.FATAL,
            escalate=True,
            terminal=True,
        )

    if signal.classification is FailureClass.FATAL:
        return RecoveryAction(
            kind=RecoveryKind.QUARANTINE,
            reason=f"fatal: {signal.reason}",
            classification=FailureClass.FATAL,
            escalate=True,
            terminal=True,
        )

    if signal.classification is FailureClass.TRANSIENT:
        if state.retries_used < policy.max_retries:
            return RecoveryAction(
                kind=RecoveryKind.RETRY,
                reason=f"transient: {signal.reason}",
                classification=FailureClass.TRANSIENT,
                delay_s=backoff_for(state.retries_used + 1, policy, rng=rng),
            )
        # Retries exhausted: escalate exactly one level, to a replan.
        if state.replans_used < policy.max_replans:
            return RecoveryAction(
                kind=RecoveryKind.REPLAN,
                reason=(
                    f"retry budget exhausted after {state.retries_used} attempts; replanning"
                ),
                classification=FailureClass.TRANSIENT,
            )
        return _exhausted(state, policy, signal, "retry budget exhausted")

    # PLANNING
    if state.replans_used < policy.max_replans:
        return RecoveryAction(
            kind=RecoveryKind.REPLAN,
            reason=f"planning failure: {signal.reason}",
            classification=FailureClass.PLANNING,
        )
    return _exhausted(state, policy, signal, "replan budget exhausted")


def _exhausted(
    state: RecoveryState,
    policy: FailurePolicy,
    signal: FailureSignal,
    reason: str,
) -> RecoveryAction:
    """One more level: the fallback, if the objective has one. Then stop."""
    if state.has_fallback and state.fallbacks_used < policy.max_fallbacks:
        return RecoveryAction(
            kind=RecoveryKind.FALLBACK,
            reason=f"{reason}; trying the objective's fallback",
            classification=signal.classification,
        )
    return RecoveryAction(
        kind=RecoveryKind.QUARANTINE,
        reason=f"{reason}; no recovery left",
        classification=FailureClass.FATAL,
        escalate=True,
        terminal=True,
    )


def _identical_run_length(recent_signatures: list[str], signature: str) -> int:
    """How many times in a row this exact failure has now occurred."""
    run = 1
    for previous in reversed(list(recent_signatures)):
        if previous != signature:
            break
        run += 1
    return run
