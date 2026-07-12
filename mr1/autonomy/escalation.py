"""
Escalation — how MR1 asks for a human.

An autonomous system without an escalation path has only two failure modes:
infinite retry, or silent death. This module is the third option.

Every escalation does four things, always together:

  1. parks the objective in an explicit state (waiting_human / quarantined)
  2. sends a message to Marwan's inbox (the existing MessageStore)
  3. emits an `escalation_raised` timeline event
  4. says, in the message body: what MR1 was trying to do, what happened, what
     it already tried, what authority or decision it needs, and what it
     recommends next

MR1 never resolves an escalation by itself. In particular it never grants
itself the consent it is asking for — that is the whole point of asking.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mr1.autonomy.objectives import (
    STATUS_QUARANTINED,
    STATUS_WAITING_HUMAN,
    Objective,
    ObjectiveStore,
)
from mr1.clock import Clock, default_clock
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore


# Why a human is needed. Each maps to a different ask.
REASON_CONSENT_MISSING = "consent_missing"
REASON_CONSENT_EXPIRED = "consent_expired"
REASON_APPROVAL_EXPIRED = "approval_expired"
REASON_BLOCKED = "blocked"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"
REASON_REPEATED_FAILURE = "repeated_failure"
REASON_QUARANTINED = "quarantined"
REASON_AMBIGUOUS = "ambiguous_objective"
REASON_PLAN_FAILED = "plan_failed"
REASON_HEALTH_DEGRADED = "health_degraded"
REASON_STUCK = "stuck"

_RECOMMENDATIONS = {
    REASON_CONSENT_MISSING: (
        "Grant standing consent if this is expected:\n"
        "  mr1 grant create --objective {objective_id} --capability {capability} \\\n"
        "                   --scope <path> --allow '<regex>' --ttl 7d\n"
        "Or abandon the objective:  mr1 objective abandon {objective_id}"
    ),
    REASON_CONSENT_EXPIRED: (
        "The standing consent for this objective has expired. Re-grant it:\n"
        "  mr1 grant create --objective {objective_id} --capability {capability} \\\n"
        "                   --scope <path> --allow '<regex>' --ttl 7d"
    ),
    REASON_APPROVAL_EXPIRED: (
        "The one-off approval timed out with nobody to answer it. Either decide the\n"
        "next request promptly, or grant standing consent so this stops asking:\n"
        "  mr1 grant create --objective {objective_id} --capability {capability} --ttl 7d"
    ),
    REASON_BLOCKED: (
        "MR1 cannot self-authorize. Review the block, then either approve the request\n"
        "(mr1 approvals list) or grant consent (mr1 grant create)."
    ),
    REASON_BUDGET_EXHAUSTED: (
        "The recovery budget is spent. Inspect the history, fix the underlying cause,\n"
        "then resume:  mr1 objective resume {objective_id}"
    ),
    REASON_REPEATED_FAILURE: (
        "The same failure keeps recurring; retrying will not help. Fix the cause, then:\n"
        "  mr1 objective resume {objective_id}"
    ),
    REASON_QUARANTINED: (
        "The objective is quarantined and will not be retried. Resume it after fixing\n"
        "the cause:  mr1 objective resume {objective_id}\n"
        "Or drop it:  mr1 objective abandon {objective_id}"
    ),
    REASON_AMBIGUOUS: (
        "MR1 could not turn this objective into a workflow it trusts. Rewrite the\n"
        "statement to be more concrete, then resume it."
    ),
    REASON_PLAN_FAILED: (
        "Planning failed. Check the planner error below, then resume:\n"
        "  mr1 objective resume {objective_id}"
    ),
    REASON_HEALTH_DEGRADED: (
        "Runtime health is in error, so planning has stopped. Run `mr1 doctor` and fix\n"
        "the failing check; planning resumes automatically once health recovers."
    ),
    REASON_STUCK: (
        "This work has been stuck past its threshold. Inspect the workflow, then either\n"
        "cancel it (mr1 cancel-workflow <id>) or resume the objective."
    ),
}

_STATUS_FOR_REASON = {
    REASON_BUDGET_EXHAUSTED: STATUS_QUARANTINED,
    REASON_REPEATED_FAILURE: STATUS_QUARANTINED,
    REASON_QUARANTINED: STATUS_QUARANTINED,
}


@dataclass(frozen=True)
class Escalation:
    escalation_id: str
    objective_id: str
    reason: str
    summary: str
    body: str
    message_id: Optional[str]
    status: str
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalation_id": self.escalation_id,
            "objective_id": self.objective_id,
            "reason": self.reason,
            "summary": self.summary,
            "message_id": self.message_id,
            "status": self.status,
            "at": self.at,
        }


class Escalator:
    def __init__(
        self,
        runtime_root: Path,
        *,
        objective_store: ObjectiveStore,
        message_store: MessageStore,
        scoped_agent_store: PersistentAgentStore,
        clock: Optional[Clock] = None,
        event_log: Optional[EventLog] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._objectives = objective_store
        self._messages = message_store
        self._scoped_agents = scoped_agent_store
        self._clock = clock or default_clock()
        self._event_log = event_log or EventLog(self._runtime_root / "events")

    def escalate(
        self,
        objective: Objective,
        *,
        reason: str,
        problem: str,
        attempted: Optional[list[str]] = None,
        needs: str,
        capability: Optional[str] = None,
        workflow_id: Optional[str] = None,
        approval_request_id: Optional[str] = None,
        status: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Escalation:
        """
        Park the objective and tell the human. Never silent, never duplicated.

        Re-escalating the *same* condition for the same objective is a no-op
        beyond the state change: a supervisor that ticks every 60s must not
        send 60 identical messages an hour.
        """
        target_status = status or _STATUS_FOR_REASON.get(reason, STATUS_WAITING_HUMAN)
        escalation_id = self._escalation_id(objective.objective_id, reason, problem)
        already_raised = (
            (objective.last_escalation or {}).get("escalation_id") == escalation_id
            and objective.status == target_status
        )

        summary = f"{objective.title or objective.objective_id}: {reason.replace('_', ' ')}"
        body = self._build_body(
            objective,
            reason=reason,
            problem=problem,
            attempted=attempted or [],
            needs=needs,
            capability=capability,
            workflow_id=workflow_id,
            approval_request_id=approval_request_id,
            extra=extra or {},
        )

        message_id: Optional[str] = None
        if not already_raised:
            message_id = self._send(objective, summary, body, workflow_id)
            self._emit(
                objective,
                escalation_id=escalation_id,
                reason=reason,
                summary=summary,
                message_id=message_id,
                workflow_id=workflow_id,
                approval_request_id=approval_request_id,
                target_status=target_status,
            )

        escalation = Escalation(
            escalation_id=escalation_id,
            objective_id=objective.objective_id,
            reason=reason,
            summary=summary,
            body=body,
            message_id=message_id,
            status=target_status,
            at=self._clock.now_iso(),
        )
        self._objectives.update(
            objective.objective_id,
            status=target_status,
            status_reason=f"{reason}: {problem}"[:300],
            current_workflow_id=None,
            next_attempt_at=None,
            last_escalation=escalation.to_dict(),
        )
        return escalation

    # -- internals -----------------------------------------------------

    def _escalation_id(self, objective_id: str, reason: str, problem: str) -> str:
        raw = f"{objective_id}|{reason}|{problem[:200]}"
        return f"esc-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"

    def _build_body(
        self,
        objective: Objective,
        *,
        reason: str,
        problem: str,
        attempted: list[str],
        needs: str,
        capability: Optional[str],
        workflow_id: Optional[str],
        approval_request_id: Optional[str],
        extra: dict[str, Any],
    ) -> str:
        recommendation = _RECOMMENDATIONS.get(reason, "Inspect the objective and decide.").format(
            objective_id=objective.objective_id,
            capability=capability or "<capability>",
        )
        lines = [
            f"MR1 needs you on objective {objective.objective_id}.",
            "",
            f"OBJECTIVE:  {objective.title or objective.objective_id}",
            f"            {objective.statement}",
            f"STATUS:     {objective.status} -> {_STATUS_FOR_REASON.get(reason, STATUS_WAITING_HUMAN)}",
            f"REASON:     {reason}",
            "",
            "WHAT HAPPENED",
            f"  {problem}",
        ]
        if workflow_id:
            lines.append(f"  workflow: {workflow_id}")
        if approval_request_id:
            lines.append(f"  approval: {approval_request_id}")

        lines.extend(["", "WHAT MR1 TRIED"])
        if attempted:
            lines.extend(f"  - {item}" for item in attempted)
        else:
            lines.append("  - nothing yet; this was blocked before any work ran")

        history = objective.history[-5:]
        if history:
            lines.extend(["", "RECENT ATTEMPTS"])
            for item in history:
                lines.append(
                    f"  {item.at[:19]}  {item.outcome:<10} "
                    f"{item.classification or '-':<10} {item.action or '-'}"
                )

        lines.extend([
            "",
            "WHAT MR1 NEEDS FROM YOU",
            f"  {needs}",
            "",
            "RECOMMENDED NEXT ACTION",
        ])
        lines.extend(f"  {line}" for line in recommendation.splitlines())

        if extra:
            lines.extend(["", "DETAIL"])
            for key, value in sorted(extra.items()):
                lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "MR1 will not proceed on this objective until you act. It cannot grant",
            "itself the authority it is asking for.",
        ])
        return "\n".join(lines)

    def _send(
        self,
        objective: Objective,
        summary: str,
        body: str,
        workflow_id: Optional[str],
    ) -> Optional[str]:
        root_id = self._scoped_agents.root_agent_id
        try:
            message = self._messages.create_message(
                from_agent_id=objective.owner_agent_id or root_id,
                to_agent_id=root_id,
                kind="alert",
                subject=f"[MR1] {summary}"[:200],
                body=body,
                workflow_id=workflow_id,
            )
            return message.message_id
        except Exception:
            # The inbox failing must not swallow the escalation: the timeline
            # event and the objective state change still happen.
            return None

    def _emit(
        self,
        objective: Objective,
        *,
        escalation_id: str,
        reason: str,
        summary: str,
        message_id: Optional[str],
        workflow_id: Optional[str],
        approval_request_id: Optional[str],
        target_status: str,
    ) -> None:
        try:
            self._event_log.emit(
                event_type="escalation_raised",
                actor_id=objective.owner_agent_id or self._scoped_agents.root_agent_id,
                actor_type="mr1",
                target_id=objective.objective_id,
                target_type="objective",
                status=target_status,
                summary=f"escalation: {summary}"[:300],
                workflow_id=workflow_id,
                message_id=message_id,
                approval_request_id=approval_request_id,
                record_path=str(self._objectives.objective_path(objective.objective_id)),
                metadata={
                    "escalation_id": escalation_id,
                    "objective_id": objective.objective_id,
                    "reason": reason,
                    "target_status": target_status,
                    "message_id": message_id,
                },
            )
        except Exception:
            pass
