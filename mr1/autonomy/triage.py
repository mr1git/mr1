"""
Governed inbox triage (A9).

The inbox-triage loop is already autonomous today: every 30 seconds it makes an
LLM call that may take several actions — including `agent_run`, `assign_mission`,
and `create_workflow` — with no budget and no control-plane hook. It is the one
unattended surface nobody governs, and it predates the supervisor.

This module does not reimplement triage. It wraps the existing
`InboxTriageRunner` in the same gates everything else in Phase A goes through:

  * control plane — `pause` / `stopping` / `halted` stop new triage actions
  * shared budget — triage spends from the *same* ledger as the supervisor's
    planner, so a runaway loop in one is visible to the cap on the other
  * observability — a failure is recorded, emitted, and (if it keeps failing)
    escalated to a human rather than retried forever in silence
  * governance — triage-created workflows carry no objective_id, so they can
    never ride an objective's consent grant

Zero-cost when idle: with no unread mail, a triage pass makes no LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from mr1.autonomy.budget import BudgetLedger
from mr1.autonomy.control import ControlPlane
from mr1.clock import Clock, default_clock
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore


# After this many consecutive failures, triage stops being a warning and starts
# being an escalation: something is broken and retrying will not fix it.
_ESCALATE_AFTER_CONSECUTIVE_FAILURES = 3


@dataclass(frozen=True)
class TriageOutcome:
    ran: bool
    reason: str
    actions_executed: int = 0
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "reason": self.reason,
            "actions_executed": self.actions_executed,
            "summary": self.summary,
            "error": self.error,
        }


class GovernedTriage:
    """
    The gate in front of the existing triage runner.

    `runner_factory` returns an object with `.run(policy, caller_agent_id=...)`
    — i.e. the real `InboxTriageRunner`, constructed by whoever owns it (MR1 in
    the REPL, the supervisor when headless). Triage logic is not duplicated here.
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        message_store: MessageStore,
        scoped_agent_store: PersistentAgentStore,
        runner_factory: Callable[[], Any],
        control: Optional[ControlPlane] = None,
        budget: Optional[BudgetLedger] = None,
        clock: Optional[Clock] = None,
        event_log: Optional[EventLog] = None,
        error_sink: Optional[Callable[..., Any]] = None,
        escalate: Optional[Callable[[str, str], None]] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._messages = message_store
        self._scoped_agents = scoped_agent_store
        self._runner_factory = runner_factory
        self._clock = clock or default_clock()
        self._control = control or ControlPlane(self._runtime_root, clock=self._clock)
        self._budget = budget or BudgetLedger(self._runtime_root, clock=self._clock)
        self._event_log = event_log or EventLog(self._runtime_root / "events")
        self._error_sink = error_sink
        self._escalate = escalate
        self._consecutive_failures = 0
        self._skips: dict[str, int] = {}

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def skips(self) -> dict[str, int]:
        return dict(self._skips)

    # ------------------------------------------------------------------

    def _is_self_escalation(self, message: Any) -> bool:
        """
        A message MR1 sent to the human, sitting in MR1's own inbox.

        Root is both the escalation target and the inbox triage owns, so an
        escalation looks like unread mail to triage. Waking the brain over it
        would be a feedback loop — and worse, triage is allowed to `archive` and
        `mark_read`, so MR1 could quietly dispose of the very message asking
        Marwan for the consent it is blocked on. Escalations are for the human;
        triage does not get a vote on them.
        """
        root_id = self._scoped_agents.root_agent_id
        return (
            getattr(message, "kind", "") == "alert"
            and getattr(message, "from_agent_id", "") == root_id
            and getattr(message, "to_agent_id", "") == root_id
        )

    def unread_count(self) -> int:
        """Unread mail that triage is actually allowed to act on."""
        try:
            root_id = self._scoped_agents.root_agent_id
            return sum(
                1
                for message in self._messages.list_inbox(root_id)
                if message.status == "unread" and not self._is_self_escalation(message)
            )
        except Exception:
            return 0

    def gate(self) -> tuple[bool, str]:
        """
        May triage act right now? Checked before every pass.

        `paused`, `stopping`, and `halted` all stop *new* triage actions. There
        is nothing to drain — a triage pass is atomic — so stopping and paused
        behave the same here: no new actions.
        """
        state = self._control.read()
        if not state.planning_allowed:
            return False, f"control_mode:{state.mode}"
        if self.unread_count() == 0:
            return False, "no_unread_messages"
        return True, "ok"

    def run_pass(self, policy: Any, *, caller_agent_id: Optional[str] = None) -> TriageOutcome:
        allowed, reason = self.gate()
        if not allowed:
            self._skips[reason] = self._skips.get(reason, 0) + 1
            if reason.startswith("control_mode:"):
                self._emit_skip(reason)
            return TriageOutcome(ran=False, reason=reason)

        # One LLM call per pass, charged to the shared action budget before it
        # happens — an unaffordable pass must not spend a token first.
        decision = self._budget.try_consume_action(1)
        if not decision.allowed:
            self._skips["budget"] = self._skips.get("budget", 0) + 1
            self._emit_skip(f"budget:{decision.reason}")
            return TriageOutcome(ran=False, reason=f"budget:{decision.reason}")

        unread = self.unread_count()
        try:
            runner = self._runner_factory()
            result = runner.run(policy, caller_agent_id=caller_agent_id)
        except Exception as exc:
            return self._on_failure(exc, unread_count=unread)

        self._consecutive_failures = 0
        actions = list(getattr(result, "actions_executed", []) or [])
        executed = [
            action for action in actions
            if isinstance(action, dict) and action.get("status") == "executed"
        ]
        if executed:
            # Every action triage actually took counts against the same ceiling
            # the supervisor's planner draws on.
            self._budget.try_consume_action(len(executed))
        return TriageOutcome(
            ran=True,
            reason="ok",
            actions_executed=len(executed),
            summary=str(getattr(result, "summary", "") or ""),
        )

    # ------------------------------------------------------------------

    def _on_failure(self, exc: Exception, *, unread_count: int) -> TriageOutcome:
        self._consecutive_failures += 1
        detail = f"{type(exc).__name__}: {exc}"
        if self._error_sink is not None:
            try:
                self._error_sink(
                    "inbox_triage",
                    detail,
                    details={
                        "unread_count": unread_count,
                        "consecutive_failures": self._consecutive_failures,
                    },
                )
            except Exception:
                pass
        try:
            self._event_log.emit(
                event_type="inbox_triage_failed",
                actor_id=self._scoped_agents.root_agent_id,
                actor_type="mr1",
                target_id=self._scoped_agents.root_agent_id,
                target_type="agent",
                status="error",
                summary="governed inbox triage failed",
                metadata={
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                    "unread_count": unread_count,
                    "consecutive_failures": self._consecutive_failures,
                },
            )
        except Exception:
            pass

        if (
            self._escalate is not None
            and self._consecutive_failures >= _ESCALATE_AFTER_CONSECUTIVE_FAILURES
        ):
            try:
                self._escalate(
                    detail,
                    f"inbox triage has failed {self._consecutive_failures} times in a row",
                )
            except Exception:
                pass
        return TriageOutcome(ran=False, reason="failed", error=detail)

    def _emit_skip(self, reason: str) -> None:
        """
        Emitted once per distinct reason, not once per skipped pass.

        A paused MR1 ticking every 30s would otherwise write 120 identical
        events an hour — the same unbounded-growth trap the supervisor avoids.
        """
        if self._skips.get(reason, 0) != 1:
            return
        try:
            self._event_log.emit(
                event_type="inbox_triage_skipped",
                actor_id=self._scoped_agents.root_agent_id,
                actor_type="mr1",
                target_id=self._scoped_agents.root_agent_id,
                target_type="agent",
                status="skipped",
                summary=f"inbox triage skipped: {reason}",
                metadata={"reason": reason},
            )
        except Exception:
            pass
