"""A9 — the already-unattended triage loop, brought under the control plane."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mr1.autonomy.budget import BudgetLedger, BudgetLimits
from mr1.autonomy.control import (
    MODE_HALTED,
    MODE_PAUSED,
    MODE_RUNNING,
    MODE_STOPPING,
    ControlPlane,
)
from mr1.autonomy.triage import GovernedTriage
from mr1.clock import VirtualClock
from mr1.event_log import EventLog
from mr1.inbox_triage import InboxTriagePolicy, InboxTriageResult
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore


class FakeTriageRunner:
    def __init__(self, *, actions=0, raises=None):
        self.calls = 0
        self._actions = actions
        self._raises = raises

    def run(self, policy, caller_agent_id=None):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return InboxTriageResult(
            processed_messages=["msg-1"],
            actions_executed=[
                {"type": "mark_read", "reason": "read it", "status": "executed"}
                for _ in range(self._actions)
            ],
            summary="triaged",
            counts={},
        )


class Fixture:
    def __init__(self, tmp_path, *, runner=None, limits=None):
        self.clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.runtime_root = tmp_path / "runtime"
        self.agents = PersistentAgentStore(root=self.runtime_root / "agents")
        self.messages = MessageStore(
            root=self.runtime_root / "messages",
            scoped_agent_store=self.agents,
        )
        self.control = ControlPlane(self.runtime_root, clock=self.clock)
        self.budget = BudgetLedger(
            self.runtime_root,
            clock=self.clock,
            limits=limits or BudgetLimits(),
        )
        self.runner = runner or FakeTriageRunner()
        self.errors: list[tuple] = []
        self.escalations: list[tuple] = []
        self.triage = GovernedTriage(
            self.runtime_root,
            message_store=self.messages,
            scoped_agent_store=self.agents,
            runner_factory=lambda: self.runner,
            control=self.control,
            budget=self.budget,
            clock=self.clock,
            error_sink=lambda source, error, details=None: self.errors.append((source, error, details)),
            escalate=lambda detail, problem: self.escalations.append((detail, problem)),
        )

    def send_unread(self, subject="hello"):
        root_id = self.agents.root_agent_id
        return self.messages.create_message(
            from_agent_id=root_id,
            to_agent_id=root_id,
            kind="question",
            subject=subject,
            body="please look at this",
        )

    def run(self):
        return self.triage.run_pass(InboxTriagePolicy(max_messages=10, max_actions=5))

    def events(self, **kwargs):
        return EventLog(self.runtime_root / "events").filter_events(**kwargs)


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


# -- the zero-cost property -------------------------------------------------


def test_an_empty_inbox_makes_no_llm_call(fx):
    outcome = fx.run()

    assert outcome.ran is False
    assert outcome.reason == "no_unread_messages"
    assert fx.runner.calls == 0


def test_unread_mail_triggers_exactly_one_pass(fx):
    fx.send_unread()

    outcome = fx.run()

    assert outcome.ran is True
    assert fx.runner.calls == 1


# -- the control plane ------------------------------------------------------


def test_pause_prevents_new_triage_actions(fx):
    fx.send_unread()
    fx.control.set_mode(MODE_PAUSED, reason="test")

    outcome = fx.run()

    assert outcome.ran is False
    assert outcome.reason == "control_mode:paused"
    assert fx.runner.calls == 0, "paused must mean paused"


def test_stopping_prevents_new_triage_actions(fx):
    fx.send_unread()
    fx.control.set_mode(MODE_STOPPING, reason="test")

    outcome = fx.run()

    assert outcome.ran is False
    assert outcome.reason == "control_mode:stopping"
    assert fx.runner.calls == 0


def test_halt_stops_triage_immediately(fx):
    fx.send_unread()
    fx.control.set_mode(MODE_HALTED, reason="emergency")

    outcome = fx.run()

    assert outcome.ran is False
    assert outcome.reason == "control_mode:halted"
    assert fx.runner.calls == 0


def test_resume_lets_triage_act_again(fx):
    fx.send_unread()
    fx.control.set_mode(MODE_PAUSED, reason="test")
    fx.run()

    fx.control.set_mode(MODE_RUNNING, reason="test")
    outcome = fx.run()

    assert outcome.ran is True
    assert fx.runner.calls == 1


def test_a_corrupt_control_file_stops_triage(fx):
    """Fail closed: an unreadable control plane is not permission to act."""
    fx.send_unread()
    fx.control.path.write_text("{ broken", encoding="utf-8")

    outcome = fx.run()

    assert outcome.ran is False
    assert fx.runner.calls == 0


def test_a_skipped_pass_is_emitted_once_not_every_tick(fx):
    fx.send_unread()
    fx.control.set_mode(MODE_PAUSED, reason="test")

    for _ in range(20):
        fx.run()

    events = fx.events(event_type="inbox_triage_skipped")
    assert len(events) == 1
    assert fx.triage.skips["control_mode:paused"] == 20


# -- the shared budget ------------------------------------------------------


def test_a_triage_pass_spends_from_the_shared_budget(fx):
    fx.send_unread()

    fx.run()

    # One action for the LLM call itself, before it happens.
    assert fx.budget.actions_this_hour() == 1


def test_executed_actions_count_against_the_same_ceiling(tmp_path):
    fx = Fixture(tmp_path, runner=FakeTriageRunner(actions=3))
    fx.send_unread()

    fx.run()

    # 1 for the pass + 3 for the actions it took.
    assert fx.budget.actions_this_hour() == 4


def test_an_exhausted_action_budget_stops_triage(tmp_path):
    fx = Fixture(tmp_path, limits=BudgetLimits(max_actions_per_hour=1))
    fx.send_unread()

    assert fx.run().ran is True          # spends the one action available
    outcome = fx.run()                   # nothing left to spend

    assert outcome.ran is False
    assert outcome.reason.startswith("budget:")
    assert fx.runner.calls == 1


def test_the_budget_window_rolls_forward(tmp_path):
    fx = Fixture(tmp_path, limits=BudgetLimits(max_actions_per_hour=1))
    fx.send_unread()
    fx.run()
    assert fx.run().ran is False

    fx.clock.advance(3601)

    assert fx.run().ran is True


def test_the_supervisor_and_triage_draw_on_one_budget(tmp_path):
    """Two unattended surfaces, one ceiling."""
    fx = Fixture(tmp_path, limits=BudgetLimits(max_actions_per_hour=3))
    fx.send_unread()

    # The supervisor spends two actions from the same ledger...
    other = BudgetLedger(
        fx.runtime_root,
        clock=fx.clock,
        limits=BudgetLimits(max_actions_per_hour=3),
    )
    other.try_consume_action(3)

    # ...so triage finds none left.
    outcome = fx.run()

    assert outcome.ran is False
    assert outcome.reason.startswith("budget:")


# -- failures ---------------------------------------------------------------


def test_a_triage_failure_is_recorded_and_emitted_not_swallowed(tmp_path):
    fx = Fixture(tmp_path, runner=FakeTriageRunner(raises=ValueError("triage exploded")))
    fx.send_unread()

    outcome = fx.run()

    assert outcome.ran is False
    assert "triage exploded" in outcome.error
    assert fx.errors[0][0] == "inbox_triage"
    events = fx.events(event_type="inbox_triage_failed")
    assert len(events) == 1
    assert events[0].metadata["error_type"] == "ValueError"


def test_repeated_triage_failures_escalate(tmp_path):
    fx = Fixture(tmp_path, runner=FakeTriageRunner(raises=ValueError("still broken")))
    fx.send_unread()

    for _ in range(3):
        fx.run()

    assert fx.triage.consecutive_failures == 3
    assert fx.escalations, "repeated triage failure must reach a human"


def test_a_successful_pass_resets_the_failure_counter(tmp_path):
    runner = FakeTriageRunner(raises=ValueError("boom"))
    fx = Fixture(tmp_path, runner=runner)
    fx.send_unread()
    fx.run()
    assert fx.triage.consecutive_failures == 1

    fx.runner = FakeTriageRunner()
    fx.triage._runner_factory = lambda: fx.runner
    fx.run()

    assert fx.triage.consecutive_failures == 0


# -- MR1 does not get a vote on its own escalations -------------------------


def test_triage_ignores_mr1s_own_escalations_to_the_human(fx):
    """
    An escalation is addressed to Marwan and lands in root's own inbox. Triage
    may archive and mark-read; letting it act here would let MR1 quietly dispose
    of the message asking for the consent it is blocked on.
    """
    root_id = fx.agents.root_agent_id
    fx.messages.create_message(
        from_agent_id=root_id,
        to_agent_id=root_id,
        kind="alert",
        subject="[MR1] Genesis: consent missing",
        body="MR1 needs you.",
    )

    outcome = fx.run()

    assert outcome.ran is False
    assert outcome.reason == "no_unread_messages"
    assert fx.runner.calls == 0


def test_real_mail_is_still_triaged_alongside_an_escalation(fx):
    root_id = fx.agents.root_agent_id
    fx.messages.create_message(
        from_agent_id=root_id,
        to_agent_id=root_id,
        kind="alert",
        subject="[MR1] escalation",
        body="MR1 needs you.",
    )
    fx.send_unread(subject="a real question from a child agent")

    outcome = fx.run()

    assert outcome.ran is True
    assert fx.runner.calls == 1
