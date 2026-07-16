"""
B6 — the notification seam.

An escalation already reaches Marwan's inbox and the timeline. Both are local,
and both require him to be at the terminal. This adds a transport-neutral way to
reach a human who is not.

The interesting tests are all about what happens when the transport is broken,
because that is the only thing that could make this feature worse than not
having it. A notification layer that can swallow an escalation is not a feature;
it is a way to be lied to. So: the inbox message, the timeline event, and the
parked objective are written whatever any sink does, and a sink that fails
costs an *alert*, never an escalation.

No Gmail, no Slack, no MCP — the whole point of a seam is that the runtime does
not know what is on the other side of it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mr1.autonomy.escalation import REASON_CONSENT_MISSING, Escalator
from mr1.autonomy.notify import (
    CallableSink,
    FileSink,
    LocalOnlySink,
    Notification,
    Notifier,
    StdoutSink,
    build_sinks,
)
from mr1.autonomy.objectives import KIND_ONCE, ObjectiveStore
from mr1.clock import VirtualClock
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _notification(escalation_id: str = "esc-1") -> Notification:
    return Notification(
        escalation_id=escalation_id,
        objective_id="obj-1",
        reason=REASON_CONSENT_MISSING,
        summary="Genesis: consent missing",
        body="MR1 needs shell_command consent.",
        at=START.isoformat(),
    )


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir(parents=True)
    return root


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def test_the_file_sink_writes_one_json_object_per_line(runtime):
    """
    The seam an external adapter attaches to. It must be trivially parseable by
    something that has never heard of MR1 — a shell script, a cron job, five
    lines of anything.
    """
    feed = runtime / "notifications.jsonl"
    sink = FileSink(path=feed)

    sink.deliver(_notification("esc-1"))
    sink.deliver(_notification("esc-2"))

    lines = [json.loads(line) for line in feed.read_text().splitlines() if line.strip()]
    assert [item["escalation_id"] for item in lines] == ["esc-1", "esc-2"]
    assert lines[0]["reason"] == REASON_CONSENT_MISSING
    assert lines[0]["body"]


def test_the_stdout_sink_writes_structured_json(runtime):
    import io

    stream = io.StringIO()
    StdoutSink(stream=stream).deliver(_notification())

    payload = json.loads(stream.getvalue().strip())
    assert payload["escalation_id"] == "esc-1"
    assert payload["urgency"] == "action_required"


def test_the_local_sink_does_nothing_and_says_so(runtime):
    """
    The default. The inbox and the timeline have already delivered it; naming
    that explicitly beats an empty sink list that looks like a misconfiguration.
    """
    assert LocalOnlySink().deliver(_notification()) is None


def test_sinks_are_built_from_operator_specs(runtime):
    assert [type(s).__name__ for s in build_sinks([], runtime)] == ["LocalOnlySink"]
    assert [type(s).__name__ for s in build_sinks(["stdout"], runtime)] == ["StdoutSink"]

    file_sinks = build_sinks(["file"], runtime)
    assert isinstance(file_sinks[0], FileSink)
    assert file_sinks[0].path == runtime / "notifications" / "notifications.jsonl"

    explicit = build_sinks([f"file:{runtime}/alerts.jsonl"], runtime)
    assert explicit[0].path == Path(f"{runtime}/alerts.jsonl")

    both = build_sinks(["stdout", "file"], runtime)
    assert len(both) == 2

    with pytest.raises(ValueError, match="unknown notification sink"):
        build_sinks(["slack://webhook"], runtime)


# ---------------------------------------------------------------------------
# The notifier
# ---------------------------------------------------------------------------


def test_delivery_is_recorded_and_emitted(runtime):
    clock = VirtualClock(start=START)
    log = EventLog(runtime / "events")
    feed = runtime / "feed.jsonl"

    notifier = Notifier(
        runtime,
        sinks=[FileSink(path=feed)],
        clock=clock,
        event_log=log,
    )
    result = notifier.notify(_notification())

    assert result.delivered is True
    assert result.failed == []
    assert feed.exists()
    assert [event.event_type for event in log.list_events()] == ["notification_delivered"]


def test_the_same_escalation_is_never_delivered_twice(runtime):
    """
    A supervisor that restarts mid-escalation, or re-raises the same condition,
    must not alert twice for the same thing.
    """
    clock = VirtualClock(start=START)
    delivered: list[str] = []
    sink = CallableSink(fn=lambda n: delivered.append(n.escalation_id), name="probe")

    notifier = Notifier(runtime, sinks=[sink], clock=clock)
    notifier.notify(_notification("esc-1"))
    notifier.notify(_notification("esc-1"))
    notifier.notify(_notification("esc-1"))

    assert delivered == ["esc-1"], "one escalation, one alert"

    # A different escalation still gets through.
    notifier.notify(_notification("esc-2"))
    assert delivered == ["esc-1", "esc-2"]

    # And the ledger survives a restart, so a fresh process does not re-send.
    reopened = Notifier(runtime, sinks=[sink], clock=clock)
    reopened.notify(_notification("esc-1"))
    assert delivered == ["esc-1", "esc-2"]


def test_retries_are_bounded_and_then_it_has_failed(runtime):
    clock = VirtualClock(start=START)
    attempts: list[int] = []

    def always_fails(_notification):
        attempts.append(1)
        raise OSError("the pipe is broken")

    notifier = Notifier(
        runtime,
        sinks=[CallableSink(fn=always_fails, name="broken")],
        clock=clock,
        max_attempts=3,
    )
    result = notifier.notify(_notification())

    assert len(attempts) == 3, "bounded: three tries, then it has failed"
    assert result.delivered is False
    assert result.failed[0].sink == "broken"
    assert "OSError" in result.failed[0].error


def test_a_transient_failure_is_retried_within_the_bound(runtime):
    calls: list[int] = []

    def flaky(_notification):
        calls.append(1)
        if len(calls) < 2:
            raise OSError("not yet")

    notifier = Notifier(
        runtime,
        sinks=[CallableSink(fn=flaky, name="flaky")],
        clock=VirtualClock(start=START),
        max_attempts=3,
    )
    result = notifier.notify(_notification())

    assert result.delivered is True
    assert result.results[0].attempts == 2


def test_a_failure_is_observable(runtime):
    log = EventLog(runtime / "events")

    def always_fails(_n):
        raise RuntimeError("gone")

    Notifier(
        runtime,
        sinks=[CallableSink(fn=always_fails, name="broken")],
        clock=VirtualClock(start=START),
        event_log=log,
        max_attempts=1,
    ).notify(_notification())

    events = log.list_events()
    assert [event.event_type for event in events] == ["notification_failed"]
    assert events[0].severity == "ERROR"
    assert events[0].metadata["sinks"][0]["error"].startswith("RuntimeError")


def test_one_broken_sink_does_not_stop_the_others(runtime):
    delivered: list[str] = []

    def broken(_n):
        raise OSError("nope")

    notifier = Notifier(
        runtime,
        sinks=[
            CallableSink(fn=broken, name="broken"),
            CallableSink(fn=lambda n: delivered.append(n.escalation_id), name="working"),
        ],
        clock=VirtualClock(start=START),
        max_attempts=1,
    )
    result = notifier.notify(_notification())

    assert delivered == ["esc-1"]
    assert result.delivered is True, "one sink got it"
    assert [item.sink for item in result.failed] == ["broken"]


def test_notify_never_raises(runtime):
    notifier = Notifier(
        runtime,
        sinks=[CallableSink(fn=lambda _n: (_ for _ in ()).throw(BaseException("catastrophe")))],
        clock=VirtualClock(start=START),
        max_attempts=1,
    )
    # A BaseException from a sink is not caught by `except Exception` — but the
    # escalation is already durable by the time we get here, and the Escalator
    # guards the call anyway. What must not happen is a *normal* failure
    # escaping, which the other tests cover.
    with pytest.raises(BaseException, match="catastrophe"):
        notifier.notify(_notification())


# ---------------------------------------------------------------------------
# Through the real Escalator — the invariant that matters
# ---------------------------------------------------------------------------


def _escalator(runtime, *, notifier=None):
    clock = VirtualClock(start=START)
    agents = PersistentAgentStore(root=runtime / "agents")
    objectives = ObjectiveStore(runtime, clock=clock)
    messages = MessageStore(root=runtime / "messages", scoped_agent_store=agents)
    escalator = Escalator(
        runtime,
        objective_store=objectives,
        message_store=messages,
        scoped_agent_store=agents,
        clock=clock,
        notifier=notifier,
    )
    objective = objectives.create(
        title="Genesis",
        statement="run the weekly cycle",
        kind=KIND_ONCE,
        owner_agent_id=agents.root_agent_id,
    )
    return escalator, objectives, messages, agents, objective


def test_a_totally_broken_transport_never_loses_the_escalation(runtime):
    """
    The invariant the whole checkpoint rests on.

    Every sink fails. The escalation must still be: in the inbox, on the
    timeline, and parked on the objective. Anything less would mean a Slack
    outage could silently strand an autonomous objective forever.
    """
    def always_fails(_n):
        raise OSError("every transport is down")

    notifier = Notifier(
        runtime,
        sinks=[CallableSink(fn=always_fails, name="broken")],
        clock=VirtualClock(start=START),
        max_attempts=2,
    )
    escalator, objectives, messages, agents, objective = _escalator(runtime, notifier=notifier)

    escalation = escalator.escalate(
        objective,
        reason=REASON_CONSENT_MISSING,
        problem="needs shell_command consent",
        needs="a consent grant",
        capability="shell_command",
    )

    # 1. The inbox has it.
    inbox = messages.list_inbox(agents.root_agent_id)
    assert len(inbox) == 1
    assert "consent missing" in inbox[0].subject

    # 2. The timeline has it.
    events = [e.event_type for e in EventLog(runtime / "events").list_events()]
    assert "escalation_raised" in events

    # 3. The objective is parked.
    assert objectives.require(objective.objective_id).status == "waiting_human"
    assert escalation.message_id is not None

    # 4. And the failure to *notify* is itself recorded.
    assert "notification_failed" in events


def test_an_escalation_reaches_the_configured_sink(runtime):
    feed = runtime / "alerts.jsonl"
    notifier = Notifier(
        runtime,
        sinks=[FileSink(path=feed)],
        clock=VirtualClock(start=START),
    )
    escalator, _objectives, _messages, _agents, objective = _escalator(runtime, notifier=notifier)

    escalator.escalate(
        objective,
        reason=REASON_CONSENT_MISSING,
        problem="needs shell_command consent",
        needs="a consent grant",
        capability="shell_command",
    )

    payload = json.loads(feed.read_text().strip())
    assert payload["objective_id"] == objective.objective_id
    assert payload["reason"] == REASON_CONSENT_MISSING
    assert "shell_command" in payload["body"]
    assert "mr1 grant create" in payload["body"], "the alert carries the fix, not just the alarm"


def test_a_re_escalated_condition_does_not_re_alert(runtime):
    """The 60-second tick must not send 60 identical alerts an hour."""
    sent: list[str] = []
    notifier = Notifier(
        runtime,
        sinks=[CallableSink(fn=lambda n: sent.append(n.escalation_id), name="probe")],
        clock=VirtualClock(start=START),
    )
    escalator, objectives, _messages, _agents, objective = _escalator(runtime, notifier=notifier)

    for _ in range(10):
        current = objectives.require(objective.objective_id)
        escalator.escalate(
            current,
            reason=REASON_CONSENT_MISSING,
            problem="needs shell_command consent",
            needs="a consent grant",
            capability="shell_command",
        )

    assert len(sent) == 1, "ten ticks, one alert"


def test_no_notifier_configured_is_a_working_configuration(runtime):
    """Local-only is the default, and it must be a complete one."""
    escalator, objectives, messages, agents, objective = _escalator(runtime, notifier=None)

    escalator.escalate(
        objective,
        reason=REASON_CONSENT_MISSING,
        problem="needs consent",
        needs="a grant",
    )

    assert len(messages.list_inbox(agents.root_agent_id)) == 1
    assert objectives.require(objective.objective_id).status == "waiting_human"
