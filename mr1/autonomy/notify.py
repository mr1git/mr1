"""
Notification transport (B6).

An escalation already reaches Marwan: it lands in the `MessageStore` inbox and
on the timeline. Both are local, durable, and authoritative — and both require
him to be at the terminal to see them. MR1 running unattended for a day needs a
way to reach a human who is not looking at it.

This module is the **seam**, not the transport. It defines what a notification
is and how a sink is asked to deliver one, and it ships two sinks that depend on
nothing: a local no-op, and a structured JSONL file that any external adapter
(a cron job, a systemd unit, a five-line script that shells out to a phone) can
tail. Gmail, Slack, and anything else with an API key are deliberately *not*
here. The point of a seam is that the runtime does not know what is on the other
side of it.

Three invariants, in priority order.

**A notification is never the delivery of record.** The inbox message and the
timeline event are written first and are not conditional on any sink. If every
sink fails, the escalation still happened, is still on disk, and still parks the
objective. A notification transport that can lose an escalation is worse than no
transport at all, because it invites you to trust it.

**Retries are bounded.** A sink gets `max_attempts` tries and then it has failed.
No exponential ladder, no queue that grows while the network is down.

**Delivery is idempotent per escalation, per sink.** The ledger records what has
already gone out, so a supervisor that restarts mid-escalation, or re-raises the
same condition, does not send the same alert twice.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from mr1.clock import Clock, default_clock


NOTIFY_DIR_NAME = "notifications"
LEDGER_NAME = "delivered.json"
DEFAULT_FEED_NAME = "notifications.jsonl"

DEFAULT_MAX_ATTEMPTS = 3

URGENCY_ACTION_REQUIRED = "action_required"
URGENCY_INFO = "info"


@dataclass(frozen=True)
class Notification:
    """
    What a sink is given. Deliberately flat and JSON-safe: an adapter on the
    other side of a file should not need to import MR1 to read it.
    """

    escalation_id: str
    objective_id: str
    reason: str
    summary: str
    body: str
    at: str
    urgency: str = URGENCY_ACTION_REQUIRED
    workflow_id: Optional[str] = None
    message_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalation_id": self.escalation_id,
            "objective_id": self.objective_id,
            "reason": self.reason,
            "summary": self.summary,
            "body": self.body,
            "at": self.at,
            "urgency": self.urgency,
            "workflow_id": self.workflow_id,
            "message_id": self.message_id,
        }


class NotificationSink(Protocol):
    """A place a notification can go. Raise to signal failure; return to signal success."""

    name: str

    def deliver(self, notification: Notification) -> None: ...


@dataclass
class LocalOnlySink:
    """
    The default, and the honest one.

    The inbox and the timeline have already recorded the escalation, so there is
    genuinely nothing more to do. This exists so that "no external transport
    configured" is an explicit, named choice rather than a silently empty list.
    """

    name: str = "local"

    def deliver(self, notification: Notification) -> None:
        return None


@dataclass
class FileSink:
    """
    Append-only JSONL, one notification per line.

    This is the seam an external adapter attaches to: `tail -f` it, or poll it
    from a cron job, and forward however you like. MR1 stays ignorant of the
    transport, which is exactly what keeps a Slack outage from being an MR1
    outage.
    """

    path: Path
    name: str = "file"

    def deliver(self, notification: Notification) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(notification.to_dict(), sort_keys=True) + "\n"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


@dataclass
class StdoutSink:
    """Structured JSON on stdout — for a supervisor run under systemd or a pipe."""

    name: str = "stdout"
    stream: Any = None

    def deliver(self, notification: Notification) -> None:
        stream = self.stream or sys.stdout
        stream.write(json.dumps(notification.to_dict(), sort_keys=True) + "\n")
        stream.flush()


@dataclass
class CallableSink:
    """Wrap any function. The seam future transports plug into without touching MR1."""

    fn: Callable[[Notification], None]
    name: str = "callable"

    def deliver(self, notification: Notification) -> None:
        self.fn(notification)


@dataclass
class DeliveryResult:
    sink: str
    delivered: bool
    attempts: int
    error: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink": self.sink,
            "delivered": self.delivered,
            "attempts": self.attempts,
            "error": self.error,
            "skipped": self.skipped,
        }


@dataclass
class NotificationResult:
    escalation_id: str
    results: list[DeliveryResult] = field(default_factory=list)

    @property
    def delivered(self) -> bool:
        return any(item.delivered for item in self.results)

    @property
    def failed(self) -> list[DeliveryResult]:
        return [item for item in self.results if not item.delivered and not item.skipped]

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalation_id": self.escalation_id,
            "delivered": self.delivered,
            "results": [item.to_dict() for item in self.results],
        }


class Notifier:
    """
    Fans one escalation out to its sinks, once, with a bounded number of tries.

    `notify()` never raises. A transport failure is an operational event, not a
    control-flow one: the escalation it describes is already durable, and the
    caller must not be given the chance to treat "the alert did not send" as
    "the escalation did not happen".
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        sinks: Optional[list[NotificationSink]] = None,
        clock: Optional[Clock] = None,
        event_log: Optional[Any] = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        self._runtime_root = Path(runtime_root)
        self._clock = clock or default_clock()
        self._sinks = list(sinks) if sinks is not None else [LocalOnlySink()]
        self._max_attempts = max(1, int(max_attempts))
        # "Failures are observable" must not be contingent on a caller having
        # remembered to pass a log. Pass `event_log=False` to genuinely opt out.
        if event_log is None:
            from mr1.event_log import EventLog

            event_log = EventLog(self._runtime_root / "events")
        self._event_log = event_log or None

    @property
    def sinks(self) -> list[NotificationSink]:
        return list(self._sinks)

    @property
    def ledger_path(self) -> Path:
        return self._runtime_root / NOTIFY_DIR_NAME / LEDGER_NAME

    def notify(self, notification: Notification) -> NotificationResult:
        result = NotificationResult(escalation_id=notification.escalation_id)
        ledger = self._read_ledger()
        already = dict(ledger.get(notification.escalation_id) or {})

        for sink in self._sinks:
            name = getattr(sink, "name", type(sink).__name__)

            # Idempotent per (escalation, sink): a restarted supervisor or a
            # re-raised condition must not alert twice for the same thing.
            if already.get(name, {}).get("delivered"):
                result.results.append(DeliveryResult(
                    sink=name,
                    delivered=True,
                    attempts=0,
                    skipped=True,
                ))
                continue

            delivery = self._deliver_with_retries(sink, name, notification)
            result.results.append(delivery)
            already[name] = {
                "delivered": delivery.delivered,
                "attempts": delivery.attempts,
                "error": delivery.error,
                "at": self._clock.now_iso(),
            }

        ledger[notification.escalation_id] = already
        self._write_ledger(ledger)
        self._emit(notification, result)
        return result

    def _deliver_with_retries(
        self,
        sink: NotificationSink,
        name: str,
        notification: Notification,
    ) -> DeliveryResult:
        last_error = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                sink.deliver(notification)
                return DeliveryResult(sink=name, delivered=True, attempts=attempt)
            except Exception as exc:  # noqa: BLE001 - a sink may fail any way it likes
                last_error = f"{type(exc).__name__}: {exc}"
        # Bounded: that is all the tries there are. The escalation is already
        # safe in the inbox and on the timeline; this is a lost *alert*, and it
        # is recorded as one.
        return DeliveryResult(
            sink=name,
            delivered=False,
            attempts=self._max_attempts,
            error=last_error,
        )

    # -- the ledger ----------------------------------------------------

    def _read_ledger(self) -> dict[str, Any]:
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_ledger(self, ledger: dict[str, Any]) -> None:
        try:
            path = self.ledger_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(ledger, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
        except OSError:
            # Losing the ledger costs us idempotence, not the escalation. Worst
            # case a future notification is sent twice, which beats not sending.
            pass

    def _emit(self, notification: Notification, result: NotificationResult) -> None:
        if self._event_log is None:
            return
        failed = result.failed
        try:
            self._event_log.emit(
                event_type="notification_failed" if failed else "notification_delivered",
                actor_id="notifier",
                actor_type="mr1",
                target_id=notification.objective_id,
                target_type="objective",
                status="error" if failed else "ok",
                summary=(
                    f"notification failed on {', '.join(item.sink for item in failed)}: "
                    f"{notification.summary}"
                    if failed else
                    f"notification delivered: {notification.summary}"
                )[:300],
                metadata={
                    "escalation_id": notification.escalation_id,
                    "reason": notification.reason,
                    "sinks": [item.to_dict() for item in result.results],
                },
            )
        except Exception:  # noqa: BLE001 - observability must not break delivery
            pass


def build_sinks(specs: list[str], runtime_root: Path) -> list[NotificationSink]:
    """
    Parse operator sink specs into sinks.

        local                 the inbox and timeline are the delivery (default)
        stdout                structured JSON on stdout
        file                  <runtime_root>/notifications/notifications.jsonl
        file:/path/to.jsonl   an explicit path an external adapter tails
    """
    sinks: list[NotificationSink] = []
    for raw in specs or []:
        spec = str(raw).strip()
        if not spec or spec == "local":
            sinks.append(LocalOnlySink())
        elif spec == "stdout":
            sinks.append(StdoutSink())
        elif spec == "file":
            sinks.append(FileSink(path=Path(runtime_root) / NOTIFY_DIR_NAME / DEFAULT_FEED_NAME))
        elif spec.startswith("file:"):
            sinks.append(FileSink(path=Path(spec[len("file:"):]).expanduser()))
        else:
            raise ValueError(
                f"unknown notification sink '{spec}' "
                "(expected: local, stdout, file, or file:<path>)"
            )
    return sinks or [LocalOnlySink()]
