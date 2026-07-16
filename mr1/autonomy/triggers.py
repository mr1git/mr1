"""
Recurrence: intervals, cron, and what to do about the runs you missed (B2).

Phase A gave objectives `immediate` / `interval` / `manual` / `watcher` triggers
and derived interval due-times from the last completion. That works while MR1 is
up. It says nothing about the case that actually matters for continuous
operation: **MR1 was down, and its objectives kept coming due while it was.**

A daily objective and a week of downtime is seven pending runs. A naive
scheduler fires seven workflows the moment it comes back — seven planning calls,
seven consent-authorized executions, in one tick, for work whose window has
already passed. That is the failure this module exists to prevent.

Semantics
---------
`due` is the number of scheduled occurrences that have elapsed since the trigger
last fired.

  due == 0   nothing to do.
  due == 1   normal operation. The trigger fires. No policy is involved — one
             occurrence elapsed and one run happens.
  due  > 1   MR1 fell behind: downtime, a pause, backpressure. `missed = due - 1`
             occurrences were lost, and `missed_run_policy` decides how many
             make-up runs are allowed:

               skip           0 make-up runs. Realign to the next boundary and
                              wait. "The window passed; don't pretend otherwise."
               catch_up_once  1 make-up run. The whole backlog coalesces into a
                              single run. (Default — usually what you want: run
                              the weekly cycle you missed, once.)
               bounded        up to `max_catch_up_runs` make-up runs, worked off
                              one per tick. Never more, whatever the outage.

In every case the number of workflows an outage can produce is bounded by
configuration, never by the length of the outage.

Timezone
--------
`interval` is elapsed wall-clock time and has no timezone: 24h after the last
run is 24h later, DST or not.

`cron` is a calendar and therefore must have one. `timezone` on the trigger
(IANA name, default UTC) is the zone the fields are read in — "0 9 * * 1" means
09:00 Monday *there*. Boundaries are computed in that zone and converted back to
UTC, so a cron objective keeps its local meaning across a DST shift instead of
silently sliding an hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - stdlib since 3.9
    ZoneInfo = None  # type: ignore[assignment]


POLICY_SKIP = "skip"
POLICY_CATCH_UP_ONCE = "catch_up_once"
POLICY_BOUNDED = "bounded"

MISSED_RUN_POLICIES = (POLICY_SKIP, POLICY_CATCH_UP_ONCE, POLICY_BOUNDED)

DEFAULT_MISSED_RUN_POLICY = POLICY_CATCH_UP_ONCE
DEFAULT_MAX_CATCH_UP_RUNS = 1

# A hard ceiling on how many occurrences we will even enumerate. A cron of
# "* * * * *" and a year of downtime is half a million boundaries; we do not
# need the exact number to know the backlog is "more than we will ever run".
_MAX_ENUMERATED_OCCURRENCES = 10_000


class TriggerError(ValueError):
    """The trigger spec is not something we can schedule."""


@dataclass(frozen=True)
class TriggerDecision:
    """
    Everything the supervisor needs, and nothing it has to recompute.

    `runs_allowed` is the number of workflows this objective may produce right
    now. It is 0 or 1 on any single tick — an objective runs one workflow at a
    time — but `catch_up_remaining` carries a bounded backlog forward so that
    "work off two of the runs you missed" happens across two ticks, not in one.
    """

    ready: bool
    reason: str
    next_due_at: Optional[str] = None
    due: int = 0
    missed: int = 0
    catch_up_remaining: int = 0


# ---------------------------------------------------------------------------
# Cron
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CronSpec:
    """
    A five-field cron: `minute hour day-of-month month day-of-week`.

    Supports `*`, `a`, `a,b`, `a-b`, and `*/n` (and `a-b/n`). Day-of-week is
    0-6 with 0 = Sunday; 7 is accepted and means Sunday too. When both
    day-of-month and day-of-week are restricted, either matching is enough —
    the same rule Vixie cron uses, and the one people actually expect.

    Deliberately hand-rolled and dependency-free: this runs on every tick of a
    process that must not acquire new failure modes, and the subset above covers
    every calendar recurrence an objective has needed.
    """

    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool
    expression: str

    @classmethod
    def parse(cls, expression: str) -> "CronSpec":
        fields = str(expression or "").split()
        if len(fields) != 5:
            raise TriggerError(
                f"cron expression must have 5 fields "
                f"(minute hour day-of-month month day-of-week), got: {expression!r}"
            )
        minute, hour, dom, month, dow = fields
        days_of_week = _parse_field(dow, 0, 7, "day-of-week")
        # 7 and 0 are both Sunday.
        if 7 in days_of_week:
            days_of_week = (days_of_week - {7}) | {0}
        return cls(
            minutes=_parse_field(minute, 0, 59, "minute"),
            hours=_parse_field(hour, 0, 23, "hour"),
            days_of_month=_parse_field(dom, 1, 31, "day-of-month"),
            months=_parse_field(month, 1, 12, "month"),
            days_of_week=days_of_week,
            dom_restricted=dom.strip() != "*",
            dow_restricted=dow.strip() != "*",
            expression=str(expression),
        )

    def matches(self, moment: datetime) -> bool:
        if moment.month not in self.months:
            return False
        if moment.hour not in self.hours:
            return False
        if moment.minute not in self.minutes:
            return False

        # cron weekday: 0 = Sunday. Python's weekday(): 0 = Monday.
        weekday = (moment.weekday() + 1) % 7
        dom_ok = moment.day in self.days_of_month
        dow_ok = weekday in self.days_of_week

        if self.dom_restricted and self.dow_restricted:
            return dom_ok or dow_ok
        if self.dom_restricted:
            return dom_ok
        if self.dow_restricted:
            return dow_ok
        return True

    def next_after(self, after: datetime) -> Optional[datetime]:
        """The first matching minute strictly after `after`, in `after`'s zone."""
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        # Four years covers any Feb-29 cron. Beyond that the spec matches nothing.
        limit = after + timedelta(days=366 * 4)
        while candidate <= limit:
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        return None  # pragma: no cover - only for specs that can never match


def _parse_field(raw: str, low: int, high: int, name: str) -> frozenset[int]:
    values: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            raise TriggerError(f"empty {name} field in cron expression")

        step = 1
        if "/" in part:
            part, _, step_raw = part.partition("/")
            try:
                step = int(step_raw)
            except ValueError:
                raise TriggerError(f"invalid step '{step_raw}' in cron {name}") from None
            if step < 1:
                raise TriggerError(f"cron {name} step must be >= 1")

        if part == "*" or part == "":
            start, end = low, high
        elif "-" in part:
            start_raw, _, end_raw = part.partition("-")
            try:
                start, end = int(start_raw), int(end_raw)
            except ValueError:
                raise TriggerError(f"invalid range '{part}' in cron {name}") from None
        else:
            try:
                start = end = int(part)
            except ValueError:
                raise TriggerError(f"invalid value '{part}' in cron {name}") from None

        if start < low or end > high or start > end:
            raise TriggerError(
                f"cron {name} value out of range: '{part}' (expected {low}-{high})"
            )
        values.update(range(start, end + 1, step))

    if not values:
        raise TriggerError(f"cron {name} field matches nothing")
    return frozenset(values)


def resolve_timezone(name: Optional[str]) -> timezone | Any:
    if not name or name.upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:  # pragma: no cover - stdlib since 3.9
        raise TriggerError("zoneinfo is unavailable; only UTC is supported")
    try:
        return ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001 - ZoneInfoNotFoundError and friends
        raise TriggerError(f"unknown timezone '{name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Recurrence
# ---------------------------------------------------------------------------


def missed_run_policy(trigger: dict[str, Any]) -> tuple[str, int]:
    policy = str(trigger.get("missed_run_policy") or DEFAULT_MISSED_RUN_POLICY)
    if policy not in MISSED_RUN_POLICIES:
        raise TriggerError(
            f"unknown missed_run_policy '{policy}' "
            f"(expected one of {', '.join(MISSED_RUN_POLICIES)})"
        )
    if policy == POLICY_SKIP:
        return policy, 0
    if policy == POLICY_CATCH_UP_ONCE:
        return policy, 1
    raw = trigger.get("max_catch_up_runs", DEFAULT_MAX_CATCH_UP_RUNS)
    try:
        allowance = int(raw)
    except (TypeError, ValueError):
        raise TriggerError(f"max_catch_up_runs must be an integer, got {raw!r}") from None
    if allowance < 0:
        raise TriggerError("max_catch_up_runs must be >= 0")
    return policy, allowance


def occurrences_due(
    trigger: dict[str, Any],
    *,
    anchor: Optional[datetime],
    now: datetime,
) -> tuple[int, Optional[datetime]]:
    """
    How many scheduled occurrences have elapsed since `anchor`, and when the
    next one falls.

    `anchor` is the last time the trigger fired (or None if it never has).
    Counting is capped: we never need the exact size of an unbounded backlog,
    only that there is one.
    """
    kind = str(trigger.get("type") or "")

    if kind == "interval":
        interval_s = float(trigger.get("interval_s") or 0)
        if interval_s <= 0:
            raise TriggerError("interval trigger requires a positive interval_s")
        if anchor is None:
            return 1, now
        elapsed = (now - anchor).total_seconds()
        if elapsed < interval_s:
            return 0, anchor + timedelta(seconds=interval_s)
        due = min(int(elapsed // interval_s), _MAX_ENUMERATED_OCCURRENCES)
        next_due = anchor + timedelta(seconds=interval_s * (due + 1))
        return due, next_due

    if kind == "cron":
        spec = CronSpec.parse(trigger.get("expression") or trigger.get("cron") or "")
        tz = resolve_timezone(trigger.get("timezone"))
        local_now = now.astimezone(tz)
        if anchor is None:
            # Never fired: the objective is due at its next boundary, not now.
            # A cron objective means "at 9am Monday", not "right away, and then
            # at 9am Monday".
            upcoming = spec.next_after(local_now)
            return 0, upcoming.astimezone(timezone.utc) if upcoming else None

        local_anchor = anchor.astimezone(tz)
        due = 0
        cursor = spec.next_after(local_anchor)
        while cursor is not None and cursor <= local_now:
            due += 1
            if due >= _MAX_ENUMERATED_OCCURRENCES:
                break
            cursor = spec.next_after(cursor)
        next_due = cursor.astimezone(timezone.utc) if cursor is not None else None
        return due, next_due

    raise TriggerError(f"'{kind}' is not a recurring trigger")


def evaluate_recurrence(
    trigger: dict[str, Any],
    *,
    anchor: Optional[datetime],
    now: datetime,
    catch_up_remaining: int = 0,
) -> TriggerDecision:
    """
    Should a recurring trigger fire now, and what backlog carries forward?

    Pure — no I/O, no clock of its own. The supervisor persists what it returns.
    """
    policy, allowance = missed_run_policy(trigger)

    # A make-up run already owed from a previous tick fires regardless of where
    # the next boundary is; it is the backlog being worked off, not a new one.
    if catch_up_remaining > 0:
        due, next_due = occurrences_due(trigger, anchor=anchor, now=now)
        return TriggerDecision(
            ready=True,
            reason=f"catch-up run ({catch_up_remaining} remaining under '{policy}')",
            next_due_at=_iso(next_due),
            due=due,
            missed=0,
            catch_up_remaining=catch_up_remaining - 1,
        )

    due, next_due = occurrences_due(trigger, anchor=anchor, now=now)

    if due <= 0:
        return TriggerDecision(
            ready=False,
            reason=(
                f"next run at {_iso(next_due)}"
                if next_due else
                "no future occurrence matches this trigger"
            ),
            next_due_at=_iso(next_due),
            due=0,
        )

    if due == 1:
        # On time. The policy has nothing to say about a run that was not missed.
        return TriggerDecision(
            ready=True,
            reason="never run before" if anchor is None else "due",
            next_due_at=_iso(next_due),
            due=1,
            missed=0,
            catch_up_remaining=0,
        )

    # Behind. `missed` occurrences elapsed while MR1 was not running them.
    missed = due - 1

    if policy == POLICY_SKIP:
        return TriggerDecision(
            ready=False,
            reason=(
                f"{missed} run(s) missed and skipped per policy 'skip'; "
                f"next run at {_iso(next_due)}"
            ),
            next_due_at=_iso(next_due),
            due=due,
            missed=missed,
            catch_up_remaining=0,
        )

    # catch_up_once / bounded: fire now, and owe at most `allowance - 1` more.
    # The backlog can never exceed the allowance, however long the outage was.
    owed = max(0, min(missed, allowance) - 1)
    return TriggerDecision(
        ready=True,
        reason=(
            f"{missed} run(s) missed; catching up "
            f"{min(missed, allowance)} of them per policy '{policy}'"
        ),
        next_due_at=_iso(next_due),
        due=due,
        missed=missed,
        catch_up_remaining=owed,
    )


def validate_trigger(trigger: dict[str, Any]) -> dict[str, Any]:
    """Reject an unschedulable trigger at creation, not on the first tick."""
    if not isinstance(trigger, dict) or not trigger.get("type"):
        raise TriggerError("a trigger must be a dict with a 'type'")
    kind = str(trigger["type"])

    if kind in {"interval", "cron"}:
        missed_run_policy(trigger)
        occurrences_due(trigger, anchor=None, now=datetime.now(timezone.utc))
    elif kind not in {"immediate", "manual", "watcher"}:
        raise TriggerError(f"unknown trigger type '{kind}'")
    return trigger


def _iso(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).isoformat()
