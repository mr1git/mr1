"""CLI handlers for objectives (`mr1 objective ...`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mr1.autonomy.objectives import (
    KIND_ONCE,
    STATUS_ACTIVE,
    STATUS_ABANDONED,
    STATUS_PAUSED,
    ObjectiveError,
    ObjectiveStore,
)
from mr1.autonomy.recovery import FailurePolicy
from mr1.cli.consent import parse_duration
from mr1.clock import default_clock


def _store(store) -> ObjectiveStore:
    return ObjectiveStore(Path(store.root).parent, clock=default_clock())


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _cmd_objective_create(args, store, caller_agent_id, scoped_agent_store) -> int:
    objectives = _store(store)

    trigger: dict[str, Any]
    if getattr(args, "trigger", None):
        try:
            trigger = json.loads(args.trigger)
        except json.JSONDecodeError as exc:
            print(f"error: --trigger must be JSON: {exc}")
            return 1
        if not isinstance(trigger, dict) or not trigger.get("type"):
            print("error: --trigger must be a JSON object with a 'type'")
            return 1
    elif getattr(args, "every", None):
        try:
            trigger = {"type": "interval", "interval_s": parse_duration(args.every)}
        except ValueError as exc:
            print(f"error: {exc}")
            return 1
    elif args.kind == KIND_ONCE:
        trigger = {"type": "immediate"}
    else:
        trigger = {"type": "interval", "interval_s": 7 * 86_400.0}

    policy_overrides = {}
    for name in ("max_retries", "max_replans", "max_consecutive_failures"):
        value = getattr(args, name, None)
        if value is not None:
            policy_overrides[name] = value

    try:
        objective = objectives.create(
            title=args.title or args.statement[:60],
            statement=args.statement,
            owner_agent_id=getattr(args, "owner", None) or scoped_agent_store.root_agent_id,
            kind=args.kind,
            trigger=trigger,
            failure_policy=FailurePolicy.from_dict(policy_overrides),
            fallback_statement=getattr(args, "fallback", None),
            idempotent=bool(getattr(args, "idempotent", False)),
        )
    except ObjectiveError as exc:
        print(f"error: {exc}")
        return 1

    if getattr(args, "json", False):
        _print_json(objective.to_dict())
        return 0
    print(f"objective:  {objective.objective_id}")
    print(f"title:      {objective.title}")
    print(f"kind:       {objective.kind}")
    print(f"trigger:    {json.dumps(objective.trigger, sort_keys=True)}")
    print(f"status:     {objective.status}")
    if not objective.idempotent:
        print(
            "\nnote: this objective is not marked --idempotent, so a transient failure "
            "will\n      still resubmit the same workflow. Only enable unattended "
            "missions whose\n      work is safe to re-run."
        )
    return 0


def _cmd_objective_list(args, store, caller_agent_id, scoped_agent_store) -> int:
    objectives = _store(store).list_objectives()
    if getattr(args, "status", None):
        objectives = [item for item in objectives if item.status == args.status]

    if getattr(args, "json", False):
        _print_json([item.to_dict() for item in objectives])
        return 0
    if not objectives:
        print("No objectives.")
        return 0
    print(f"{'OBJECTIVE_ID':<26} {'STATUS':<14} {'KIND':<10} {'FAILS':>5}  TITLE")
    for item in objectives:
        print(
            f"{item.objective_id:<26} {item.status:<14} {item.kind:<10} "
            f"{item.consecutive_failures:>5}  {item.title[:40]}"
        )
    return 0


def _cmd_objective_show(args, store, caller_agent_id, scoped_agent_store) -> int:
    objective = _store(store).load(args.objective_id)
    if objective is None:
        print(f"error: objective not found: {args.objective_id}")
        return 1
    if getattr(args, "json", False):
        _print_json(objective.to_dict())
        return 0

    print(f"objective:   {objective.objective_id}")
    print(f"title:       {objective.title}")
    print(f"statement:   {objective.statement}")
    print(f"kind:        {objective.kind}")
    print(f"status:      {objective.status}" + (f"  ({objective.status_reason})" if objective.status_reason else ""))
    print(f"trigger:     {json.dumps(objective.trigger, sort_keys=True)}")
    print(f"workflow:    {objective.current_workflow_id or '-'}")
    print(f"idempotent:  {objective.idempotent}")
    print(
        f"counters:    successes={objective.success_count} failures={objective.failure_count} "
        f"consecutive={objective.consecutive_failures} "
        f"retries={objective.retries_used} replans={objective.replans_used}"
    )
    if objective.next_attempt_at:
        print(f"next try:    {objective.next_attempt_at}")
    if objective.last_escalation:
        print(f"escalation:  {objective.last_escalation.get('reason')} -> {objective.last_escalation.get('status')}")
    if objective.history:
        print("\nhistory:")
        for item in objective.history[-10:]:
            print(
                f"  {item.at[:19]}  {item.outcome:<10} {item.classification or '-':<10} "
                f"{item.action or '-':<10} {item.detail[:50]}"
            )
    return 0


def _set_status(args, store, status: str, *, reason: str) -> int:
    objectives = _store(store)
    try:
        objective = objectives.set_status(args.objective_id, status, reason=reason)
    except ObjectiveError as exc:
        print(f"error: {exc}")
        return 1
    print(f"{objective.objective_id}: {objective.status}")
    return 0


def _cmd_objective_pause(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_status(args, store, STATUS_PAUSED, reason=getattr(args, "reason", None) or "paused by operator")


def _cmd_objective_resume(args, store, caller_agent_id, scoped_agent_store) -> int:
    """
    Resume clears the failure counters.

    A quarantined objective is quarantined *because* its budget ran out; putting
    it back to active without resetting the counters would send it straight back
    to quarantine on the next failure without ever retrying.
    """
    objectives = _store(store)
    try:
        objective = objectives.update(
            args.objective_id,
            status=STATUS_ACTIVE,
            status_reason=getattr(args, "reason", None) or "resumed by operator",
            consecutive_failures=0,
            retries_used=0,
            replans_used=0,
            fallbacks_used=0,
            next_attempt_at=None,
            current_workflow_id=None,
            last_escalation=None,
        )
    except ObjectiveError as exc:
        print(f"error: {exc}")
        return 1
    print(f"{objective.objective_id}: {objective.status}  (counters reset)")
    return 0


def _cmd_objective_abandon(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_status(
        args,
        store,
        STATUS_ABANDONED,
        reason=getattr(args, "reason", None) or "abandoned by operator",
    )


def _cmd_objective_run(args, store, caller_agent_id, scoped_agent_store) -> int:
    """Force a manual-trigger objective to be eligible on the next tick."""
    objectives = _store(store)
    try:
        objective = objectives.require(args.objective_id)
    except ObjectiveError as exc:
        print(f"error: {exc}")
        return 1
    trigger = dict(objective.trigger)
    trigger["type"] = "immediate"
    objectives.update(
        args.objective_id,
        status=STATUS_ACTIVE,
        status_reason="run requested by operator",
        trigger=trigger,
        last_completed_at=None,
        next_attempt_at=None,
    )
    print(f"{objective.objective_id}: queued for the supervisor's next tick")
    return 0
