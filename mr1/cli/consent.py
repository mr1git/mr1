"""CLI handlers for objective-scoped consent grants (`mr1 grant ...`)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mr1.autonomy.consent import ConsentGrantError, ConsentGrantStore
from mr1.clock import default_clock


_DURATION_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smhdw]?)$")
_DURATION_UNITS = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86_400.0,
    "w": 604_800.0,
}


def parse_duration(text: str) -> float:
    """`7d`, `12h`, `90m`, `3600` — a TTL is mandatory, so this must be strict."""
    match = _DURATION_PATTERN.match(str(text).strip().lower())
    if match is None:
        raise ValueError(f"invalid duration: {text!r} (try 7d, 12h, 90m, 3600)")
    value = float(match.group("value"))
    unit = match.group("unit") or "s"
    seconds = value * _DURATION_UNITS[unit]
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return seconds


def _store(store, scoped_agent_store) -> ConsentGrantStore:
    return ConsentGrantStore(
        Path(store.root).parent,
        clock=default_clock(),
        scoped_agent_store=scoped_agent_store,
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _cmd_grant_create(args, store, caller_agent_id, scoped_agent_store) -> int:
    grants = _store(store, scoped_agent_store)
    try:
        ttl_s = parse_duration(args.ttl)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    arg_predicate: dict[str, Any] = {}
    if getattr(args, "allow", None):
        # `--allow` is shorthand for a regex predicate on the capability's
        # primary argument: shell_command's argv, everything else's `path`.
        field = "argv" if args.capability == "shell_command" else "path"
        arg_predicate[field] = {"regex": args.allow}
    if getattr(args, "predicate", None):
        try:
            parsed = json.loads(args.predicate)
        except json.JSONDecodeError as exc:
            print(f"error: --predicate must be JSON: {exc}")
            return 1
        if not isinstance(parsed, dict):
            print("error: --predicate must be a JSON object")
            return 1
        arg_predicate.update(parsed)

    try:
        grant = grants.create(
            objective_id=args.objective,
            capability_name=args.capability,
            scope_roots=list(args.scope),
            max_risk=float(args.max_risk),
            granted_by=caller_agent_id,
            ttl_s=ttl_s,
            arg_predicate=arg_predicate,
            reason=args.reason or "",
        )
    except ConsentGrantError as exc:
        print(f"error: {exc}")
        return 1

    if getattr(args, "json", False):
        _print_json(grant.to_dict())
        return 0
    print(f"grant:      {grant.grant_id}")
    print(f"objective:  {grant.grantee_id}")
    print(f"capability: {grant.capability_name}  (max_risk {grant.max_risk})")
    print(f"scope:      {', '.join(grant.scope_roots)}")
    if grant.arg_predicate:
        print(f"predicate:  {json.dumps(grant.arg_predicate, sort_keys=True)}")
    print(f"expires:    {grant.expires_at}")
    return 0


def _cmd_grant_list(args, store, caller_agent_id, scoped_agent_store) -> int:
    grants = _store(store, scoped_agent_store)
    now = grants._clock.now()
    items = grants.list_active() if getattr(args, "active", False) else grants.list_grants()
    if getattr(args, "objective", None):
        items = [item for item in items if item.grantee_id == args.objective]

    if getattr(args, "json", False):
        _print_json([
            {**item.to_dict(), "status": item.status(now)}
            for item in items
        ])
        return 0
    if not items:
        print("No consent grants.")
        return 0
    print(f"{'GRANT_ID':<28} {'STATUS':<8} {'CAPABILITY':<18} {'OBJECTIVE':<24} {'USES':>4}  EXPIRES")
    for item in items:
        print(
            f"{item.grant_id:<28} {item.status(now):<8} {item.capability_name:<18} "
            f"{item.grantee_id:<24} {item.use_count:>4}  {item.expires_at[:19]}"
        )
    return 0


def _cmd_grant_show(args, store, caller_agent_id, scoped_agent_store) -> int:
    grants = _store(store, scoped_agent_store)
    grant = grants.load(args.grant_id)
    if grant is None:
        print(f"error: consent grant not found: {args.grant_id}")
        return 1
    payload = {**grant.to_dict(), "status": grant.status(grants._clock.now())}
    _print_json(payload)
    return 0


def _cmd_grant_revoke(args, store, caller_agent_id, scoped_agent_store) -> int:
    grants = _store(store, scoped_agent_store)
    reason = getattr(args, "reason", None) or "revoked by operator"

    if getattr(args, "all", False):
        revoked = grants.revoke_all(
            revoked_by=caller_agent_id,
            reason=reason,
            objective_id=getattr(args, "objective", None),
        )
        if getattr(args, "json", False):
            _print_json({"revoked": revoked})
        else:
            print(f"revoked {len(revoked)} grant(s)")
        return 0

    if not getattr(args, "grant_id", None):
        print("error: pass a grant id, or --all")
        return 1
    grant = grants.revoke(args.grant_id, revoked_by=caller_agent_id, reason=reason)
    if grant is None:
        print(f"error: consent grant not found: {args.grant_id}")
        return 1
    if getattr(args, "json", False):
        _print_json(grant.to_dict())
    else:
        print(f"revoked: {grant.grant_id}  ({reason})")
    return 0
