"""CLI handlers for the autonomy service and its control plane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mr1.autonomy.control import (
    MODE_HALTED,
    MODE_PAUSED,
    MODE_RUNNING,
    MODE_STOPPING,
    ControlPlane,
    ServiceLock,
    ServiceLockError,
)
from mr1.autonomy.health import heartbeat_age_s, read_health
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.clock import default_clock


def _runtime_root(store) -> Path:
    return Path(store.root).parent


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _config_from_args(args) -> SupervisorConfig:
    config = SupervisorConfig()
    for attr in (
        "tick_interval_s",
        "max_concurrent_workflows",
        "max_plans_per_hour",
        "max_workflows_per_objective_per_day",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config, attr, value)
    return config.validate()


def _build_planner(scoped_agent_store):
    """
    The production planner: the existing workflow-compiler agent.

    An objective's plan goes through the same validated-envelope path a human's
    `compile-workflow` does — same validation, same governance, same audit.
    """
    from mr1.autonomy.planner import CompilerPlanner
    from mr1.workflow_compiler import WorkflowCompilerClient

    return CompilerPlanner(
        WorkflowCompilerClient(scoped_agent_store=scoped_agent_store)
    )


def _cmd_serve(args, store, caller_agent_id, scoped_agent_store) -> int:
    runtime_root = _runtime_root(store)
    supervisor = Supervisor(
        runtime_root,
        config=_config_from_args(args),
        workflow_store=store,
        scoped_agent_store=scoped_agent_store,
        workspace_root=Path(getattr(args, "workspace_root", None) or Path.cwd()),
        planner=(
            None
            if getattr(args, "no_planner", False) else
            _build_planner(scoped_agent_store)
        ),
        enable_triage=bool(getattr(args, "triage", False)),
    )
    print(f"[mr1] supervisor serving runtime_root={runtime_root}")
    print(
        f"[mr1] tick_interval={supervisor.config.tick_interval_s}s"
        "  (ctrl-c, or `mr1 stop` from another shell, to exit)"
    )
    try:
        return supervisor.serve()
    except ServiceLockError as exc:
        print(f"error: {exc}")
        return 1


def _set_mode(args, store, mode: str, *, default_reason: str) -> int:
    control = ControlPlane(_runtime_root(store))
    state = control.set_mode(
        mode,
        reason=getattr(args, "reason", None) or default_reason,
        requested_by=getattr(args, "requested_by", None) or "operator",
    )
    if getattr(args, "json", False):
        _print_json(state.to_dict())
    else:
        print(f"mode: {state.mode}  ({state.reason})")
    return 0


def _cmd_pause(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_mode(args, store, MODE_PAUSED, default_reason="paused by operator")


def _cmd_resume(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_mode(args, store, MODE_RUNNING, default_reason="resumed by operator")


def _cmd_stop(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_mode(args, store, MODE_STOPPING, default_reason="graceful stop requested")


def halt_runtime(
    runtime_root: Path,
    *,
    reason: str,
    requested_by: str,
    clock=None,
) -> dict[str, Any]:
    """
    Emergency stop, applied from wherever the halt was requested.

    Authority is revoked here rather than only inside a running supervisor: a
    halt that takes effect only if a supervisor happens to be alive to read it
    is not a halt. A live supervisor also acts on the mode on its next tick —
    cancelling in-flight tasks and exiting — and both paths are idempotent.
    """
    clock = clock or default_clock()
    state = ControlPlane(runtime_root, clock=clock).set_mode(
        MODE_HALTED,
        reason=reason,
        requested_by=requested_by,
    )
    payload: dict[str, Any] = {
        **state.to_dict(),
        "revoked_grants": [],
        "paused_objectives": [],
        "supervisor_running": ServiceLock(runtime_root).is_held_by_live_process(),
    }
    payload.update(
        _revoke_standing_authority(
            runtime_root,
            reason=reason,
            requested_by=requested_by,
            clock=clock,
        )
    )
    return payload


def _revoke_standing_authority(
    runtime_root: Path,
    *,
    reason: str,
    requested_by: str,
    clock,
) -> dict[str, Any]:
    """Revoke every consent grant and park every objective. A4/A5 surfaces."""
    revoked: list[str] = []
    paused: list[str] = []
    try:
        from mr1.autonomy.consent import ConsentGrantStore

        revoked = ConsentGrantStore(runtime_root, clock=clock).revoke_all(
            revoked_by=requested_by,
            reason=f"halt: {reason}",
        )
    except ImportError:  # pragma: no cover - consent lands in A4
        pass
    try:
        from mr1.autonomy.objectives import ObjectiveStore

        paused = ObjectiveStore(runtime_root, clock=clock).pause_all(
            reason=f"halt: {reason}",
        )
    except ImportError:  # pragma: no cover - objectives land in A5
        pass
    return {"revoked_grants": revoked, "paused_objectives": paused}


def _cmd_halt(args, store, caller_agent_id, scoped_agent_store) -> int:
    payload = halt_runtime(
        _runtime_root(store),
        reason=getattr(args, "reason", None) or "halted by operator",
        requested_by=getattr(args, "requested_by", None) or "operator",
    )
    if getattr(args, "json", False):
        _print_json(payload)
        return 0
    print(f"mode: halted  ({payload['reason']})")
    print(f"consent grants revoked: {len(payload['revoked_grants'])}")
    print(f"objectives paused: {len(payload['paused_objectives'])}")
    if payload["supervisor_running"]:
        print("a supervisor is live; it will cancel running tasks and exit on its next tick")
    return 0


def status_payload(runtime_root: Path) -> dict[str, Any]:
    clock = default_clock()
    state = ControlPlane(runtime_root).read()
    health = read_health(runtime_root)
    lock = ServiceLock(runtime_root)
    age = heartbeat_age_s(health, clock=clock)
    payload: dict[str, Any] = {
        "mode": state.mode,
        "reason": state.reason,
        "supervisor_running": lock.is_held_by_live_process(),
        "supervisor_pid": lock.read_pid(),
        "heartbeat_at": (health or {}).get("supervisor_heartbeat_at"),
        "heartbeat_age_s": round(age, 1) if age is not None else None,
        "health": (health or {}).get("doctor_status", "unknown"),
        "gauges": dict((health or {}).get("gauges") or {}),
        "objectives": {},
        "active_grants": 0,
        "budget": {},
    }
    payload.update(_autonomy_status(runtime_root, clock))
    return payload


def _autonomy_status(runtime_root: Path, clock) -> dict[str, Any]:
    """Objective/grant/budget rollups. A4/A5 surfaces."""
    extra: dict[str, Any] = {}
    try:
        from mr1.autonomy.objectives import ObjectiveStore

        by_status: dict[str, int] = {}
        for objective in ObjectiveStore(runtime_root, clock=clock).list_objectives():
            by_status[objective.status] = by_status.get(objective.status, 0) + 1
        extra["objectives"] = by_status
    except ImportError:  # pragma: no cover - objectives land in A5
        pass
    try:
        from mr1.autonomy.consent import ConsentGrantStore

        extra["active_grants"] = len(ConsentGrantStore(runtime_root, clock=clock).list_active())
    except ImportError:  # pragma: no cover - consent lands in A4
        pass
    try:
        from mr1.autonomy.budget import BudgetLedger

        extra["budget"] = BudgetLedger(runtime_root, clock=clock).snapshot()
    except ImportError:  # pragma: no cover - budgets land in A5
        pass
    return extra


def _cmd_status(args, store, caller_agent_id, scoped_agent_store) -> int:
    payload = status_payload(_runtime_root(store))
    if getattr(args, "json", False):
        _print_json(payload)
        return 0

    gauges = payload["gauges"]
    print(f"mode:              {payload['mode']}" + (f"  ({payload['reason']})" if payload["reason"] else ""))
    print(
        "supervisor:        "
        + (
            f"running (pid {payload['supervisor_pid']})"
            if payload["supervisor_running"] else
            "not running"
        )
    )
    age = payload["heartbeat_age_s"]
    if age is None:
        print("heartbeat:         none")
    else:
        stale = age > 3 * SupervisorConfig().tick_interval_s
        print(f"heartbeat:         {age:.0f}s ago" + ("  [STALE]" if stale else ""))
    print(f"health:            {payload['health']}")
    print(f"active workflows:  {gauges.get('active_workflows', 0)}")
    oldest = gauges.get("oldest_pending_approval_age_s")
    print(
        "oldest approval:   "
        + (f"{float(oldest):.0f}s" if oldest is not None else "none pending")
    )
    print(f"consent grants:    {payload['active_grants']} active")
    objectives = payload["objectives"]
    print(
        "objectives:        "
        + (
            ", ".join(f"{name}={count}" for name, count in sorted(objectives.items()))
            if objectives else
            "none"
        )
    )
    budget = payload["budget"]
    if budget:
        print(
            f"budget:            plans {budget.get('plans_this_hour', 0)}/{budget.get('max_plans_per_hour', 0)}"
            f" this hour, actions {budget.get('actions_this_hour', 0)}/{budget.get('max_actions_per_hour', 0)}"
        )
    errors = gauges.get("scheduler_tick_errors", 0)
    if errors:
        print(f"scheduler errors:  {errors}  (last: {gauges.get('scheduler_last_tick_error')})")
    return 0
