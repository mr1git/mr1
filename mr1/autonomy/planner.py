"""
The planner — the only place the brain is called.

`Planner.plan()` turns an objective statement (plus its history and the reason
the last attempt failed) into a workflow spec. It is called from the
supervisor's PLAN phase and nowhere else, and PLAN only runs on a state change.
That is what makes a steady-state tick cost zero tokens.

`preflight_authority()` runs immediately after planning and before submission:
it evaluates every capability the spec would use against exactly the gate the
scheduler will apply. If the objective does not hold the authority, MR1
escalates and asks — it does not submit and hope, and it never grants itself
what it is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from mr1.autonomy.objectives import Objective
from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.capability_policy import (
    CapabilityMetadata,
    CapabilityRequest,
    PolicyEngine,
    build_scope_context,
)
from mr1.scoped_agents import AgentStore
from mr1.workflow_store import WorkflowStore


class PlanningError(RuntimeError):
    """The brain could not produce a usable workflow spec."""


class Planner(Protocol):
    def plan(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        """Return a workflow spec for this objective. May call the brain."""


@dataclass(frozen=True)
class MissingAuthority:
    capability_name: str
    task_label: str
    reason: str
    risk_score: float
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_name": self.capability_name,
            "task_label": self.task_label,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "args": dict(self.args),
        }


def build_planning_context(objective: Objective) -> dict[str, Any]:
    """What the brain is told. History and failures, never raw credentials."""
    return {
        "objective_id": objective.objective_id,
        "title": objective.title,
        "statement": (
            objective.fallback_statement
            if objective.use_fallback_next and objective.fallback_statement
            else objective.statement
        ),
        "kind": objective.kind,
        "is_fallback": bool(objective.use_fallback_next and objective.fallback_statement),
        "attempt": objective.success_count + objective.failure_count + 1,
        "last_failure": dict(objective.last_failure) if objective.last_failure else None,
        "history": [item.to_dict() for item in objective.history[-5:]],
        "idempotent": objective.idempotent,
    }


def preflight_authority(
    spec: dict[str, Any],
    objective: Objective,
    *,
    workflow_store: WorkflowStore,
    scoped_agents: AgentStore,
    consent_store: Any,
    workspace_root: Path,
    policy_engine: Optional[PolicyEngine] = None,
    capabilities: Optional[CapabilityRegistry] = None,
    now: Any = None,
) -> list[MissingAuthority]:
    """
    Would every capability in this spec actually be allowed to run?

    Evaluated against the same PolicyEngine, the same scope rules, and the same
    consent grants the scheduler will use — so a clean preflight means the work
    will not stall on a human halfway through, and a dirty one means MR1 asks
    *before* creating work it cannot finish.
    """
    engine = policy_engine or PolicyEngine()
    registry = capabilities or default_capability_registry()
    owner_id = objective.owner_agent_id or scoped_agents.root_agent_id
    grants = _active_grants(consent_store, objective.objective_id)
    owner = scoped_agents.require_agent(owner_id)

    missing: list[MissingAuthority] = []
    for task in spec.get("tasks", []) or []:
        capability_name = _capability_for_task(task)
        if capability_name is None:
            continue
        try:
            capability = registry.describe_capability(capability_name)
            metadata = CapabilityMetadata.from_dict(capability)
        except Exception as exc:
            missing.append(MissingAuthority(
                capability_name=capability_name,
                task_label=str(task.get("label") or "?"),
                reason=f"unknown_capability: {exc}",
                risk_score=1.0,
                args={},
            ))
            continue

        args = dict(task.get("tool_config") or task.get("watch_config") or {})
        request = CapabilityRequest(
            actor_id=owner_id,
            actor_type=owner.actor_category,
            actor_clearance=float(owner.security_clearance),
            invocation_mode="workflow",
            capability_name=capability_name,
            args=args,
            scope=build_scope_context(
                actor_id=owner_id,
                workspace_root=workspace_root,
                scoped_agent_store=scoped_agents,
                explicit_roots=None,
            ),
            objective_id=objective.objective_id,
        )
        decision = engine.evaluate(
            request,
            metadata,
            config_schema=capability.get("config_schema", {}),
            consent_grants=grants,
            now=now,
        )
        if decision.allowed:
            continue
        missing.append(MissingAuthority(
            capability_name=capability_name,
            task_label=str(task.get("label") or "?"),
            reason=decision.reason,
            risk_score=float(metadata.risk_score),
            args=args,
        ))
    return missing


def _active_grants(consent_store: Any, objective_id: str) -> list[Any]:
    if consent_store is None:
        return []
    try:
        return consent_store.list_active(objective_id=objective_id)
    except Exception:
        return []


def _capability_for_task(task: dict[str, Any]) -> Optional[str]:
    kind = task.get("task_kind", "agent")
    if kind == "tool":
        return task.get("tool_type")
    if kind == "watcher":
        return task.get("watcher_type")
    return None


class CompilerPlanner:
    """
    The production planner: the existing workflow-compiler agent.

    Reuses `WorkflowCompilerClient`, so an objective's plan goes through the
    same validated-envelope path a human's `compile-workflow` does — same
    validation, same governance, same audit. The supervisor holds the objective;
    the compiler still writes the spec.
    """

    def __init__(self, compiler_client: Any):
        self._client = compiler_client

    def plan(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        request = context.get("statement") or objective.statement
        try:
            result = self._client.compile(
                request=request,
                context=json.dumps(context, indent=2, sort_keys=True, default=str),
                caller_agent_id=objective.owner_agent_id,
                owner_agent_id=objective.owner_agent_id,
                mode="preview_only",
            )
        except Exception as exc:
            raise PlanningError(f"planner failed: {type(exc).__name__}: {exc}") from exc

        envelope = getattr(result, "envelope", None)
        if envelope is None or not getattr(envelope, "spec", None):
            raise PlanningError("planner returned no workflow spec")
        if getattr(envelope, "needs_confirmation", False):
            # An ambiguous plan is not a plan. Escalate rather than guess.
            raise AmbiguousObjective(
                "the planner is not confident enough to run this unattended: "
                + (getattr(envelope, "preview", "") or "")[:400]
            )
        return dict(envelope.spec)


class AmbiguousObjective(PlanningError):
    """The objective could not be turned into a workflow MR1 trusts."""
