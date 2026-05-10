"""call_capability action handler.

Invokes a registered capability via the capability runner; if the
action specifies `store_as`, the result is persisted into the agent's
step_context for use in subsequent steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mr1.mrn_loop import MRnStepResult, MRnStepRunner


def execute_call_capability(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    step_call_count: list[int],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    step_call_count[0] += 1

    capability_name = action["capability"].strip()
    config = dict(action.get("config") or {})
    store_as = action.get("store_as")
    step_id = f"{agent.agent_id}:{agent.current_iteration + 1}"

    result = runner._capability_runner.run_capability(
        capability_name,
        config,
        agent.agent_id,
        mode="direct",
        step_id=step_id,
    )

    stored_as = None
    if store_as and result.status in {"succeeded", "denied", "requires_approval"}:
        agent_ctx = runner._scoped_agents.require_agent(agent.agent_id)
        agent_ctx.step_context[store_as] = result.output
        runner._scoped_agents.save_agent(agent_ctx)
        stored_as = store_as

    return runner._persist_step(
        agent,
        action=action,
        status_after=action["next_status"],
        message=f"capability {capability_name}: {result.status}",
        capability_result={
            "status": result.status,
            "output": result.output,
            "error": result.error,
            "duration_ms": result.duration_ms,
            "capability": result.capability,
            "decision": dict(result.decision),
            "approval_request_id": result.approval_request_id,
            "audit_record_path": result.audit_record_path,
        },
        stored_as=stored_as,
        prompt_artifact_path=prompt_artifact_path,
    )
