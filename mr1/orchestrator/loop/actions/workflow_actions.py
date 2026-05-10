"""create_workflow and inspect_workflow action handlers.

These two actions both touch the workflow store and share helpers
(workflow summarization, confirmation reporting, scoped access checks),
so they live together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from mr1.scoped_agents import AgentScopeError
from mr1.workflow_models import Provenance, Workflow

from ._text import _compact

if TYPE_CHECKING:
    from mr1.mrn_loop import MRnStepResult, MRnStepRunner


def execute_create_workflow(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    compile_context = _build_create_workflow_context(runner, agent, action)
    client = (
        runner._workflow_compiler_client
        or getattr(runner._workflow_authoring, "_workflow_compiler_client", None)
    )
    if client is not None:
        compiled = client.compile(
            action["workflow_request"],
            compile_context,
            agent.agent_id,
            agent.agent_id,
            "preview_only",
        )
        preview = compiled.envelope.preview
        spec = compiled.envelope.spec
        assumptions = compiled.envelope.assumptions
        risks = compiled.envelope.risks
        needs_confirmation = compiled.envelope.needs_confirmation
    else:
        authored = runner._workflow_authoring.author_request(
            action["workflow_request"],
            caller_agent_id=agent.agent_id,
            owner_agent_id=agent.agent_id,
        )
        preview = authored.preview_text
        spec = authored.spec
        assumptions = list(authored.assumptions)
        risks = list(authored.risks)
        needs_confirmation = bool(authored.needs_confirmation)

    # When confirmation is explicitly required at the run-policy level,
    # always stop for review. When that policy is disabled (for example
    # after the parent already reviewed the plan), submit immediately
    # even if the compiler envelope would normally ask for confirmation.
    should_require_confirmation = bool(runner._require_confirmation_for_workflows)

    if not should_require_confirmation:
        submission = runner._workflow_authoring.submit(
            spec,
            created_by=Provenance(type="agent", id=agent.agent_id),
            caller_agent_id=agent.agent_id,
            owner_agent_id=agent.agent_id,
            workflow_metadata=(
                {
                    "compiled_with_memory": compiled.compiled_with_memory,
                    "memory_refs_used": list(compiled.envelope.memory_refs_used),
                    "memory_tools_used": list(compiled.memory_tools_used or []),
                    "memory_context_summary": compiled.memory_context_summary,
                }
                if client is not None else None
            ),
        )
        created_workflow = runner._workflow_store.load_workflow(submission.workflow_id)
        return runner._persist_step(
            agent,
            action=action,
            status_after=action["next_status"],
            message=submission.message,
            workflow_id=submission.workflow_id,
            created_workflow_id=submission.workflow_id,
            created_workflow_status=(
                created_workflow.status.value if created_workflow is not None else None
            ),
            workflow_submitted=True,
            prompt_artifact_path=prompt_artifact_path,
        )

    report_path = runner._scoped_agents.write_report(
        agent.agent_id,
        _build_confirmation_report(
            agent,
            action["reason"],
            preview,
            assumptions,
            risks,
        ),
    )
    parent_message = None
    if agent.parent_agent_id:
        parent_message = runner._send_agent_message(
            agent,
            kind="report",
            subject=f"Workflow confirmation needed from {agent.title}",
            body="\n".join([
                "Workflow creation requires confirmation.",
                f"Agent: {agent.agent_id}",
                f"Reason: {action['reason']}",
                f"Report path: {report_path}",
            ]),
            to_agent_id=agent.parent_agent_id,
        )
    result = runner._persist_step(
        agent,
        action=action,
        status_after="reporting",
        message="workflow creation requires confirmation",
        report_path=str(report_path),
        message_id=parent_message.message_id if parent_message is not None else None,
        created_parent_message_id=(
            parent_message.message_id if parent_message is not None else None
        ),
        message_to_agent_id=parent_message.to_agent_id if parent_message is not None else None,
        confirmation_required=True,
        workflow_submitted=False,
        prompt_artifact_path=prompt_artifact_path,
    )
    runner._emit_mrn_reported(agent, result)
    return result


def execute_inspect_workflow(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    workflow = runner._workflow_store.load_workflow(action["workflow_id"])
    if workflow is None:
        raise ValueError(f"workflow not found: {action['workflow_id']}")
    workflow = runner._scoped_agents.normalize_workflow_ownership(workflow)
    if not runner._scoped_agents.can_agent_access_workflow(agent.agent_id, workflow):
        raise AgentScopeError("access denied: workflow not in agent scope")
    summary = summarize_workflow(runner, workflow)
    return runner._persist_step(
        agent,
        action=action,
        status_after=action["next_status"],
        message=f"inspected workflow {workflow.workflow_id}",
        workflow_id=workflow.workflow_id,
        workflow_summary=summary,
        prompt_artifact_path=prompt_artifact_path,
    )


def summarize_workflow(runner: "MRnStepRunner", workflow: Workflow) -> dict[str, Any]:
    tasks = []
    for task in workflow.tasks.values():
        output = runner._workflow_store.load_task_output(workflow.workflow_id, task.task_id)
        tasks.append({
            "task_id": task.task_id,
            "label": task.label,
            "status": task.status.value,
            "summary": task.result_summary,
            "output_summary": output.summary if output is not None else None,
            "output_text": _compact(output.text, limit=180) if output is not None else None,
        })
    events = [
        {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "task_id": event.task_id,
            "message": _compact(event.message, limit=120),
        }
        for event in runner._workflow_store.load_events(workflow.workflow_id, limit=5)
    ]
    return {
        "workflow_id": workflow.workflow_id,
        "title": workflow.title,
        "status": workflow.status.value,
        "owner_agent_id": workflow.owner_agent_id,
        "tasks": tasks,
        "recent_events": events,
    }


def _build_create_workflow_context(
    runner: "MRnStepRunner", agent, action: dict[str, Any]
) -> str:
    parts = [
        f"Agent: {agent.agent_id} ({agent.title})",
        f"Mission: {_compact(agent.mission, limit=400)}",
        f"Scoped workflows visible: {len(runner._workflow_summaries(agent))}",
    ]
    if isinstance(action.get("workflow_context"), str) and action["workflow_context"].strip():
        parts.extend([
            "Extra workflow context:",
            action["workflow_context"].strip(),
        ])
    return "\n\n".join(parts)


def _build_confirmation_report(
    agent,
    reason: str,
    preview: str,
    assumptions: list[str],
    risks: list[str],
) -> str:
    lines = [
        f"# MRn Workflow Preview for {agent.title}",
        "",
        f"- agent_id: {agent.agent_id}",
        f"- mission: {_compact(agent.mission, limit=240)}",
        f"- iteration: {agent.current_iteration + 1}",
        f"- reason: {reason}",
        "",
        "## Preview",
        preview.strip() or "-",
        "",
        "## Assumptions",
    ]
    if assumptions:
        lines.extend(f"- {item}" for item in assumptions)
    else:
        lines.append("- none")
    lines.extend(["", "## Risks"])
    if risks:
        lines.extend(f"- {item}" for item in risks)
    else:
        lines.append("- none")
    return "\n".join(lines)
