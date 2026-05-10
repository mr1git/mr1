"""write_report action handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ._text import _compact

if TYPE_CHECKING:
    from mr1.mrn_loop import MRnStepResult, MRnStepRunner


def execute_write_report(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    content = "\n".join([
        f"# MRn Report for {agent.title}",
        "",
        f"- mission: {_compact(agent.mission, limit=240)}",
        f"- iteration: {agent.current_iteration + 1}",
        f"- reason: {action['reason']}",
        "",
        action["report"].rstrip(),
    ])
    report_path = runner._scoped_agents.write_report(agent.agent_id, content)
    message = None
    if agent.parent_agent_id:
        message = runner._send_agent_message(
            agent,
            kind="report",
            subject=f"Report from {agent.title}",
            body="\n".join([content, "", f"Report path: {report_path}"]),
        )
    result = runner._persist_step(
        agent,
        action=action,
        status_after=action["next_status"],
        message="report written",
        report_path=str(report_path),
        message_id=message.message_id if message is not None else None,
        created_parent_message_id=message.message_id if message is not None else None,
        message_to_agent_id=message.to_agent_id if message is not None else None,
        prompt_artifact_path=prompt_artifact_path,
    )
    runner._emit_mrn_reported(agent, result)
    return result
