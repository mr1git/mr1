"""send_message, ask_parent, and idle action handlers.

All three are thin: they either route a message via `runner._send_agent_message`
or persist a no-op idle step. They share access to the message store, so
they live together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mr1.mrn_loop import MRnStepResult, MRnStepRunner


def execute_send_message(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    message = runner._send_agent_message(
        agent,
        kind=action["message_kind"],
        subject=action["message_subject"],
        body=action["message_body"],
        workflow_id=action.get("workflow_id"),
        to_agent_id=action.get("to_agent_id"),
    )
    return runner._persist_step(
        agent,
        action=action,
        status_after=action["next_status"],
        message=f"message sent to {message.to_agent_id}",
        workflow_id=action.get("workflow_id"),
        message_id=message.message_id,
        created_parent_message_id=(
            message.message_id
            if message.to_agent_id == agent.parent_agent_id
            else None
        ),
        message_to_agent_id=message.to_agent_id,
        prompt_artifact_path=prompt_artifact_path,
    )


def execute_ask_parent(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    message = runner._send_agent_message(
        agent,
        kind="question",
        subject=f"Parent request from {agent.title}",
        body=action["parent_request"],
        to_agent_id=agent.parent_agent_id,
    )
    return runner._persist_step(
        agent,
        action=action,
        status_after="waiting",
        message="parent clarification requested",
        message_id=message.message_id,
        parent_request=action["parent_request"],
        created_parent_message_id=message.message_id,
        message_to_agent_id=message.to_agent_id,
        prompt_artifact_path=prompt_artifact_path,
    )


def execute_idle(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    return runner._persist_step(
        agent,
        action=action,
        status_after="idle",
        message="agent remains idle",
        prompt_artifact_path=prompt_artifact_path,
    )
