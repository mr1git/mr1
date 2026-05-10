"""Runtime grounding construction for MR1.

The orchestrator builds a "runtime grounding" payload — a snapshot of
agents, workflows, messages, approvals, and events — that is injected
into MR1's per-turn prompt so the model has a fresh, authoritative
view of system state. Pinned IDs (resolved references, recently
created/referenced agents and workflows) bubble to the top of each
list so the model is most likely to see them within the limits.

Constants:
    GROUNDING_AGENT_LIMIT, GROUNDING_WORKFLOW_LIMIT,
    GROUNDING_MESSAGE_LIMIT, GROUNDING_APPROVAL_LIMIT,
    GROUNDING_EVENT_LIMIT
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mr1.mr1 import MR1


GROUNDING_AGENT_LIMIT = 8
GROUNDING_WORKFLOW_LIMIT = 8
GROUNDING_MESSAGE_LIMIT = 8
GROUNDING_APPROVAL_LIMIT = 8
GROUNDING_EVENT_LIMIT = 10


def prioritize_items(
    items: list[dict[str, Any]],
    *,
    id_field: str,
    pinned_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    pinned_set = {item for item in pinned_ids if item}
    prioritized = [item for item in items if item.get(id_field) in pinned_set]
    remaining = [item for item in items if item.get(id_field) not in pinned_set]
    return (prioritized + remaining)[:limit]


def grounding_message_ids_from_agents(agents: list[dict[str, Any]]) -> list[str]:
    message_ids: list[str] = []
    for agent in agents:
        for field in (
            "latest_inbox_messages",
            "latest_outbox_messages",
            "pending_parent_messages",
        ):
            for item in agent.get(field, []):
                message_id = item.get("message_id")
                if isinstance(message_id, str) and message_id:
                    message_ids.append(message_id)
    return message_ids


def grounding_agents(
    mr1: "MR1",
    *,
    limit: int = GROUNDING_AGENT_LIMIT,
    pinned_agent_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    pinned = pinned_agent_ids or [
        mr1._state.last_referenced_agent_id,
        mr1._state.last_created_agent_id,
    ]
    return mr1._runtime_access.list_agents(
        caller_agent_id=mr1._root_agent_id,
        limit=limit,
        pinned_agent_ids=pinned,
    )


def grounding_workflows(
    mr1: "MR1",
    *,
    limit: int = GROUNDING_WORKFLOW_LIMIT,
    pinned_workflow_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    pinned = pinned_workflow_ids or [
        mr1._state.last_referenced_workflow_id,
        mr1._state.last_created_workflow_id,
    ]
    return mr1._runtime_access.list_workflows(
        caller_agent_id=mr1._root_agent_id,
        limit=limit,
        pinned_workflow_ids=pinned,
    )


def grounding_messages(
    mr1: "MR1",
    *,
    limit: int = GROUNDING_MESSAGE_LIMIT,
    pinned_message_ids: Optional[list[str]] = None,
    referenced_message_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    pinned_ids = [item for item in (pinned_message_ids or []) if item]
    referenced_ids = [item for item in (referenced_message_ids or []) if item]
    base = mr1._runtime_access.list_messages(
        caller_agent_id=mr1._root_agent_id,
        to_agent_id=mr1._root_agent_id,
        status="unread",
        include_archived=False,
    )
    extra = mr1._runtime_access.list_messages(
        caller_agent_id=mr1._root_agent_id,
        message_ids=sorted(set(pinned_ids + referenced_ids)),
        include_archived=True,
    )
    deduped: dict[str, dict[str, Any]] = {}
    for item in base + extra:
        deduped[item["message_id"]] = item
    payload = list(deduped.values())
    payload.sort(key=lambda item: (item["created_at"], item["message_id"]), reverse=True)
    return prioritize_items(
        payload,
        id_field="message_id",
        pinned_ids=pinned_ids + referenced_ids,
        limit=limit,
    )


def grounding_approvals(
    mr1: "MR1", *, limit: int = GROUNDING_APPROVAL_LIMIT
) -> list[dict[str, Any]]:
    return mr1._runtime_access.list_pending_approvals(
        caller_agent_id=mr1._root_agent_id,
        limit=limit,
    )


def grounding_events(
    mr1: "MR1", *, limit: int = GROUNDING_EVENT_LIMIT
) -> list[dict[str, Any]]:
    return mr1._runtime_access.list_recent_events(
        caller_agent_id=mr1._root_agent_id,
        limit=limit,
    )


def build_runtime_grounding(
    mr1: "MR1",
    *,
    resolved_references: Optional[dict[str, Any]] = None,
    ambiguities: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    resolved = dict(resolved_references or {})
    pinned_agent_ids = [
        item.get("id")
        for item in resolved.values()
        if item.get("kind") == "agent"
    ]
    pinned_workflow_ids = [
        item.get("id")
        for item in resolved.values()
        if item.get("kind") == "workflow"
    ]
    pinned_message_ids = [
        item.get("id")
        for item in resolved.values()
        if item.get("kind") == "message"
    ]
    agents = grounding_agents(mr1, pinned_agent_ids=pinned_agent_ids)
    workflows = grounding_workflows(mr1, pinned_workflow_ids=pinned_workflow_ids)
    referenced_message_ids = grounding_message_ids_from_agents(agents)
    messages = grounding_messages(
        mr1,
        pinned_message_ids=pinned_message_ids,
        referenced_message_ids=referenced_message_ids,
    )
    return {
        "agents": agents,
        "workflows": workflows,
        "messages": messages,
        "approvals": grounding_approvals(mr1),
        "events": grounding_events(mr1),
        "resolved_references": resolved,
        "ambiguities": list(ambiguities or []),
    }


def format_runtime_grounding_block(runtime_grounding: dict[str, Any]) -> str:
    return "\n".join([
        "RUNTIME STATE IS SOURCE OF TRUTH.",
        "If this conflicts with remembered conversation context, trust runtime state.",
        'Resolve references such as "that MR2" using the runtime state first.',
        "Preview fields may be truncated. If a preview says full_available=true, full detail exists in backend observations even if not shown here.",
        "",
        "=== RUNTIME GROUNDING ===",
        json.dumps(runtime_grounding, indent=2, sort_keys=True),
        "=== END RUNTIME GROUNDING ===",
    ])
