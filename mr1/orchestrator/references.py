"""Runtime reference resolution for the MR1 root orchestrator.

Resolves user-input strings like agent IDs, workflow IDs, "that agent",
"the workflow from earlier", and bulk references ("kill all agents")
into concrete runtime payloads.

Originally defined as instance methods on `MR1`. Extracted as free
functions that take the `MR1` instance (named `root`) as their first
parameter. `MR1` retains thin wrappers so existing callers keep working.

Module-level constants like `_AGENT_ID_PATTERN` live in
`mr1.orchestrator.root`. They are imported here at module top; this
works because the `from mr1.orchestrator import references` line in
`root.py` is placed *after* the constants block, so the constants are
defined by the time this module loads.
"""

import re
from typing import Any, Optional

from mr1.orchestrator.root import (
    _AGENT_CREATED_REFERENCE_ALIASES,
    _AGENT_ID_PATTERN,
    _BULK_AGENT_ACTIONS,
    _MESSAGE_ID_PATTERN,
    _TASK_ID_PATTERN,
    _WORKFLOW_ID_PATTERN,
    _normalize_routing_text,
)


def agent_reference_payload(root, agent) -> dict[str, Any]:
    return {
        "kind": "agent",
        "id": agent.agent_id,
        "title": agent.title,
    }

def workflow_reference_payload(root, workflow) -> dict[str, Any]:
    return {
        "kind": "workflow",
        "id": workflow.workflow_id,
        "title": workflow.title,
    }

def message_reference_payload(root, message) -> dict[str, Any]:
    return {
        "kind": "message",
        "id": message.message_id,
        "subject": message.subject,
        "from_agent_id": message.from_agent_id,
        "to_agent_id": message.to_agent_id,
    }

def is_mr_style_agent_title(title: str) -> bool:
    return bool(re.fullmatch(r"mr\d+", title, flags=re.IGNORECASE))

def bulk_agent_action(normalized: str) -> str:
    for action in _BULK_AGENT_ACTIONS:
        if re.search(rf"\b{action}\b", normalized):
            return action
    return "manage"

def agent_title_reference_kind(root, normalized: str, title: str) -> Optional[str]:
    escaped_title = re.escape(title)
    if not re.search(rf"\b{escaped_title}\b", normalized):
        return None

    bulk_patterns = (
        rf"\ball\s+{escaped_title}\s+agents\b",
        rf"\b(?:all\s+(?:of\s+)?)?(?:those|these|the)\s+{escaped_title}\s+agents\b",
        rf"\bevery\s+agent\s+named\s+{escaped_title}\b",
    )
    if any(re.search(pattern, normalized) for pattern in bulk_patterns):
        return "bulk"
    if (
        re.search(rf"\b{escaped_title}\s+agents\b", normalized)
        and any(re.search(rf"\b{action}\b", normalized) for action in _BULK_AGENT_ACTIONS)
    ):
        return "bulk"

    single_patterns = (
        rf"\bwhat\s+is\s+(?:that\s+|the\s+)?{escaped_title}\s+doing\b",
        rf"\b(?:run|kill|terminate|resume|message|ask)\s+(?:that\s+|the\s+)?{escaped_title}\b",
        rf"\b(?:status|inbox|approval)\s+(?:for|of)?\s*(?:that\s+|the\s+)?{escaped_title}\b",
        rf"\bworkflow\s+owner\s+(?:for|of)?\s*(?:that\s+|the\s+)?{escaped_title}\b",
        rf"\b(?:that|the)\s+{escaped_title}\s+(?:agent|child)\b",
        rf"\b(?:agent|child)\s+(?:named\s+)?{escaped_title}\b",
        rf"\b{escaped_title}\s+(?:agent|child)\b",
    )
    if any(re.search(pattern, normalized) for pattern in single_patterns):
        return "single"
    if root._is_mr_style_agent_title(title):
        return "single"
    return None

def resolve_runtime_references(
    root,
    user_input: str,
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    ambiguities: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    bulk_targets: list[dict[str, Any]] = []
    normalized = _normalize_routing_text(user_input)
    visible_agents = root._scoped_agents.list_visible_agents(root._root_agent_id)
    agents_by_id = {agent.agent_id: agent for agent in visible_agents}
    workflows = root._workflow_store.list_workflows()
    workflows_by_id = {workflow.workflow_id: workflow for workflow in workflows}

    for agent_id in sorted(set(_AGENT_ID_PATTERN.findall(user_input))):
        agent = agents_by_id.get(agent_id)
        if agent is None:
            missing.append({"reference": agent_id, "kind": "agent"})
            continue
        resolved[agent_id] = root._agent_reference_payload(agent)

    workflow_ids = set(re.findall(r"\bwf-\d{8}T\d{6}-[0-9a-f]{6}\b", user_input))
    explicit_workflow_id = root._workflow_authoring.extract_workflow_id(user_input)
    if explicit_workflow_id:
        workflow_ids.add(explicit_workflow_id)
    for workflow_id in sorted(workflow_ids):
        workflow = workflows_by_id.get(workflow_id)
        if workflow is None:
            missing.append({"reference": workflow_id, "kind": "workflow"})
            continue
        resolved[workflow_id] = root._workflow_reference_payload(workflow)

    for message_id in sorted(set(_MESSAGE_ID_PATTERN.findall(user_input))):
        message = root._message_store.get_message(message_id)
        if message is None:
            missing.append({"reference": message_id, "kind": "message"})
            continue
        resolved[message_id] = root._message_reference_payload(message)

    title_matches: dict[str, list[Any]] = {}
    for agent in visible_agents:
        if agent.status != "terminated":
            title_matches.setdefault(agent.title.lower(), []).append(agent)

    for title, candidates in title_matches.items():
        plain_match = re.search(rf"\b{re.escape(title)}\b", normalized)
        that_match = re.search(rf"\bthat\s+{re.escape(title)}\b", normalized)
        if not plain_match and not that_match:
            continue
        reference_kind = root._agent_title_reference_kind(normalized, title)
        if reference_kind is None:
            continue
        if reference_kind == "bulk":
            bulk_targets.append({
                "reference": f"{candidates[0].title} agents",
                "kind": "agent",
                "action": root._bulk_agent_action(normalized),
                "candidates": [root._agent_reference_payload(item) for item in candidates],
            })
            continue
        if len(candidates) == 1:
            source = f"that {candidates[0].title}" if that_match else candidates[0].title
            resolved[source] = root._agent_reference_payload(candidates[0])
            continue
        explicit_agent_ids = set(_AGENT_ID_PATTERN.findall(user_input))
        id_matched = [c for c in candidates if c.agent_id in explicit_agent_ids]
        if len(id_matched) == 1:
            source = f"that {id_matched[0].title}" if that_match else id_matched[0].title
            resolved[source] = root._agent_reference_payload(id_matched[0])
            continue
        pinned_id = None
        if that_match:
            pinned_id = root._state.last_referenced_agent_id or root._state.last_created_agent_id
        if pinned_id is not None:
            pinned_candidate = next((item for item in candidates if item.agent_id == pinned_id), None)
            if pinned_candidate is not None:
                resolved[f"that {pinned_candidate.title}"] = root._agent_reference_payload(pinned_candidate)
                continue
        ambiguities.append({
            "reference": f"that {candidates[0].title}" if that_match else candidates[0].title,
            "kind": "agent",
            "candidates": [root._agent_reference_payload(item) for item in candidates],
        })

    if "that agent" in normalized and "that agent" not in resolved:
        target_id = root._state.last_referenced_agent_id or root._state.last_created_agent_id
        if target_id and target_id in agents_by_id:
            resolved["that agent"] = root._agent_reference_payload(agents_by_id[target_id])
        else:
            missing.append({"reference": "that agent", "kind": "agent"})
    for alias in _AGENT_CREATED_REFERENCE_ALIASES:
        if alias not in normalized or alias in resolved:
            continue
        target_id = root._state.last_created_agent_id or root._state.last_referenced_agent_id
        if target_id and target_id in agents_by_id:
            resolved[alias] = root._agent_reference_payload(agents_by_id[target_id])
        else:
            missing.append({"reference": alias, "kind": "agent"})

    workflow_alias = "the workflow from earlier"
    if workflow_alias in normalized and workflow_alias not in resolved:
        workflow_id = root._state.last_referenced_workflow_id or root._state.last_created_workflow_id
        workflow = workflows_by_id.get(workflow_id or "")
        if workflow is not None:
            resolved[workflow_alias] = root._workflow_reference_payload(workflow)
        else:
            missing.append({"reference": workflow_alias, "kind": "workflow"})

    for alias, agent_id in root._state.reference_aliases.get("agents", {}).items():
        if (
            alias in normalized
            and alias not in resolved
            and agent_id in agents_by_id
            and root._agent_title_reference_kind(normalized, alias) == "single"
        ):
            resolved[alias] = root._agent_reference_payload(agents_by_id[agent_id])
    for alias, workflow_id in root._state.reference_aliases.get("workflows", {}).items():
        workflow = workflows_by_id.get(workflow_id)
        if alias in normalized and alias not in resolved and workflow is not None:
            resolved[alias] = root._workflow_reference_payload(workflow)

    # Generic "kill/terminate all agents" — fires when no title-specific bulk target was
    # matched but the text clearly asks to kill every agent (e.g. "kill all of them",
    # "kill all active agents", "kill all agents").
    if not bulk_targets and not resolved:
        _kill_all_agents_patterns = (
            r"\b(?:kill|terminate)\s+all\s+(?:of\s+(?:them|the\s+agents?)|agents?|(?:active|running|current)\s+agents?)\b",
            r"\b(?:kill|terminate)\s+all\s+of\s+them\b",
            r"\b(?:kill|terminate)\s+(?:all\s+)?(?:those|these|the)\s+agents?\b",
            r"\b(?:yes[,.]?\s+)?(?:kill|terminate)\s+(?:all\s+)?(?:them|agents?)\b",
        )
        child_agents = [a for a in visible_agents if a.agent_id != root._root_agent_id and a.status != "terminated"]
        if child_agents and any(re.search(p, normalized) for p in _kill_all_agents_patterns):
            bulk_targets.append({
                "reference": "all agents",
                "kind": "agent",
                "action": "kill",
                "candidates": [root._agent_reference_payload(a) for a in child_agents],
            })

    return {
        "resolved_references": resolved,
        "ambiguities": ambiguities,
        "missing": missing,
        "bulk_targets": bulk_targets,
    }

def format_reference_ambiguity(root, ambiguities: list[dict[str, Any]]) -> str:
    first = ambiguities[0]
    candidates = ", ".join(
        f"{item.get('title') or item.get('id')} ({item['id']})"
        for item in first.get("candidates", [])
    )
    return f"Ambiguous reference '{first['reference']}'. Clarify one of: {candidates}"

def format_bulk_agent_target_clarification(
    root,
    bulk_targets: list[dict[str, Any]],
) -> str:
    first = bulk_targets[0]
    candidates = first.get("candidates", [])
    candidate_list = ", ".join(
        f"{item.get('title') or item.get('id')} ({item['id']})"
        for item in candidates
    )
    action = first.get("action") or "manage"
    return (
        f"Bulk agent reference '{first.get('reference', 'agents')}' matches {len(candidates)} agent(s): "
        f"{candidate_list}. Clarify the exact agent_ids you want me to {action}."
    )

def resolved_workflow_target_id(root, resolved_references: dict[str, Any]) -> Optional[str]:
    for payload in resolved_references.values():
        if payload.get("kind") == "workflow":
            return payload.get("id")
    return None

def explicit_workflow_ids(
    root,
    user_input: str,
    resolved_references: dict[str, Any],
) -> list[str]:
    workflow_ids = {
        payload["id"]
        for payload in resolved_references.values()
        if payload.get("kind") == "workflow" and payload.get("id")
    }
    workflow_ids.update(_WORKFLOW_ID_PATTERN.findall(user_input))
    explicit_workflow_id = root._workflow_authoring.extract_workflow_id(user_input)
    if explicit_workflow_id:
        workflow_ids.add(explicit_workflow_id)
    return sorted(workflow_ids)

def explicit_task_ids(root, user_input: str) -> list[str]:
    return sorted(set(_TASK_ID_PATTERN.findall(user_input)))

def explicit_agent_ids(
    root,
    user_input: str,
    resolved_references: dict[str, Any],
) -> list[str]:
    agent_ids = set(_AGENT_ID_PATTERN.findall(user_input))
    for payload in resolved_references.values():
        if payload.get("kind") == "agent" and payload.get("id"):
            agent_ids.add(str(payload["id"]))
    return sorted(agent_ids)

