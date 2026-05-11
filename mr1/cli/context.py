"""CLI context: workflow-store/visibility/audit helpers.

Most CLI sub-command handlers need: scoped workflow access, approval/
timeline visibility filtering, capability-audit log lookup, and helpers
that construct sibling stores (memory graph, insights, event log,
approval store) from a `WorkflowStore` root. This module collects them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mr1.capability_policy import (
    CapabilityApprovalRequest,
    CapabilityApprovalStore,
)
from mr1.event_log import EventLog, SystemEvent
from mr1.memory_curator import InsightStore
from mr1.memory_graph import MemoryGraph, MemoryGraphStore
from mr1.messages import MessageStore, PersistentMessage
from mr1.runtime_access import (
    RuntimeAccess,
    load_scoped_workflow as _load_scoped_workflow_shared,
    require_visible_approval as _require_visible_approval_shared,
    require_visible_message as _require_message_shared,
    visible_approvals as _visible_approvals_shared,
    visible_timeline_events as _visible_timeline_events_shared,
    visible_workflows as _visible_workflows_shared,
)
from mr1.scheduler import WorkflowSpecError
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore
from mr1.workflow_models import Task, Workflow
from mr1.workflow_store import WorkflowStore


def _find_workflow_for_task(
    store: WorkflowStore,
    task_id: str,
    *,
    workflows: Optional[list[Workflow]] = None,
) -> tuple[Optional[Workflow], Optional[Task]]:
    for wf in workflows if workflows is not None else store.list_workflows():
        task = wf.tasks.get(task_id)
        if task is not None:
            return wf, task
    return None, None


def _visible_workflows(
    store: WorkflowStore,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> list[Workflow]:
    return _visible_workflows_shared(
        store,
        scoped_agents,
        caller_agent_id,
    )


def _load_scoped_workflow(
    store: WorkflowStore,
    workflow_id: str,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> Workflow:
    return _load_scoped_workflow_shared(
        store,
        workflow_id,
        scoped_agents,
        caller_agent_id,
    )


def _load_json_file(path_str: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    path = Path(path_str)
    if not path.exists():
        return None, f"spec file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "workflow JSON must be an object"
    return payload, None


def _load_text_file(path_str: str) -> tuple[Optional[str], Optional[str]]:
    path = Path(path_str)
    if not path.exists():
        return None, f"request file not found: {path}"
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, str(exc)
    if not payload.strip():
        return None, "request file must not be empty"
    return payload, None


def _approval_store_for(store: WorkflowStore) -> CapabilityApprovalStore:
    return CapabilityApprovalStore(store.root.parent / "capability_approvals")


def _timeline_for(store: WorkflowStore) -> EventLog:
    return EventLog(store.root.parent / "events")


def _graph_store_for(store: WorkflowStore) -> MemoryGraphStore:
    return MemoryGraphStore(store.root.parent / "graph")


def _insight_store_for(store: WorkflowStore) -> InsightStore:
    return InsightStore(store.root.parent / "insights")


def _runtime_root_for(store: WorkflowStore) -> Path:
    return store.root.parent


def _load_memory_graph(store: WorkflowStore) -> tuple[MemoryGraph, int]:
    graph_store = _graph_store_for(store)
    return graph_store.load_graph(), graph_store.load_cursor()


def _visible_approvals(
    approval_store: CapabilityApprovalStore,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> list[CapabilityApprovalRequest]:
    return _visible_approvals_shared(
        approval_store,
        scoped_agents,
        caller_agent_id,
    )


def _resolve_mailbox_agent_id(
    target_agent_id: Optional[str],
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> str:
    resolved = target_agent_id or caller_agent_id
    if scoped_agents.load_agent(resolved) is None:
        raise ValueError(f"agent not found: {resolved}")
    if scoped_agents.is_root_agent(caller_agent_id):
        return resolved
    if resolved != caller_agent_id:
        raise AgentScopeError("access denied: message not in agent scope")
    return resolved


def _require_message(
    message_store: MessageStore,
    message_id: str,
    caller_agent_id: str,
) -> PersistentMessage:
    return _require_message_shared(
        message_store,
        message_id,
        caller_agent_id,
    )


def _event_visible(
    event: SystemEvent,
    *,
    store: WorkflowStore,
    scoped_agents: PersistentAgentStore,
    message_store: MessageStore,
    caller_agent_id: str,
) -> bool:
    if scoped_agents.is_root_agent(caller_agent_id):
        return True
    if event.workflow_id is not None:
        try:
            _load_scoped_workflow(store, event.workflow_id, scoped_agents, caller_agent_id)
            return True
        except WorkflowSpecError:
            return False
    if event.approval_request_id is not None:
        try:
            _require_visible_approval(
                _approval_store_for(store),
                event.approval_request_id,
                scoped_agents,
                caller_agent_id,
            )
            return True
        except (ValueError, AgentScopeError):
            return False
    if event.message_id is not None:
        try:
            _require_message(message_store, event.message_id, caller_agent_id)
            return True
        except (ValueError, AgentScopeError):
            return False
    if event.target_type == "agent" and event.target_id is not None:
        return scoped_agents.is_visible(caller_agent_id, event.target_id)
    if event.actor_id is not None and scoped_agents.load_agent(event.actor_id) is not None:
        return scoped_agents.is_visible(caller_agent_id, event.actor_id)
    return True


def _visible_timeline_events(
    store: WorkflowStore,
    scoped_agents: PersistentAgentStore,
    message_store: MessageStore,
    caller_agent_id: str,
) -> list[SystemEvent]:
    return _visible_timeline_events_shared(
        store,
        scoped_agents,
        message_store,
        caller_agent_id,
        event_log=_timeline_for(store),
        approval_store=_approval_store_for(store),
    )


def _require_visible_approval(
    approval_store: CapabilityApprovalStore,
    approval_request_id: str,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> CapabilityApprovalRequest:
    return _require_visible_approval_shared(
        approval_store,
        approval_request_id,
        scoped_agents,
        caller_agent_id,
    )


def _runtime_access_for(
    store: WorkflowStore,
    scoped_agents: PersistentAgentStore,
    message_store: MessageStore,
    *,
    approval_store: Optional[CapabilityApprovalStore] = None,
) -> RuntimeAccess:
    return RuntimeAccess(
        workflow_store=store,
        scoped_agent_store=scoped_agents,
        message_store=message_store,
        approval_store=approval_store or _approval_store_for(store),
        event_log=_timeline_for(store),
    )


def _audit_entries_for_agent(agent_store: PersistentAgentStore, agent_id: str) -> list[dict[str, Any]]:
    path = agent_store.capability_call_log_path(agent_id)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _visible_audit_entries(
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
    target_agent_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    agent_ids: list[str]
    if target_agent_id is not None:
        scoped_agents.get_visible_agent(caller_agent_id, target_agent_id)
        agent_ids = [target_agent_id]
    else:
        agent_ids = [agent.agent_id for agent in scoped_agents.list_visible_agents(caller_agent_id)]
    entries: list[dict[str, Any]] = []
    for agent_id in agent_ids:
        entries.extend(_audit_entries_for_agent(scoped_agents, agent_id))
    entries.sort(key=lambda item: (str(item.get("audit_id")), str(item.get("audit_path"))), reverse=True)
    return entries


def _find_visible_audit_entry(
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
    audit_id: str,
) -> dict[str, Any]:
    for item in _visible_audit_entries(scoped_agents, caller_agent_id):
        if item.get("audit_id") == audit_id:
            return item
    raise ValueError(f"capability audit not found: {audit_id}")
