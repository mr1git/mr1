"""
Shared runtime observation helpers and typed accessors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mr1.capability_policy import CapabilityApprovalRequest, CapabilityApprovalStore
from mr1.event_log import EventLog, SystemEvent
from mr1.messages import MessageStore, PersistentMessage
from mr1.scoped_agents import AgentScopeError, PersistentAgent, PersistentAgentStore
from mr1.scheduler import WorkflowSpecError
from mr1.workflow_models import TaskStatus, Workflow
from mr1.workflow_store import WorkflowStore


_AGENT_PREVIEW_LIMIT = 240
_MESSAGE_PREVIEW_LIMIT = 4096
_WORKFLOW_SUMMARY_PREVIEW_LIMIT = 240
_RECENT_ERROR_STATUSES = {"failed", "blocked", "denied", "timed_out", "cancelled"}
_RECENT_ERROR_SEVERITIES = {"ERROR", "CRITICAL"}
_RECENT_ERROR_TYPES = {
    "workflow_task_failed",
    "workflow_failed",
    "capability_failed",
    "capability_blocked",
    "approval_denied",
}


def _preview_text(value: Optional[str], *, limit: int) -> tuple[str, bool, bool]:
    if not isinstance(value, str):
        return "-", False, False
    normalized = " ".join(value.split())
    if not normalized:
        return "-", False, False
    if len(normalized) <= limit:
        return normalized, False, True
    return normalized[:limit] + "...", True, True


def _serialize_last_action(action: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(action, dict):
        return None
    return json.dumps(action, sort_keys=True)


def _message_peer_preview(message: PersistentMessage) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "from_agent_id": message.from_agent_id,
        "to_agent_id": message.to_agent_id,
        "kind": message.kind,
        "subject": message.subject,
        "created_at": message.created_at,
        "status": message.status,
    }


def _pending_parent_messages(
    agent: PersistentAgent,
    message_store: MessageStore,
) -> list[PersistentMessage]:
    if not agent.parent_agent_id:
        return []
    action_name = (agent.last_action or {}).get("action")
    if agent.run_status != "waiting" and action_name not in {"ask_parent", "send_message"}:
        return []
    return [
        item
        for item in message_store.list_outbox(agent.agent_id)
        if item.to_agent_id == agent.parent_agent_id
        and item.kind in {"question", "request"}
    ]


def _agent_runtime_activity_payload(
    agent: PersistentAgent,
    workflow_store: WorkflowStore,
) -> dict[str, Any]:
    active_jobs = 0
    active_workflows = 0
    has_running_processes = False
    for workflow_id in agent.owned_workflow_ids:
        workflow = workflow_store.load_workflow(workflow_id)
        if workflow is None:
            continue
        workflow_is_active = False
        for task in workflow.tasks.values():
            if not task.is_terminal():
                workflow_is_active = True
            if task.status == TaskStatus.RUNNING:
                active_jobs += 1
                if task.pid is not None:
                    has_running_processes = True
        if workflow_is_active:
            active_workflows += 1
    runtime_activity = (
        f"{active_jobs} active job(s)"
        if active_jobs
        else "no active jobs"
    )
    return {
        "active_jobs": active_jobs,
        "active_workflows": active_workflows,
        "has_running_processes": has_running_processes,
        "runtime_activity": runtime_activity,
    }


def _normalize_caller_agent_id(
    scoped_agents: PersistentAgentStore,
    caller_agent_id: Optional[str],
) -> str:
    return caller_agent_id or scoped_agents.root_agent_id


def _prioritize_items(
    items: list[dict[str, Any]],
    *,
    id_field: str,
    pinned_ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    pinned = [item for item in list(pinned_ids or []) if item]
    pinned_set = set(pinned)
    prioritized = [item for item in items if item.get(id_field) in pinned_set]
    remaining = [item for item in items if item.get(id_field) not in pinned_set]
    combined = prioritized + remaining
    if limit is None:
        return combined
    return combined[:limit]


def visible_workflows(
    store: WorkflowStore,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> list[Workflow]:
    workflows = []
    for workflow in store.list_workflows():
        workflow = scoped_agents.normalize_workflow_ownership(workflow)
        if scoped_agents.can_agent_access_workflow(caller_agent_id, workflow):
            workflows.append(workflow)
    return workflows


def load_scoped_workflow(
    store: WorkflowStore,
    workflow_id: str,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> Workflow:
    workflow = store.load_workflow(workflow_id)
    if workflow is None:
        raise WorkflowSpecError(f"workflow not found: {workflow_id}")
    workflow = scoped_agents.normalize_workflow_ownership(workflow)
    if not scoped_agents.can_agent_access_workflow(caller_agent_id, workflow):
        raise WorkflowSpecError("access denied: workflow not in agent scope")
    return workflow


def require_visible_message(
    message_store: MessageStore,
    message_id: str,
    caller_agent_id: str,
) -> PersistentMessage:
    message = message_store.get_message(message_id)
    if message is None:
        raise ValueError(f"message not found: {message_id}")
    if not message_store.can_agent_access_message(caller_agent_id, message):
        raise AgentScopeError("access denied: message not in agent scope")
    return message


def visible_approvals(
    approval_store: CapabilityApprovalStore,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> list[CapabilityApprovalRequest]:
    visible_ids = {agent.agent_id for agent in scoped_agents.list_visible_agents(caller_agent_id)}
    if scoped_agents.is_root_agent(caller_agent_id):
        visible_ids = {agent.agent_id for agent in scoped_agents.list_agents()}
    approvals = []
    for approval in approval_store.list_requests():
        if (
            approval.requesting_actor_id in visible_ids
            or (approval.designated_approver_id or "") in visible_ids
        ):
            approvals.append(approval)
    return approvals


def require_visible_approval(
    approval_store: CapabilityApprovalStore,
    approval_request_id: str,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> CapabilityApprovalRequest:
    approval = approval_store.require(approval_request_id)
    visible = {
        item.approval_request_id
        for item in visible_approvals(approval_store, scoped_agents, caller_agent_id)
    }
    if approval.approval_request_id not in visible:
        raise AgentScopeError("access denied: approval not in agent scope")
    return approval


def event_visible(
    event: SystemEvent,
    *,
    store: WorkflowStore,
    scoped_agents: PersistentAgentStore,
    message_store: MessageStore,
    caller_agent_id: str,
    approval_store: Optional[CapabilityApprovalStore] = None,
) -> bool:
    if scoped_agents.is_root_agent(caller_agent_id):
        return True
    if event.workflow_id is not None:
        try:
            load_scoped_workflow(store, event.workflow_id, scoped_agents, caller_agent_id)
            return True
        except WorkflowSpecError:
            return False
    if event.approval_request_id is not None:
        try:
            require_visible_approval(
                approval_store or CapabilityApprovalStore(store.root.parent / "capability_approvals"),
                event.approval_request_id,
                scoped_agents,
                caller_agent_id,
            )
            return True
        except (ValueError, AgentScopeError):
            return False
    if event.message_id is not None:
        try:
            require_visible_message(message_store, event.message_id, caller_agent_id)
            return True
        except (ValueError, AgentScopeError):
            return False
    if event.target_type == "agent" and event.target_id is not None:
        return scoped_agents.is_visible(caller_agent_id, event.target_id)
    if event.actor_id is not None and scoped_agents.load_agent(event.actor_id) is not None:
        return scoped_agents.is_visible(caller_agent_id, event.actor_id)
    return True


def visible_timeline_events(
    store: WorkflowStore,
    scoped_agents: PersistentAgentStore,
    message_store: MessageStore,
    caller_agent_id: str,
    *,
    event_log: Optional[EventLog] = None,
    approval_store: Optional[CapabilityApprovalStore] = None,
) -> list[SystemEvent]:
    events = (event_log or EventLog(store.root.parent / "events")).list_events()
    return [
        event
        for event in events
        if event_visible(
            event,
            store=store,
            scoped_agents=scoped_agents,
            message_store=message_store,
            caller_agent_id=caller_agent_id,
            approval_store=approval_store,
        )
    ]


class RuntimeAccess:
    def __init__(
        self,
        *,
        workflow_store: WorkflowStore,
        scoped_agent_store: PersistentAgentStore,
        message_store: MessageStore,
        approval_store: Optional[CapabilityApprovalStore] = None,
        event_log: Optional[EventLog] = None,
    ):
        self._workflow_store = workflow_store
        self._scoped_agents = scoped_agent_store
        self._message_store = message_store
        self._approval_store = approval_store or CapabilityApprovalStore(
            workflow_store.root.parent / "capability_approvals"
        )
        self._event_log = event_log or EventLog(workflow_store.root.parent / "events")

    def _caller(self, caller_agent_id: Optional[str]) -> str:
        return _normalize_caller_agent_id(self._scoped_agents, caller_agent_id)

    def _agent_preview_payload(self, agent: PersistentAgent) -> dict[str, Any]:
        mission_preview, mission_truncated, mission_full_available = _preview_text(
            agent.mission,
            limit=_AGENT_PREVIEW_LIMIT,
        )
        parent_preview, parent_truncated, parent_full_available = _preview_text(
            agent.parent_request,
            limit=_AGENT_PREVIEW_LIMIT,
        )
        last_action_preview, last_action_truncated, last_action_full_available = _preview_text(
            _serialize_last_action(agent.last_action),
            limit=_AGENT_PREVIEW_LIMIT,
        )
        unread_count = sum(
            1
            for message in self._message_store.list_inbox(agent.agent_id)
            if message.status == "unread"
        )
        inbox = self._message_store.list_inbox(agent.agent_id)
        outbox = self._message_store.list_outbox(agent.agent_id)
        pending_parent = _pending_parent_messages(agent, self._message_store)
        return {
            "agent_id": agent.agent_id,
            "title": agent.title,
            "agent_type": agent.agent_type,
            "status": agent.status,
            "run_status": agent.run_status,
            "mission_preview": mission_preview,
            "mission_truncated": mission_truncated,
            "mission_full_available": mission_full_available,
            "parent_request_preview": parent_preview,
            "parent_request_truncated": parent_truncated,
            "parent_request_full_available": parent_full_available,
            "last_action_preview": last_action_preview,
            "last_action_truncated": last_action_truncated,
            "last_action_full_available": last_action_full_available,
            "latest_run": dict(agent.last_run) if isinstance(agent.last_run, dict) else None,
            "unread_inbox_count": unread_count,
            "parent_agent_id": agent.parent_agent_id,
            "scope_roots_summary": list(agent.scope_roots or [])[:5],
            "created_at": agent.created_at,
            "latest_inbox_messages": [_message_peer_preview(item) for item in inbox[:3]],
            "latest_outbox_messages": [_message_peer_preview(item) for item in outbox[:3]],
            "pending_parent_messages": [_message_peer_preview(item) for item in pending_parent[:3]],
            "pending_parent_questions": len(pending_parent),
        }

    def _workflow_preview_payload(self, workflow: Workflow) -> dict[str, Any]:
        tasks = list(workflow.tasks.values())
        tasks.sort(key=lambda item: item.created_at)
        recent_task_status_summary = []
        for task in tasks[:5]:
            summary_preview, summary_truncated, summary_full_available = _preview_text(
                task.result_summary or task.last_error or task.blocked_reason,
                limit=_WORKFLOW_SUMMARY_PREVIEW_LIMIT,
            )
            recent_task_status_summary.append({
                "task_id": task.task_id,
                "label": task.label,
                "status": task.status.value,
                "summary_preview": summary_preview,
                "summary_truncated": summary_truncated,
                "summary_full_available": summary_full_available,
            })
        return {
            "workflow_id": workflow.workflow_id,
            "title": workflow.title,
            "status": workflow.status.value,
            "owner_agent_id": workflow.owner_agent_id,
            "owner_agent_title": workflow.owner_agent_title,
            "parent_agent_id": workflow.parent_agent_id,
            "created_at": workflow.created_at,
            "finished_at": workflow.finished_at,
            "recent_task_status_summary": recent_task_status_summary,
            "memory_refs_used": list(workflow.metadata.get("memory_refs_used", []))
            if isinstance(workflow.metadata, dict) else [],
        }

    def _message_preview_payload(self, message: PersistentMessage) -> dict[str, Any]:
        body_preview, body_truncated, body_full_available = _preview_text(
            message.body,
            limit=_MESSAGE_PREVIEW_LIMIT,
        )
        return {
            "message_id": message.message_id,
            "from_agent_id": message.from_agent_id,
            "to_agent_id": message.to_agent_id,
            "kind": message.kind,
            "subject": message.subject,
            "status": message.status,
            "created_at": message.created_at,
            "workflow_id": message.workflow_id,
            "task_id": message.task_id,
            "body_preview": body_preview,
            "body_truncated": body_truncated,
            "body_full_available": body_full_available,
        }

    def _approval_preview_payload(self, approval: CapabilityApprovalRequest) -> dict[str, Any]:
        return {
            "approval_request_id": approval.approval_request_id,
            "requesting_actor_id": approval.requesting_actor_id,
            "capability_name": approval.capability_name,
            "risk_score": approval.risk_score,
            "designated_approver_id": approval.designated_approver_id,
            "status": approval.status,
            "created_at": approval.created_at,
            "workflow_id": approval.workflow_id,
            "task_id": approval.task_id,
            "message_id": approval.message_id,
            "requested_scope_path": approval.requested_scope_path,
        }

    def list_agents(
        self,
        *,
        caller_agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        pinned_agent_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        caller = self._caller(caller_agent_id)
        payload = [
            self._agent_preview_payload(agent)
            for agent in self._scoped_agents.list_visible_agents(caller)
        ]
        payload.sort(key=lambda item: (item["created_at"], item["agent_id"]), reverse=True)
        return _prioritize_items(
            payload,
            id_field="agent_id",
            pinned_ids=pinned_agent_ids,
            limit=limit,
        )

    def read_agent(
        self,
        agent_id: str,
        *,
        full: bool = True,
        caller_agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        caller = self._caller(caller_agent_id)
        agent = self._scoped_agents.get_visible_agent(caller, agent_id)
        if not full:
            return self._agent_preview_payload(agent)
        payload = agent.to_dict()
        payload.update(self._agent_runtime_activity_payload(agent))
        last_run = agent.last_run or {}
        payload["latest_run_id"] = last_run.get("run_id")
        payload["latest_run_stopped_reason"] = last_run.get("stopped_reason")
        payload["latest_run_step_count"] = last_run.get("step_count")
        payload["latest_run_at"] = last_run.get("finished_at")
        payload["reports"] = [str(path) for path in self._scoped_agents.list_reports(agent.agent_id)]
        payload["latest_reports"] = [Path(path).name for path in payload["reports"][:5]]
        inbox = self._message_store.list_inbox(agent.agent_id)
        outbox = self._message_store.list_outbox(agent.agent_id)
        pending_parent_messages = _pending_parent_messages(agent, self._message_store)
        waiting_on_parent = bool(pending_parent_messages)
        if agent.parent_agent_id and agent.run_status == "waiting":
            action_name = (agent.last_action or {}).get("action")
            waiting_on_parent = waiting_on_parent or action_name in {"ask_parent", "send_message"}
        payload["unread_inbox_count"] = sum(1 for item in inbox if item.status == "unread")
        payload["latest_inbox_messages"] = [_message_peer_preview(item) for item in inbox[:3]]
        payload["latest_outbox_messages"] = [_message_peer_preview(item) for item in outbox[:3]]
        payload["waiting_on_parent"] = waiting_on_parent
        payload["pending_parent_questions"] = len(pending_parent_messages)
        payload["pending_parent_messages"] = [
            _message_peer_preview(item)
            for item in pending_parent_messages[:3]
        ]
        return payload

    def _agent_runtime_activity_payload(self, agent: PersistentAgent) -> dict[str, Any]:
        return _agent_runtime_activity_payload(agent, self._workflow_store)

    def list_messages(
        self,
        *,
        caller_agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        pinned_message_ids: Optional[list[str]] = None,
        message_ids: Optional[list[str]] = None,
        to_agent_id: Optional[str] = None,
        from_agent_id: Optional[str] = None,
        status: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        caller = self._caller(caller_agent_id)
        id_filter = {item for item in list(message_ids or []) if item}
        messages = []
        for message in self._message_store.list_messages():
            if not self._message_store.can_agent_access_message(caller, message):
                continue
            if id_filter and message.message_id not in id_filter:
                continue
            if to_agent_id is not None and message.to_agent_id != to_agent_id:
                continue
            if from_agent_id is not None and message.from_agent_id != from_agent_id:
                continue
            if status is not None and message.status != status:
                continue
            if not include_archived and message.status == "archived":
                continue
            messages.append(self._message_preview_payload(message))
        messages.sort(key=lambda item: (item["created_at"], item["message_id"]), reverse=True)
        return _prioritize_items(
            messages,
            id_field="message_id",
            pinned_ids=pinned_message_ids or list(message_ids or []),
            limit=limit,
        )

    def read_message(
        self,
        message_id: str,
        *,
        full: bool = True,
        caller_agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        caller = self._caller(caller_agent_id)
        message = require_visible_message(self._message_store, message_id, caller)
        if not full:
            return self._message_preview_payload(message)
        payload = message.to_dict()
        payload["body_truncated"] = False
        payload["body_full_available"] = bool(message.body)
        return payload

    def list_workflows(
        self,
        *,
        caller_agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        pinned_workflow_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        caller = self._caller(caller_agent_id)
        payload = [
            self._workflow_preview_payload(workflow)
            for workflow in visible_workflows(
                self._workflow_store,
                self._scoped_agents,
                caller,
            )
        ]
        payload.sort(key=lambda item: (item["created_at"], item["workflow_id"]), reverse=True)
        return _prioritize_items(
            payload,
            id_field="workflow_id",
            pinned_ids=pinned_workflow_ids,
            limit=limit,
        )

    def read_workflow(
        self,
        workflow_id: str,
        *,
        full: bool = True,
        caller_agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        caller = self._caller(caller_agent_id)
        workflow = load_scoped_workflow(
            self._workflow_store,
            workflow_id,
            self._scoped_agents,
            caller,
        )
        if not full:
            return self._workflow_preview_payload(workflow)
        payload = workflow.to_dict()
        task_details: dict[str, Any] = {}
        for label, task_id in workflow.label_to_task_id.items():
            task = workflow.tasks.get(task_id)
            if task is None:
                continue
            output_payload = self._workflow_store.load_task_output(workflow.workflow_id, task_id)
            inputs_payload = self._workflow_store.load_task_inputs(workflow.workflow_id, task_id)
            task_details[task_id] = {
                "label": label,
                "task": task.to_dict(),
                "result_payload": self._workflow_store.read_result(workflow.workflow_id, task_id),
                "output_payload": output_payload.to_dict() if output_payload is not None else None,
                "inputs_payload": [
                    item.to_dict() for item in inputs_payload
                ] if inputs_payload is not None else None,
            }
        payload["task_details"] = task_details
        return payload

    def list_pending_approvals(
        self,
        *,
        caller_agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        caller = self._caller(caller_agent_id)
        payload = [
            self._approval_preview_payload(approval)
            for approval in visible_approvals(
                self._approval_store,
                self._scoped_agents,
                caller,
            )
            if approval.status == "pending"
        ]
        payload.sort(key=lambda item: (item["created_at"], item["approval_request_id"]), reverse=True)
        if limit is not None:
            return payload[:limit]
        return payload

    def list_recent_events(
        self,
        *,
        caller_agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        caller = self._caller(caller_agent_id)
        events = visible_timeline_events(
            self._workflow_store,
            self._scoped_agents,
            self._message_store,
            caller,
            event_log=self._event_log,
            approval_store=self._approval_store,
        )
        events.sort(key=lambda item: item.event_index, reverse=True)
        return [
            {
                "event_index": event.event_index,
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "status": event.status,
                "severity": event.severity,
                "summary": event.summary,
                "actor_id": event.actor_id,
                "target_id": event.target_id,
                "workflow_id": event.workflow_id,
                "task_id": event.task_id,
                "message_id": event.message_id,
                "approval_request_id": event.approval_request_id,
                "record_path": event.record_path,
            }
            for event in events[:limit]
        ]

    def list_recent_errors(
        self,
        *,
        caller_agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        caller = self._caller(caller_agent_id)
        events = visible_timeline_events(
            self._workflow_store,
            self._scoped_agents,
            self._message_store,
            caller,
            event_log=self._event_log,
            approval_store=self._approval_store,
        )
        error_events = [
            event
            for event in events
            if (
                event.event_type in _RECENT_ERROR_TYPES
                or event.severity in _RECENT_ERROR_SEVERITIES
                or str(event.status).lower() in _RECENT_ERROR_STATUSES
            )
        ]
        error_events.sort(key=lambda item: item.event_index, reverse=True)
        return [
            {
                "event_index": event.event_index,
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "status": event.status,
                "severity": event.severity,
                "summary": event.summary,
                "actor_id": event.actor_id,
                "target_id": event.target_id,
                "workflow_id": event.workflow_id,
                "task_id": event.task_id,
                "message_id": event.message_id,
                "approval_request_id": event.approval_request_id,
                "record_path": event.record_path,
            }
            for event in error_events[:limit]
        ]

    def search_memory(
        self,
        *,
        query: str,
        limit: int = 5,
        caller_agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        from mr1.memory_queries import memory_search

        del caller_agent_id
        safe_limit = max(1, min(int(limit), 10))
        memory_root = self._workflow_store.root.parent
        try:
            raw = memory_search(memory_root, query=query, limit=safe_limit)
        except Exception:
            return {"query": query, "results": []}
        results = []
        for item in raw.get("items", []):
            refs = item.get("refs") or []
            ref_id = refs[0].get("id") if refs else None
            results.append({
                "source": item.get("doc_type"),
                "id": item.get("doc_id"),
                "summary": item.get("summary") or item.get("title"),
                "score": item.get("score"),
                "ref": ref_id,
                "full_available": True,
            })
        return {"query": raw.get("query", query), "results": results}
