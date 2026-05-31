from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mr1.capability_policy import CapabilityApprovalStore
from mr1.event_log import EventLog, SystemEvent
from mr1.messages import MessageStore
from mr1.runtime_access import RuntimeAccess
from mr1.scoped_agents import (
    PersistentAgent,
    PersistentAgentStore,
    is_agent_live as scoped_agent_is_live,
    is_agent_terminal as scoped_agent_is_terminal,
)
from mr1.workflow_store import WorkflowStore


@dataclass(frozen=True)
class AgentTreeModel:
    root_agent_id: str
    nodes: dict[str, PersistentAgent]
    children_by_parent: dict[str, tuple[str, ...]]
    previews: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSnapshot:
    tree: AgentTreeModel
    events: tuple[SystemEvent, ...]
    refreshed_at: str

    @property
    def live_count(self) -> int:
        return sum(1 for agent in self.tree.nodes.values() if agent_is_live(agent))

    @property
    def done_count(self) -> int:
        return sum(1 for agent in self.tree.nodes.values() if agent_is_terminal(agent))


def agent_is_terminal(agent: PersistentAgent) -> bool:
    return scoped_agent_is_terminal(agent)


def agent_is_live(agent: PersistentAgent) -> bool:
    return scoped_agent_is_live(agent)


def build_agent_tree(
    agents: list[PersistentAgent],
    *,
    root_agent_id: str,
    previews: Optional[dict[str, dict[str, Any]]] = None,
) -> AgentTreeModel:
    nodes = {agent.agent_id: agent for agent in agents}
    if root_agent_id not in nodes:
        raise ValueError(f"root agent missing from tree: {root_agent_id}")

    mutable_children: dict[str, list[str]] = {agent_id: [] for agent_id in nodes}
    mutable_children.setdefault(root_agent_id, [])
    for agent in agents:
        if agent.agent_id == root_agent_id:
            continue
        parent_id = agent.parent_agent_id
        if parent_id not in nodes:
            parent_id = root_agent_id
        mutable_children.setdefault(parent_id, []).append(agent.agent_id)

    for parent_id, child_ids in mutable_children.items():
        child_ids.sort(key=lambda item: (nodes[item].created_at, item))

    children_by_parent = {
        parent_id: tuple(child_ids)
        for parent_id, child_ids in mutable_children.items()
    }
    return AgentTreeModel(
        root_agent_id=root_agent_id,
        nodes=nodes,
        children_by_parent=children_by_parent,
        previews=dict(previews or {}),
    )


def visible_agent_ids(tree: AgentTreeModel, *, show_dead: bool) -> set[str]:
    if show_dead:
        return set(tree.nodes)

    visible: set[str] = set()

    def visit(agent_id: str) -> bool:
        agent = tree.nodes[agent_id]
        child_visible = any(visit(child_id) for child_id in tree.children_by_parent.get(agent_id, ()))
        keep_self = (
            agent_id == tree.root_agent_id
            or agent_is_live(agent)
            or child_visible
        )
        if keep_self:
            visible.add(agent_id)
        return keep_self

    visit(tree.root_agent_id)
    return visible


def preorder_agent_ids(tree: AgentTreeModel, *, show_dead: bool) -> list[str]:
    visible = visible_agent_ids(tree, show_dead=show_dead)
    ordered: list[str] = []

    def walk(agent_id: str) -> None:
        if agent_id not in visible:
            return
        ordered.append(agent_id)
        for child_id in tree.children_by_parent.get(agent_id, ()):
            walk(child_id)

    walk(tree.root_agent_id)
    return ordered


def live_focus_agent_id(tree: AgentTreeModel, *, show_dead: bool) -> str:
    visible = visible_agent_ids(tree, show_dead=show_dead)
    live_agents = [
        agent
        for agent_id, agent in tree.nodes.items()
        if agent_id in visible and agent_is_live(agent)
    ]
    if not live_agents:
        return tree.root_agent_id
    live_agents.sort(
        key=lambda agent: (agent.tree_level, agent.created_at, agent.agent_id),
    )
    return live_agents[-1].agent_id


def compact_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except TypeError:
        return json.dumps(str(payload))


class RuntimeDataSource:
    def __init__(self, *, store_root: Optional[Path] = None):
        self.workflow_store = WorkflowStore(root=store_root)
        runtime_root = self.workflow_store.root.parent
        self.agent_store = PersistentAgentStore(root=runtime_root / "agents")
        self.message_store = MessageStore(
            root=runtime_root / "messages",
            scoped_agent_store=self.agent_store,
        )
        self.approval_store = CapabilityApprovalStore(runtime_root / "capability_approvals")
        self.event_log = EventLog(runtime_root / "events")
        self.runtime_access = RuntimeAccess(
            workflow_store=self.workflow_store,
            scoped_agent_store=self.agent_store,
            message_store=self.message_store,
            approval_store=self.approval_store,
            event_log=self.event_log,
        )
        self.root_agent_id = self.agent_store.root_agent_id

    def load_snapshot(self, *, event_limit: int = 200) -> RuntimeSnapshot:
        agents = self.agent_store.list_agents()
        root_agent = self.agent_store.ensure_root_agent()
        if not any(agent.agent_id == root_agent.agent_id for agent in agents):
            agents = [root_agent, *agents]
        previews = {
            agent.agent_id: self.runtime_access.read_agent(
                agent.agent_id,
                full=False,
                caller_agent_id=self.root_agent_id,
            )
            for agent in agents
        }
        tree = build_agent_tree(agents, root_agent_id=self.root_agent_id, previews=previews)
        recent_events = tuple(
            sorted(
                self.event_log.list_events(limit=event_limit),
                key=lambda event: (event.event_index, event.event_id),
                reverse=True,
            )
        )
        return RuntimeSnapshot(
            tree=tree,
            events=recent_events,
            refreshed_at=root_agent.created_at if not recent_events else recent_events[0].timestamp,
        )

    def agent_detail(self, agent_id: str) -> dict[str, Any]:
        return self.runtime_access.read_agent(
            agent_id,
            full=True,
            caller_agent_id=self.root_agent_id,
        )

    def event_detail(self, event_id: str) -> dict[str, Any]:
        event = self.event_log.get_event(event_id)
        if event is None:
            raise ValueError(f"event not found: {event_id}")
        detail: dict[str, Any] = {
            "event": event.to_dict(),
            "payload_text": compact_payload(event.metadata),
        }
        if event.workflow_id:
            try:
                detail["workflow"] = self.runtime_access.read_workflow(
                    event.workflow_id,
                    full=False,
                    caller_agent_id=self.root_agent_id,
                )
            except Exception:
                detail["workflow"] = None
        if event.message_id:
            try:
                detail["message"] = self.runtime_access.read_message(
                    event.message_id,
                    full=False,
                    caller_agent_id=self.root_agent_id,
                )
            except Exception:
                detail["message"] = None
        if event.approval_request_id:
            try:
                detail["approval"] = self.approval_store.require(event.approval_request_id).to_dict()
            except Exception:
                detail["approval"] = None
        return detail
