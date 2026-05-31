from __future__ import annotations

from mr1.event_log import SystemEvent

from mr1.tui.data import AgentTreeModel, preorder_agent_ids, visible_agent_ids


def visible_children(tree: AgentTreeModel, *, show_dead: bool) -> dict[str, tuple[str, ...]]:
    visible = visible_agent_ids(tree, show_dead=show_dead)
    return {
        parent_id: tuple(child_id for child_id in child_ids if child_id in visible)
        for parent_id, child_ids in tree.children_by_parent.items()
        if parent_id in visible
    }


def previous_sibling_agent_id(
    tree: AgentTreeModel,
    agent_id: str,
    *,
    show_dead: bool,
) -> str | None:
    agent = tree.nodes.get(agent_id)
    if agent is None:
        return None
    siblings = visible_children(tree, show_dead=show_dead).get(agent.parent_agent_id or "", ())
    if not siblings and agent.parent_agent_id in tree.children_by_parent:
        siblings = visible_children(tree, show_dead=show_dead).get(agent.parent_agent_id, ())
    if agent_id not in siblings:
        return None
    index = siblings.index(agent_id)
    if index <= 0:
        return None
    return siblings[index - 1]


def next_sibling_agent_id(
    tree: AgentTreeModel,
    agent_id: str,
    *,
    show_dead: bool,
) -> str | None:
    agent = tree.nodes.get(agent_id)
    if agent is None:
        return None
    siblings = visible_children(tree, show_dead=show_dead).get(agent.parent_agent_id, ())
    if agent_id not in siblings:
        return None
    index = siblings.index(agent_id)
    if index + 1 >= len(siblings):
        return None
    return siblings[index + 1]


def parent_agent_id(tree: AgentTreeModel, agent_id: str, *, show_dead: bool) -> str | None:
    visible = visible_agent_ids(tree, show_dead=show_dead)
    current = tree.nodes.get(agent_id)
    while current is not None and current.parent_agent_id is not None:
        parent_id = current.parent_agent_id
        if parent_id in visible:
            return parent_id
        current = tree.nodes.get(parent_id)
    return None


def first_child_agent_id(tree: AgentTreeModel, agent_id: str, *, show_dead: bool) -> str | None:
    return next(iter(visible_children(tree, show_dead=show_dead).get(agent_id, ())), None)


def coerce_selected_agent_id(
    tree: AgentTreeModel,
    selected_agent_id: str | None,
    *,
    show_dead: bool,
) -> str:
    visible = visible_agent_ids(tree, show_dead=show_dead)
    if selected_agent_id in visible:
        return selected_agent_id  # type: ignore[return-value]
    current_id = selected_agent_id
    while current_id:
        current = tree.nodes.get(current_id)
        if current is None:
            break
        parent_id = current.parent_agent_id
        if parent_id in visible:
            return parent_id  # type: ignore[return-value]
        current_id = parent_id
    ordered = preorder_agent_ids(tree, show_dead=show_dead)
    return ordered[0] if ordered else tree.root_agent_id


def coerce_selected_event_id(
    events: tuple[SystemEvent, ...],
    selected_event_id: str | None,
) -> str | None:
    if not events:
        return None
    event_ids = {event.event_id for event in events}
    if selected_event_id in event_ids:
        return selected_event_id
    return events[0].event_id


def older_event_id(events: tuple[SystemEvent, ...], selected_event_id: str | None) -> str | None:
    if not events:
        return None
    ids = [event.event_id for event in events]
    if selected_event_id not in ids:
        return ids[0]
    index = ids.index(selected_event_id)
    if index + 1 >= len(ids):
        return ids[index]
    return ids[index + 1]


def newer_event_id(events: tuple[SystemEvent, ...], selected_event_id: str | None) -> str | None:
    if not events:
        return None
    ids = [event.event_id for event in events]
    if selected_event_id not in ids:
        return ids[0]
    index = ids.index(selected_event_id)
    if index <= 0:
        return ids[index]
    return ids[index - 1]
