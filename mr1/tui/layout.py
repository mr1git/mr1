from __future__ import annotations

import io
from typing import Any, Iterable

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mr1.scoped_agents import agent_display_status
from mr1.tui.colors import color_for_agent, severity_style
from mr1.tui.data import AgentTreeModel, RuntimeSnapshot, preorder_agent_ids
from mr1.tui.navigation import visible_children
from mr1.tui.state import (
    DETAIL_KIND_AGENT,
    DETAIL_KIND_EVENT,
    TUI_MODE_DETAIL,
    TUI_MODE_TIMELINE,
    TuiState,
)


def _truncate(value: str | None, limit: int = 80) -> str:
    if not value:
        return "-"
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _crop_lines(lines: list[tuple[str | None, Text]], selected_id: str | None, height: int) -> list[Text]:
    if height <= 0:
        return []
    if len(lines) <= height:
        return [item for _, item in lines]
    selected_index = 0
    for index, (line_id, _) in enumerate(lines):
        if line_id == selected_id:
            selected_index = index
            break
    start = max(0, selected_index - (height // 2))
    end = min(len(lines), start + height)
    start = max(0, end - height)
    return [item for _, item in lines[start:end]]


def render_tree_lines(
    tree: AgentTreeModel,
    *,
    selected_agent_id: str | None,
    show_dead: bool,
    max_lines: int,
) -> list[Text]:
    children = visible_children(tree, show_dead=show_dead)
    lines: list[tuple[str | None, Text]] = []

    def walk(agent_id: str, prefix: tuple[bool, ...]) -> None:
        agent = tree.nodes[agent_id]
        preview = tree.previews.get(agent_id, {})
        lifecycle_status = str(preview.get("lifecycle_status") or agent_display_status(agent))
        label = Text()
        marker = "▶ " if agent_id == selected_agent_id else "  "
        label.append(marker, style="bold white" if agent_id == selected_agent_id else "dim")
        for index, has_more in enumerate(prefix):
            is_last_prefix = index == len(prefix) - 1
            if is_last_prefix:
                label.append("├─ " if has_more else "└─ ", style="dim")
            else:
                label.append("│  " if has_more else "   ", style="dim")
        style = color_for_agent(
            depth=agent.mr_level,
            status=agent.status,
            run_status=agent.run_status,
            selected=agent_id == selected_agent_id,
        ).style
        label.append(agent.title, style=style)
        label.append(f" [{agent.agent_id}]", style="dim")
        label.append(f"  {lifecycle_status}/{agent.run_status}", style="dim")
        if preview.get("unread_inbox_count"):
            label.append(f"  inbox:{preview['unread_inbox_count']}", style="yellow")
        lines.append((agent_id, label))
        child_ids = children.get(agent_id, ())
        for child_index, child_id in enumerate(child_ids):
            walk(child_id, (*prefix, child_index < len(child_ids) - 1))

    walk(tree.root_agent_id, ())
    if max_lines <= 0:
        return []
    if len(lines) <= max_lines:
        return [item for _, item in lines]
    root_line = lines[0][1]
    if max_lines == 1:
        return [root_line]
    remaining = _crop_lines(lines[1:], selected_agent_id, max_lines - 1)
    return [root_line, *remaining]


def render_timeline_lines(
    snapshot: RuntimeSnapshot,
    *,
    selected_event_id: str | None,
    max_lines: int,
) -> list[Text]:
    lines: list[tuple[str | None, Text]] = []
    for event in snapshot.events:
        line = Text()
        line.append("▶ " if event.event_id == selected_event_id else "  ", style="bold white" if event.event_id == selected_event_id else "dim")
        line.append(event.timestamp[:19], style="dim")
        line.append(f"  {event.event_type}", style=severity_style(event.severity))
        line.append(f"  {event.status}", style="dim")
        line.append(f"  {_truncate(event.summary, 90)}")
        lines.append((event.event_id, line))
    if not lines:
        lines.append((None, Text("No events yet.", style="dim")))
    return _crop_lines(lines, selected_event_id, max_lines)


def format_agent_detail(detail: dict[str, Any]) -> list[str]:
    lines = [
        detail.get("title") or "Unnamed agent",
        f"agent_id: {detail.get('agent_id', '-')}",
        f"role: {detail.get('role', '-')}",
        f"status: {detail.get('status', '-')}",
        f"run_status: {detail.get('run_status', '-')}",
        f"activity: {detail.get('lifecycle_status', detail.get('status', '-'))}",
        f"lifecycle: {detail.get('lifecycle', '-')}",
        f"status_conflict: {'yes' if detail.get('status_conflict') else 'no'}",
        f"mr_level: {detail.get('mr_level', '-')}",
        f"parent: {detail.get('parent_agent_id') or '-'}",
        f"clearance: {detail.get('security_clearance', '-')}",
        f"owned_workflows: {len(detail.get('owned_workflow_ids', []))}",
        f"unread_inbox: {detail.get('unread_inbox_count', 0)}",
        f"latest_run_id: {detail.get('latest_run_id') or '-'}",
    ]
    last_action = detail.get("last_action")
    if last_action:
        lines.append(f"last_action: {_truncate(str(last_action), 100)}")
    mission = detail.get("mission") or detail.get("mission_preview")
    lines.append(f"mission: {_truncate(mission, 180)}")
    latest_messages = [
        item.get("message_id")
        for item in list(detail.get("latest_inbox_messages", []))[:2]
        + list(detail.get("latest_outbox_messages", []))[:2]
        if isinstance(item, dict) and item.get("message_id")
    ]
    lines.append(f"latest_message_ids: {', '.join(latest_messages) if latest_messages else '-'}")
    return lines


def format_event_detail(detail: dict[str, Any]) -> list[str]:
    event = detail["event"]
    lines = [
        event.get("event_type", "-"),
        f"event_id: {event.get('event_id', '-')}",
        f"timestamp: {event.get('timestamp', '-')}",
        f"severity/status: {event.get('severity', '-')}/{event.get('status', '-')}",
        f"actor: {event.get('actor_id') or '-'}",
        f"target: {event.get('target_id') or '-'}",
        f"workflow_id: {event.get('workflow_id') or '-'}",
        f"message_id: {event.get('message_id') or '-'}",
        f"approval_request_id: {event.get('approval_request_id') or '-'}",
        f"summary: {_truncate(event.get('summary'), 160)}",
    ]
    workflow = detail.get("workflow")
    if workflow:
        lines.append(
            f"workflow: {workflow.get('workflow_id')} {workflow.get('title', '-')}"
            f" [{workflow.get('status', '-')}]"
        )
    message = detail.get("message")
    if message:
        lines.append(
            f"message: {message.get('message_id')} {message.get('subject', '-')}"
            f" [{message.get('status', '-')}]"
        )
    approval = detail.get("approval")
    if approval:
        lines.append(
            f"approval: {approval.get('approval_request_id')} {approval.get('capability_name', '-')}"
            f" [{approval.get('status', '-')}]"
        )
    lines.append("payload:")
    lines.extend((detail.get("payload_text") or "{}").splitlines())
    return lines


def _lines_panel(title: str, lines: Iterable[str], *, border_style: str = "white") -> Panel:
    text = Text()
    for index, line in enumerate(lines):
        if index:
            text.append("\n")
        text.append(line)
    return Panel(text if text.plain else Text("-", style="dim"), title=title, border_style=border_style)


def _render_detail_panel(state: TuiState, detail_payload: dict[str, Any] | None) -> Panel:
    if state.show_help:
        return _lines_panel(
            "Help",
            [
                "Tree: Left/Right sibling, Up parent, Down first child",
                "Tree: Enter detail, f timeline, n now, d toggle dead, r refresh",
                "Timeline: Up older, Down newer, Enter detail, f/Esc tree, n newest",
                "Detail: Esc back",
                "Global: q quit, ? help",
            ],
            border_style="bright_white",
        )
    if not detail_payload:
        return _lines_panel("Detail", ["Select an agent or event."], border_style="dim")
    if state.detail_kind == DETAIL_KIND_EVENT:
        return _lines_panel("Event Detail", format_event_detail(detail_payload), border_style="bright_cyan")
    return _lines_panel("Agent Detail", format_agent_detail(detail_payload), border_style="bright_magenta")


def render_screen(
    snapshot: RuntimeSnapshot,
    state: TuiState,
    *,
    width: int,
    height: int,
    detail_payload: dict[str, Any] | None,
) -> str:
    console_stream = io.StringIO()
    console = Console(
        file=console_stream,
        width=max(width, 80),
        height=max(height, 24),
        force_terminal=True,
        color_system="truecolor",
    )
    header = Text()
    header.append(" MR1 ", style="bold white on black")
    header.append(f" {state.base_mode().upper()} ", style="bold bright_white")
    header.append(" live" if state.live_follow and state.base_mode() == "tree" else " frozen", style="bright_green" if state.live_follow and state.base_mode() == "tree" else "yellow")
    header.append(f"  {snapshot.live_count} live", style="bright_green")
    header.append(f"  {snapshot.done_count} done", style="dim")
    if state.status_message:
        header.append(f"  {state.status_message}", style="dim")

    body_layout = Layout()
    body_layout.split_row(
        Layout(name="main", ratio=2),
        Layout(name="detail", ratio=1),
    )
    available_height = max(height - 8, 8)
    main_title = "Tree" if state.base_mode() != TUI_MODE_TIMELINE else "Timeline"
    if state.base_mode() == TUI_MODE_TIMELINE:
        main_lines = render_timeline_lines(
            snapshot,
            selected_event_id=state.selected_event_id,
            max_lines=available_height,
        )
    else:
        main_lines = render_tree_lines(
            snapshot.tree,
            selected_agent_id=state.selected_agent_id,
            show_dead=state.show_dead,
            max_lines=available_height,
        )
    main_group = Group(*main_lines) if main_lines else Text("-", style="dim")
    border_style = "bright_white" if state.mode != TUI_MODE_DETAIL else "dim"
    body_layout["main"].update(Panel(main_group, title=main_title, border_style=border_style))
    detail_panel = _render_detail_panel(state, detail_payload)
    if state.mode == TUI_MODE_DETAIL:
        detail_panel.border_style = "bold bright_white"
    body_layout["detail"].update(detail_panel)

    footer_text = Text()
    footer_text.append(" q quit ", style="bold")
    footer_text.append(" ? help ", style="bold")
    footer_text.append(" Enter detail ", style="bold")
    footer_text.append(" f timeline ", style="bold")
    footer_text.append(" n now ", style="bold")
    footer_text.append(" d dead ", style="bold")
    footer_text.append(" r refresh ", style="bold")

    layout = Layout()
    layout.split_column(
        Layout(Panel(header, border_style="dim"), size=3),
        Layout(body_layout, ratio=1),
        Layout(Panel(footer_text, border_style="dim"), size=3),
    )
    console.print(layout)
    return console_stream.getvalue()
