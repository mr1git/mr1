"""Event/Timeline CLI: events and timeline commands and formatters.

Includes `_cmd_events` and all `_cmd_timeline_*` handlers plus their
display formatters.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from mr1.event_log import SystemEvent
from mr1.messages import MessageStore
from mr1.scheduler import WorkflowSpecError
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore
from mr1.workflow_store import WorkflowStore

from mr1.cli.context import (
    _event_visible,
    _load_scoped_workflow,
    _runtime_root_for,
    _timeline_for,
    _visible_timeline_events,
    _visible_workflows,
)
from mr1.cli.formatting import (
    _compact_text,
    _render_table,
    _short_ts,
)


def _format_events(events: list) -> str:
    if not events:
        return "No events."
    rows = [("TIMESTAMP", "EVENT", "TASK_ID", "ATTEMPT", "MESSAGE")]
    for ev in events:
        rows.append((
            _short_ts(ev.timestamp),
            ev.event_type,
            ev.task_id or "-",
            str(ev.attempt_id) if ev.attempt_id is not None else "-",
            (ev.message or "")[:60],
        ))
    return _render_table(rows)

def _format_timeline_events(
    events: list[SystemEvent],
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    if json_output:
        payload = [event.to_dict() for event in events]
        if brief:
            payload = [
                {
                    "event_index": item["event_index"],
                    "timestamp": item["timestamp"],
                    "event_type": item["event_type"],
                    "status": item["status"],
                    "summary": item["summary"],
                    "correlation_id": item["correlation_id"],
                }
                for item in payload
            ]
        return json.dumps(payload, indent=2, sort_keys=True)
    if not events:
        return "No timeline events."
    rows = [(
        "INDEX",
        "TIMESTAMP",
        "TYPE",
        "KIND",
        "ACTOR",
        "TARGET",
        "STATUS",
        "SEVERITY",
        "CORRELATION",
        "SUMMARY",
    )]
    for event in events:
        rows.append((
            str(event.event_index),
            _short_ts(event.timestamp),
            event.event_type,
            event.event_kind,
            event.actor_id or "-",
            event.target_id or "-",
            event.status,
            event.severity,
            (event.correlation_id or "-")[:32],
            event.summary[:60],
        ))
    return _render_table(rows)

def _format_timeline_event_detail(
    event: SystemEvent,
    *,
    json_output: bool = False,
) -> str:
    if json_output:
        return json.dumps(event.to_dict(), indent=2, sort_keys=True)
    return "\n".join([
        f"event_id:            {event.event_id}",
        f"event_index:         {event.event_index}",
        f"event_version:       {event.event_version}",
        f"timestamp:           {event.timestamp}",
        f"event_type:          {event.event_type}",
        f"event_kind:          {event.event_kind}",
        f"actor_id:            {event.actor_id or '-'}",
        f"actor_type:          {event.actor_type or '-'}",
        f"target_id:           {event.target_id or '-'}",
        f"target_type:         {event.target_type or '-'}",
        f"status:              {event.status}",
        f"severity:            {event.severity}",
        f"summary:             {event.summary}",
        f"correlation_id:      {event.correlation_id or '-'}",
        f"parent_event_id:     {event.parent_event_id or '-'}",
        f"workflow_id:         {event.workflow_id or '-'}",
        f"task_id:             {event.task_id or '-'}",
        f"step_id:             {event.step_id or '-'}",
        f"message_id:          {event.message_id or '-'}",
        f"approval_request_id: {event.approval_request_id or '-'}",
        f"audit_id:            {event.audit_id or '-'}",
        f"record_path:         {event.record_path or '-'}",
        "metadata:",
        json.dumps(event.metadata, indent=2, sort_keys=True),
    ])

def _cmd_events(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    events = store.load_events(
        args.workflow_id,
        since=args.since,
        until=args.until,
        task_id=args.task,
        limit=args.limit,
    )
    print(_format_events(events))
    return 0

def _cmd_timeline_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    message_store = getattr(args, "message_store")
    events = _visible_timeline_events(
        store,
        scoped_agents,
        message_store,
        caller_agent_id,
    )
    if getattr(args, "limit", None):
        events = events[-args.limit:]
    print(_format_timeline_events(events, json_output=args.json, brief=args.brief))
    return 0

def _cmd_timeline_recent(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    message_store = getattr(args, "message_store")
    events = _visible_timeline_events(
        store,
        scoped_agents,
        message_store,
        caller_agent_id,
    )
    limit = getattr(args, "limit", 20)
    events = sorted(events, key=lambda item: item.event_index, reverse=True)[:limit]
    print(_format_timeline_events(events, json_output=args.json, brief=args.brief))
    return 0

def _cmd_timeline_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    message_store = getattr(args, "message_store")
    event = _timeline_for(store).get_event(args.event_id)
    if event is None:
        print(f"error: event not found: {args.event_id}", file=sys.stderr)
        return 2
    if not _event_visible(
        event,
        store=store,
        scoped_agents=scoped_agents,
        message_store=message_store,
        caller_agent_id=caller_agent_id,
    ):
        print("error: access denied: event not in agent scope", file=sys.stderr)
        return 2
    print(_format_timeline_event_detail(event, json_output=args.json))
    return 0

def _cmd_timeline_trace(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    message_store = getattr(args, "message_store")
    events = [
        event for event in _timeline_for(store).trace_by_correlation(args.correlation_id)
        if _event_visible(
            event,
            store=store,
            scoped_agents=scoped_agents,
            message_store=message_store,
            caller_agent_id=caller_agent_id,
        )
    ]
    print(_format_timeline_events(events, json_output=args.json, brief=args.brief))
    return 0

def _cmd_timeline_blocked(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    message_store = getattr(args, "message_store")
    events = [
        event for event in _timeline_for(store).blocked_now()
        if _event_visible(
            event,
            store=store,
            scoped_agents=scoped_agents,
            message_store=message_store,
            caller_agent_id=caller_agent_id,
        )
    ]
    print(_format_timeline_events(events, json_output=args.json, brief=args.brief))
    return 0

def _cmd_timeline_approvals(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    message_store = getattr(args, "message_store")
    events = [
        event for event in _timeline_for(store).approval_history()
        if _event_visible(
            event,
            store=store,
            scoped_agents=scoped_agents,
            message_store=message_store,
            caller_agent_id=caller_agent_id,
        )
    ]
    print(_format_timeline_events(events, json_output=args.json, brief=args.brief))
    return 0

def _cmd_timeline_agent(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    message_store = getattr(args, "message_store")
    try:
        scoped_agents.get_visible_agent(caller_agent_id, args.agent_id)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    events = [
        event for event in _timeline_for(store).agent_activity(args.agent_id)
        if _event_visible(
            event,
            store=store,
            scoped_agents=scoped_agents,
            message_store=message_store,
            caller_agent_id=caller_agent_id,
        )
    ]
    print(_format_timeline_events(events, json_output=args.json, brief=args.brief))
    return 0

def _cmd_timeline_workflow(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    events = _timeline_for(store).workflow_trace(args.workflow_id)
    print(_format_timeline_events(events, json_output=args.json, brief=args.brief))
    return 0
