"""Message-related CLI: inbox/outbox/message commands and formatters.

Contains the `_cmd_inbox`, `_cmd_outbox`, `_cmd_message`,
`_cmd_message_read`, `_cmd_message_archive`, `_cmd_message_send`, and
`_cmd_inbox_triage` handlers plus the message-display formatters
(`_format_messages_table`, `_format_message_detail`, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from mr1.capability_policy import CapabilityApprovalStore
from mr1.inbox_triage import InboxTriagePolicy, InboxTriageResult, InboxTriageRunner
from mr1.messages import MessageStore, PersistentMessage
from mr1.scoped_agents import AgentScopeError, PersistentAgent, PersistentAgentStore
from mr1.workflow_compiler import WorkflowCompilerClient
from mr1.workflow_store import WorkflowStore

from mr1.cli.context import (
    _approval_store_for,
    _require_message,
    _resolve_mailbox_agent_id,
    _runtime_access_for,
)
from mr1.cli.formatting import _compact_text, _render_table, _short_ts


def _format_inbox_triage_result(result: InboxTriageResult) -> str:
    lines = [
        f"summary: {result.summary}",
        f"processed_messages: {len(result.processed_messages)}",
        "actions:",
    ]
    if result.actions_executed:
        for action in result.actions_executed:
            parts = [
                action["type"],
                f"status={action.get('status', '-')}",
                f"reason={action.get('reason', '-')}",
            ]
            if action.get("message_id"):
                parts.append(f"message_id={action['message_id']}")
            if action.get("agent_id"):
                parts.append(f"agent_id={action['agent_id']}")
            if action.get("created_message_id"):
                parts.append(f"created_message_id={action['created_message_id']}")
            if action.get("created_workflow_id"):
                parts.append(f"created_workflow_id={action['created_workflow_id']}")
            if action.get("run_id"):
                parts.append(f"run_id={action['run_id']}")
            if action.get("step_iteration") is not None:
                parts.append(f"step_iteration={action['step_iteration']}")
            if action.get("user_message"):
                parts.append(f"user_message={_compact_text(action['user_message'], limit=80)}")
            lines.append("  - " + " | ".join(parts))
    else:
        lines.append("  none")
    lines.append("counts:")
    lines.append(f"  messages_read={result.counts['messages_read']}")
    lines.append(f"  messages_archived={result.counts['messages_archived']}")
    lines.append(f"  messages_sent={result.counts['messages_sent']}")
    lines.append(f"  agents_run={result.counts['agents_run']}")
    lines.append(f"  workflows_created={result.counts['workflows_created']}")
    return "\n".join(lines)

def _message_preview_payload(message: PersistentMessage) -> dict[str, str]:
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

def _format_message_preview_line(item: dict[str, str], *, direction: str) -> str:
    peer = item["from_agent_id"] if direction == "from" else item["to_agent_id"]
    return (
        f"{item['message_id']} {direction}={peer} "
        f"kind={item['kind']} subject={item['subject']} "
        f"status={item['status']} created={_short_ts(item['created_at'])}"
    )

def _format_messages_table(
    messages: list[PersistentMessage],
    *,
    mode: str,
    json_output: bool = False,
) -> str:
    if json_output:
        return json.dumps([message.to_dict() for message in messages], indent=2, sort_keys=True)
    if not messages:
        return "No messages."
    header = "FROM" if mode == "inbox" else "TO"
    rows = [("MESSAGE_ID", header, "KIND", "SUBJECT", "CREATED_AT", "STATUS")]
    for message in messages:
        rows.append((
            message.message_id,
            message.from_agent_id if mode == "inbox" else message.to_agent_id,
            message.kind,
            message.subject[:40],
            _short_ts(message.created_at),
            message.status,
        ))
    return _render_table(rows)

def _format_message_detail(
    message: PersistentMessage,
    *,
    json_output: bool = False,
) -> str:
    payload = message if isinstance(message, dict) else message.to_dict()
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return "\n".join([
        f"message_id:   {payload['message_id']}",
        f"from:         {payload['from_agent_id']}",
        f"to:           {payload['to_agent_id']}",
        f"kind:         {payload['kind']}",
        f"subject:      {payload['subject']}",
        f"status:       {payload['status']}",
        f"workflow_id:  {payload.get('workflow_id') or '-'}",
        f"task_id:      {payload.get('task_id') or '-'}",
        f"created_at:   {payload['created_at']}",
        f"read_at:      {payload.get('read_at') or '-'}",
        f"archived_at:  {payload.get('archived_at') or '-'}",
        "body:",
        str(payload.get("body", "")),
    ])

def _cmd_inbox(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    message_store = getattr(args, "message_store")
    try:
        target_agent_id = _resolve_mailbox_agent_id(
            getattr(args, "agent", None),
            caller_agent_id,
            scoped_agents,
        )
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_messages_table(
        message_store.list_inbox(target_agent_id, include_archived=bool(args.archived)),
        mode="inbox",
        json_output=args.json,
    ))
    return 0

def _cmd_inbox_triage(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    from mr1.mr1 import StateManager

    message_store = getattr(args, "message_store")
    compiler_client = WorkflowCompilerClient(
        compiler=getattr(args, "workflow_compiler", None),
        scoped_agent_store=scoped_agents,
    )
    runner = InboxTriageRunner(
        workflow_store=store,
        scoped_agent_store=scoped_agents,
        message_store=message_store,
        workflow_compiler_client=compiler_client,
        pending_workflow_state=StateManager(),
    )
    policy = InboxTriagePolicy(
        max_messages=args.max_messages,
        max_actions=args.max_actions,
    )
    try:
        result = runner.run(policy, caller_agent_id=caller_agent_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_inbox_triage_result(result))
    return 0

def _cmd_outbox(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    message_store = getattr(args, "message_store")
    try:
        target_agent_id = _resolve_mailbox_agent_id(
            getattr(args, "agent", None),
            caller_agent_id,
            scoped_agents,
        )
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_messages_table(
        message_store.list_outbox(target_agent_id, include_archived=bool(args.archived)),
        mode="outbox",
        json_output=args.json,
    ))
    return 0

def _cmd_message(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    runtime_access = _runtime_access_for(
        store,
        scoped_agents,
        getattr(args, "message_store"),
    )
    message_store = getattr(args, "message_store")
    try:
        _require_message(message_store, args.message_id, caller_agent_id)
        message = runtime_access.read_message(
            args.message_id,
            caller_agent_id=caller_agent_id,
        )
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_message_detail(message, json_output=args.json))
    return 0

def _cmd_message_read(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store, scoped_agents
    message_store = getattr(args, "message_store")
    try:
        _require_message(message_store, args.message_id, caller_agent_id)
        message = message_store.mark_read(args.message_id, actor_id=caller_agent_id)
        if message is None:
            raise ValueError(f"message not found: {args.message_id}")
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(message.message_id)
    return 0

def _cmd_message_archive(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store, scoped_agents
    message_store = getattr(args, "message_store")
    try:
        _require_message(message_store, args.message_id, caller_agent_id)
        message = message_store.archive_message(args.message_id)
        if message is None:
            raise ValueError(f"message not found: {args.message_id}")
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(message.message_id)
    return 0

def _cmd_message_send(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    message_store = getattr(args, "message_store")
    body_path = Path(args.body_file)
    if not body_path.exists():
        print(f"error: message body file not found: {body_path}", file=sys.stderr)
        return 2
    try:
        body = body_path.read_text(encoding="utf-8")
    except OSError:
        print(f"error: message body file not found: {body_path}", file=sys.stderr)
        return 2
    if not message_store.can_agent_send_message(caller_agent_id, args.to_agent_id):
        print("error: access denied: recipient not in agent scope", file=sys.stderr)
        return 2
    if scoped_agents.load_agent(caller_agent_id) is None:
        print("error: access denied: recipient not in agent scope", file=sys.stderr)
        return 2
    message = message_store.create_message(
        from_agent_id=caller_agent_id,
        to_agent_id=args.to_agent_id,
        kind=args.kind,
        subject=args.subject,
        body=body,
    )
    print(message.message_id)
    return 0
