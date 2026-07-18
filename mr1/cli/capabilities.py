"""Capability/Tool/Approval CLI: handlers + formatters.

Includes the `_cmd_capability*`, `_cmd_tool*`, `_cmd_approvals_*`, and
`_cmd_capability_audit_*` handlers plus their display formatters.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.capability_policy import (
    CapabilityApprovalDecision,
    CapabilityApprovalRequest,
    CapabilityApprovalStore,
)
from mr1.event_log import bind_correlation_id, cli_correlation_id
from mr1.scoped_agents import AgentScopeError, AgentStore
from mr1.tools import ToolRegistry, default_tool_registry
from mr1.workflow_store import WorkflowStore

from mr1.cli.context import (
    _approval_store_for,
    _find_visible_audit_entry,
    _require_visible_approval,
    _runtime_access_for,
    _runtime_root_for,
    _timeline_for,
    _visible_approvals,
    _visible_audit_entries,
)
from mr1.cli.formatting import (
    _brief_description,
    _config_shape_for_tool,
    _format_description_text,
    _reject_invalid_flag_combination,
    _render_table,
    _select_description_view,
    _short_ts,
)


def _format_capabilities(
    registry: Optional[CapabilityRegistry] = None,
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    active_registry = registry or default_capability_registry()
    capabilities = active_registry.describe_all()
    view = [_brief_description(item) for item in capabilities] if brief else capabilities
    if json_output:
        return json.dumps(view, indent=2, sort_keys=True)
    rows = [("NAME", "TYPE", "DESCRIPTION")]
    for item in capabilities:
        rows.append((item["name"], item["type"], item["description"]))
    return _render_table(rows)

def _format_capability(
    name: str,
    registry: Optional[CapabilityRegistry] = None,
    *,
    json_output: bool = False,
    example_only: bool = False,
    brief: bool = False,
) -> str:
    active_registry = registry or default_capability_registry()
    description = active_registry.describe_capability(name)
    view = _select_description_view(
        description,
        example_only=example_only,
        brief=brief,
    )
    if json_output:
        return json.dumps(view, indent=2, sort_keys=True)
    if example_only:
        return json.dumps(view, indent=2, sort_keys=True)
    if brief:
        return "\n".join([
            f"name:        {view['name']}",
            f"type:        {view['type']}",
            f"description: {view['description']}",
        ])
    return _format_description_text(description)

def _format_tools(
    registry: Optional[ToolRegistry] = None,
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    active_registry = registry or default_tool_registry()
    descriptions = active_registry.describe_all_tools()
    if not descriptions:
        return "No tools registered."
    view = [_brief_description(item) for item in descriptions] if brief else descriptions
    if json_output:
        return json.dumps(view, indent=2, sort_keys=True)
    if brief:
        rows = [("TOOL", "DESCRIPTION")]
        for item in descriptions:
            rows.append((item["name"], item["description"]))
        return _render_table(rows)
    rows = [("TOOL", "DESCRIPTION", "CONFIG_SHAPE")]
    for item in descriptions:
        rows.append((item["name"], item["description"], _config_shape_for_tool(active_registry, item["name"])))
    return _render_table(rows)

def _format_tool(
    tool_type: str,
    registry: Optional[ToolRegistry] = None,
    *,
    json_output: bool = False,
    example_only: bool = False,
    brief: bool = False,
) -> str:
    active_registry = registry or default_tool_registry()
    description = active_registry.describe_tool(tool_type)
    view = _select_description_view(
        description,
        example_only=example_only,
        brief=brief,
    )
    if json_output:
        return json.dumps(view, indent=2, sort_keys=True)
    if example_only:
        return json.dumps(view, indent=2, sort_keys=True)
    if brief:
        return "\n".join([
            f"name:        {view['name']}",
            f"type:        {view['type']}",
            f"description: {view['description']}",
        ])
    return _format_description_text(description)

def _format_approval(
    approval: CapabilityApprovalRequest,
    *,
    json_output: bool = False,
) -> str:
    if json_output:
        return json.dumps(approval.to_dict(), indent=2, sort_keys=True)
    lines = [
        f"approval_request_id:  {approval.approval_request_id}",
        f"status:               {approval.status}",
        f"capability_name:      {approval.capability_name}",
        f"actor_id:             {approval.requesting_actor_id}",
        f"designated_approver:  {approval.designated_approver_id or '-'}",
        f"risk_score:           {approval.risk_score:.2f}",
        f"reason:               {approval.reason}",
        f"workflow_id:          {approval.workflow_id or '-'}",
        f"task_id:              {approval.task_id or '-'}",
        f"step_id:              {approval.original_step_id or '-'}",
        f"message_id:           {approval.message_id or '-'}",
        f"requested_scope_path: {approval.requested_scope_path or '-'}",
        f"used_at:              {approval.used_at or '-'}",
        f"used_by_audit_id:     {approval.used_by_audit_id or '-'}",
        "args_summary:",
        json.dumps(approval.args, indent=2, sort_keys=True),
        "scope_roots:",
        json.dumps(approval.scope_summary.get("allowed_roots", []), indent=2, sort_keys=True),
        "decision:",
        json.dumps(approval.decision, indent=2, sort_keys=True),
    ]
    return "\n".join(lines)

def _format_approvals_table(
    approvals: list[CapabilityApprovalRequest],
    *,
    json_output: bool = False,
) -> str:
    if json_output:
        return json.dumps([item.to_dict() for item in approvals], indent=2, sort_keys=True)
    if not approvals:
        return "No approval requests."
    rows = [("APPROVAL_ID", "STATUS", "CAPABILITY", "ACTOR", "APPROVER", "RISK")]
    for item in approvals:
        rows.append((
            item.approval_request_id,
            item.status,
            item.capability_name,
            item.requesting_actor_id,
            item.designated_approver_id or "-",
            f"{item.risk_score:.2f}",
        ))
    return _render_table(rows)

def _format_approval_decision_result(
    approval: CapabilityApprovalRequest,
    *,
    grant_scope: bool = False,
) -> str:
    lines = ["Approved." if approval.status == "approved" else "Denied."]
    if approval.status == "approved" and approval.workflow_id and approval.task_id:
        lines.extend([
            "Blocked workflow task reopened automatically.",
            "The scheduler will resume it on the next tick.",
        ])
    if (
        approval.status == "approved"
        and approval.reason == "outside_actor_scope"
        and not grant_scope
    ):
        lines.append("This was single-use. Use --grant-scope to persist scope access if intended.")
    return "\n".join(lines)

def _format_capability_audit_table(
    items: list[dict[str, Any]],
    *,
    json_output: bool = False,
) -> str:
    if json_output:
        return json.dumps(items, indent=2, sort_keys=True)
    if not items:
        return "No capability audits."
    rows = [("AUDIT_ID", "ACTOR", "CAPABILITY", "STATUS", "RISK", "WORKFLOW", "TASK")]
    for item in items:
        rows.append((
            str(item.get("audit_id", "-")),
            str(item.get("actor_id", "-")),
            str(item.get("capability_name", "-")),
            str(item.get("status", "-")),
            str(item.get("risk_score", "-")),
            str(item.get("workflow_id") or "-"),
            str(item.get("task_id") or "-"),
        ))
    return _render_table(rows)

def _format_capability_audit_detail(
    index_entry: dict[str, Any],
    record: dict[str, Any],
    *,
    json_output: bool = False,
) -> str:
    payload = {
        "index": dict(index_entry),
        "record": dict(record),
    }
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return "\n".join([
        f"audit_id:        {index_entry.get('audit_id')}",
        f"audit_path:      {index_entry.get('audit_path')}",
        f"actor_id:        {index_entry.get('actor_id')}",
        f"capability_name: {index_entry.get('capability_name')}",
        f"status:          {index_entry.get('status')}",
        f"risk_score:      {index_entry.get('risk_score')}",
        f"reason:          {index_entry.get('reason')}",
        f"workflow_id:     {index_entry.get('workflow_id') or '-'}",
        f"task_id:         {index_entry.get('task_id') or '-'}",
        f"step_id:         {index_entry.get('step_id') or '-'}",
        "args_summary:",
        json.dumps(index_entry.get("args_summary"), indent=2, sort_keys=True),
        "scope_roots:",
        json.dumps(index_entry.get("scope_roots"), indent=2, sort_keys=True),
        "record:",
        json.dumps(record, indent=2, sort_keys=True),
    ])

def _cmd_capabilities(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    rc = _reject_invalid_flag_combination(args)
    if rc is not None:
        return rc
    print(_format_capabilities(json_output=args.json, brief=args.brief))
    return 0

def _cmd_capability(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    rc = _reject_invalid_flag_combination(args)
    if rc is not None:
        return rc
    try:
        print(_format_capability(
            args.name,
            json_output=args.json,
            example_only=args.example,
            brief=args.brief,
        ))
    except ValueError:
        print(f"error: capability not found: {args.name}", file=sys.stderr)
        return 2
    return 0

def _cmd_capability_call(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    from mr1.capability_runner import CapabilityRunner
    config_path = Path(args.config_file)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not isinstance(config, dict):
        print("error: config file must be a JSON object", file=sys.stderr)
        return 2
    runner = CapabilityRunner(
        scoped_agent_store=scoped_agents,
        workspace_root=store.root.parent,
    )
    try:
        result = runner.run_capability(args.name, config, caller_agent_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines = [
        f"capability:   {result.capability}",
        f"status:       {result.status}",
        f"duration_ms:  {result.duration_ms}",
        "output:",
        json.dumps(result.output, indent=2, sort_keys=True),
    ]
    if result.error is not None:
        lines.append(f"error:        {result.error}")
    print("\n".join(lines))
    return 0

def _cmd_tools(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    rc = _reject_invalid_flag_combination(args)
    if rc is not None:
        return rc
    print(_format_tools(json_output=args.json, brief=args.brief))
    return 0

def _cmd_tool(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    rc = _reject_invalid_flag_combination(args)
    if rc is not None:
        return rc
    try:
        print(_format_tool(
            args.tool_type,
            json_output=args.json,
            example_only=args.example,
            brief=args.brief,
        ))
    except ValueError:
        print(f"error: tool not found: {args.tool_type}", file=sys.stderr)
        return 2
    return 0

def _cmd_approvals_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    approvals = _visible_approvals(_approval_store_for(store), scoped_agents, caller_agent_id)
    print(_format_approvals_table(approvals, json_output=args.json))
    return 0

def _cmd_approvals_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    try:
        approval = _require_visible_approval(
            _approval_store_for(store),
            args.approval_request_id,
            scoped_agents,
            caller_agent_id,
        )
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_approval(approval, json_output=args.json))
    return 0

def _cmd_approvals_decide(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    approval_store = _approval_store_for(store)
    try:
        _require_visible_approval(
            approval_store,
            args.approval_request_id,
            scoped_agents,
            caller_agent_id,
        )
        decision = CapabilityApprovalDecision(
            approval_request_id=args.approval_request_id,
            decision=args.decision,
            decided_by=caller_agent_id,
            reason=args.reason,
            timestamp=datetime.now(timezone.utc).timestamp(),
            approval_scope="grant_scope" if getattr(args, "grant_scope", False) else "single_use",
        )
        approval = approval_store.apply_decision(
            args.approval_request_id,
            decision=decision,
            scoped_agent_store=scoped_agents,
        )
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_approval_decision_result(
        approval,
        grant_scope=getattr(args, "grant_scope", False),
    ))
    return 0

def _cmd_capability_audit_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    del store
    try:
        entries = _visible_audit_entries(scoped_agents, caller_agent_id, args.agent_id)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_capability_audit_table(entries, json_output=args.json))
    return 0

def _cmd_capability_audit_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: AgentStore,
) -> int:
    del store
    try:
        index_entry = _find_visible_audit_entry(scoped_agents, caller_agent_id, args.audit_id)
        record = json.loads(Path(index_entry["audit_path"]).read_text(encoding="utf-8"))
    except (ValueError, AgentScopeError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_capability_audit_detail(index_entry, record, json_output=args.json))
    return 0
