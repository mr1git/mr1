"""Agent-related CLI: agent/agents/scope/lifecycle commands and formatters.

Contains all `_cmd_agent*` handlers, runtime-agent formatters, MRn
step/run result formatters, scope-grant formatting, and persistent-agent
payload helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from mr1.agents import AgentRegistry, default_agent_registry, run_agent_health
from mr1.capability_policy import CapabilityApprovalStore
from mr1.dataflow import TaskOutput
from mr1.mrn_loop import MRnStepResult, MRnStepRunner
from mr1.mrn_run import MRnRunPolicy, MRnRunResult, MRnRunRunner
from mr1.messages import MessageStore
from mr1.scoped_agents import (
    AgentScopeError,
    PersistentAgent,
    PersistentAgentStore,
)
from mr1.workflow_models import Provenance, TaskStatus, Workflow
from mr1.workflow_store import WorkflowStore

from mr1.cli.context import (
    _approval_store_for,
    _find_visible_audit_entry,
    _runtime_access_for,
    _runtime_root_for,
    _visible_audit_entries,
)
from mr1.cli.formatting import (
    _brief_description,
    _compact_text,
    _format_description_text,
    _label_for_task_id,
    _reject_invalid_flag_combination,
    _render_table,
    _short_ts,
)
from mr1.cli.messages import (
    _format_message_preview_line,
    _message_preview_payload,
    _pending_parent_messages,
)


def _format_runtime_agents(
    registry: Optional[AgentRegistry] = None,
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    active_registry = registry or default_agent_registry()
    descriptions = active_registry.describe_all()
    if not descriptions:
        return "No agents registered."
    view = [_brief_description(item) for item in descriptions] if brief else descriptions
    if json_output:
        return json.dumps(view, indent=2, sort_keys=True)
    if brief:
        rows = [("AGENT", "DESCRIPTION")]
        for item in descriptions:
            rows.append((item["name"], item["description"]))
        return _render_table(rows)
    rows = [("AGENT", "DESCRIPTION", "BINARY")]
    for item in descriptions:
        rows.append((item["name"], item["description"], str(item.get("runtime", {}).get("binary", "-"))))
    return _render_table(rows)

def _format_runtime_agent(
    agent_name: str,
    registry: Optional[AgentRegistry] = None,
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    active_registry = registry or default_agent_registry()
    description = active_registry.describe_agent(agent_name)
    view = _brief_description(description) if brief else description
    if json_output:
        return json.dumps(view, indent=2, sort_keys=True)
    if brief:
        return "\n".join([
            f"name:        {view['name']}",
            f"type:        {view['type']}",
            f"description: {view['description']}",
    ])
    return _format_description_text(description)

def _format_runtime_agent_health(
    agent_name: str,
    registry: Optional[AgentRegistry] = None,
    *,
    json_output: bool = False,
) -> str:
    active_registry = registry or default_agent_registry()
    active_registry.get_definition(agent_name)
    # Look up `run_agent_health` through the workflow_cli facade so tests
    # that `@patch("mr1.workflow_cli.run_agent_health")` still affect the
    # call site after this function moved into mr1.cli.agents.
    from mr1 import workflow_cli as _facade
    result = _facade.run_agent_health(agent_name, registry=active_registry)
    if json_output:
        return json.dumps(result, indent=2, sort_keys=True)
    lines = [
        f"agent:       {agent_name}",
        f"status:      {result['status']}",
        "checks:",
    ]
    for key, value in result.get("checks", {}).items():
        lines.append(f"  {key}: {value}")
    if result.get("error"):
        lines.append(f"error:       {result['error']}")
    return "\n".join(lines)

def _format_agents(
    agents: list[PersistentAgent],
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    if not agents:
        return "No agents."
    payload = [_persistent_agent_payload(agent) for agent in agents]
    if brief:
        payload = [
            {
                "agent_id": item["agent_id"],
                "agent_type": item["agent_type"],
                "title": item["title"],
                "status": item["status"],
            }
            for item in payload
        ]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    rows = [("AGENT_ID", "TYPE", "TITLE", "STATUS", "LEVEL", "PARENT", "WORKFLOWS")]
    for agent in agents:
        rows.append((
            agent.agent_id,
            agent.agent_type,
            agent.title,
            agent.status,
            str(agent.tree_level),
            agent.parent_agent_id or "-",
            str(len(agent.owned_workflow_ids)),
        ))
    return _render_table(rows)

def _persistent_agent_payload(
    agent: PersistentAgent,
    *,
    reports: Optional[list[Path]] = None,
    message_store: Optional[MessageStore] = None,
    workflow_store: Optional[WorkflowStore] = None,
) -> dict[str, Any]:
    payload = agent.to_dict()
    last_run = agent.last_run or {}
    payload["latest_run_id"] = last_run.get("run_id")
    payload["latest_run_stopped_reason"] = last_run.get("stopped_reason")
    payload["latest_run_step_count"] = last_run.get("step_count")
    payload["latest_run_at"] = last_run.get("finished_at")
    if workflow_store is not None:
        payload.update(_agent_runtime_activity_payload(agent, workflow_store))
    if reports is not None:
        payload["reports"] = [str(path) for path in reports]
        payload["latest_reports"] = [path.name for path in reports[:5]]
    if message_store is not None:
        inbox = message_store.list_inbox(agent.agent_id)
        outbox = message_store.list_outbox(agent.agent_id)
        payload["unread_inbox_count"] = sum(1 for item in inbox if item.status == "unread")
        payload["latest_inbox_messages"] = [
            _message_preview_payload(item)
            for item in inbox[:3]
        ]
        payload["latest_outbox_messages"] = [
            _message_preview_payload(item)
            for item in outbox[:3]
        ]
        pending_parent_messages = _pending_parent_messages(agent, message_store)
        waiting_on_parent = bool(pending_parent_messages)
        if agent.parent_agent_id and agent.run_status == "waiting":
            action_name = (agent.last_action or {}).get("action")
            waiting_on_parent = waiting_on_parent or action_name in {"ask_parent", "send_message"}
        payload["waiting_on_parent"] = waiting_on_parent
        payload["pending_parent_questions"] = len(pending_parent_messages)
        payload["pending_parent_messages"] = [
            _message_preview_payload(item)
            for item in pending_parent_messages[:3]
        ]
    return payload

def _summarize_last_action(action: Optional[dict[str, Any]]) -> str:
    if not action:
        return "-"
    action_name = action.get("action", "-")
    reason = action.get("reason", "-")
    next_status = action.get("next_status", "-")
    return f"{action_name} -> {next_status} ({reason})"

def _format_agent(
    agent: PersistentAgent,
    *,
    reports: Optional[list[Path]] = None,
    message_store: Optional[MessageStore] = None,
    workflow_store: Optional[WorkflowStore] = None,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    if isinstance(agent, dict):
        payload = dict(agent)
        agent_id = str(payload["agent_id"])
        agent_type = str(payload["agent_type"])
        title = str(payload["title"])
        status = str(payload["status"])
        clearance = float(payload.get("security_clearance", 0.0))
        mission = payload.get("mission")
        mode = str(payload.get("mode", "-"))
        run_status = str(payload.get("run_status", "-"))
        current_iteration = int(payload.get("current_iteration", 0))
        last_step_at = payload.get("last_step_at")
        last_action = payload.get("last_action")
        tree_level = int(payload.get("tree_level", 0))
        parent_agent_id = payload.get("parent_agent_id")
        created_at = str(payload.get("created_at", "-"))
        owned_workflow_ids = list(payload.get("owned_workflow_ids", []))
        scope_roots = list(payload.get("scope_roots", []))
        report_names = [Path(str(path)).name for path in payload.get("reports", [])]
    else:
        payload = _persistent_agent_payload(
            agent,
            reports=reports,
            message_store=message_store,
            workflow_store=workflow_store,
        )
        agent_id = agent.agent_id
        agent_type = agent.agent_type
        title = agent.title
        status = agent.status
        clearance = agent.security_clearance
        mission = agent.mission
        mode = agent.mode
        run_status = agent.run_status
        current_iteration = agent.current_iteration
        last_step_at = agent.last_step_at
        last_action = agent.last_action
        tree_level = agent.tree_level
        parent_agent_id = agent.parent_agent_id
        created_at = agent.created_at
        owned_workflow_ids = list(agent.owned_workflow_ids)
        scope_roots = list(agent.scope_roots)
        report_names = [path.name for path in reports or []]
    if brief:
        payload = {
            "agent_id": payload["agent_id"],
            "agent_type": payload["agent_type"],
            "title": payload["title"],
            "status": payload["status"],
        }
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        f"agent_id:     {agent_id}",
        f"type:         {agent_type}",
        f"title:        {title}",
        f"status:       {status}",
        f"clearance:    {clearance:.2f}",
        f"mission:      {_compact_text(mission)}",
        f"mode:         {mode}",
        f"run_status:   {run_status}",
        f"active_jobs:  {payload.get('active_jobs', 0)}",
        f"active_workflows: {payload.get('active_workflows', 0)}",
        f"runtime_activity: {payload.get('runtime_activity', 'no active jobs')}",
        f"has_running_processes: {'yes' if payload.get('has_running_processes') else 'no'}",
        f"iteration:    {current_iteration}",
        f"last_step_at: {_short_ts(last_step_at)}",
        f"last_action:  {_summarize_last_action(last_action if isinstance(last_action, dict) else None)}",
        f"latest_run:   {payload.get('latest_run_id') or '-'}",
        f"run_reason:   {payload.get('latest_run_stopped_reason') or '-'}",
        f"run_steps:    {payload.get('latest_run_step_count') or 0}",
        f"run_at:       {_short_ts(payload.get('latest_run_at'))}",
        f"parent_req:   {_compact_text(payload.get('parent_request'))}",
        f"waiting_on_parent: {'yes' if payload.get('waiting_on_parent') else 'no'}",
        f"pending_parent_questions: {payload.get('pending_parent_questions', 0)}",
        f"tree_level:   {tree_level}",
        f"parent:       {parent_agent_id or '-'}",
        f"created_at:   {created_at}",
        f"workflows:    {len(owned_workflow_ids)}",
        f"unread_inbox: {payload.get('unread_inbox_count', 0)}",
    ]
    if owned_workflow_ids:
        lines.append(f"owned_ids:    {', '.join(owned_workflow_ids)}")
    lines.append(f"scope_roots:  {', '.join(scope_roots) or '-'}")
    lines.append("reports:")
    for name in report_names:
        lines.append(f"  {name}")
    if not report_names:
        lines.append("  none")
    lines.append("latest_inbox:")
    for item in payload.get("latest_inbox_messages", []):
        lines.append(f"  {_format_message_preview_line(item, direction='from')}")
    if not payload.get("latest_inbox_messages"):
        lines.append("  none")
    lines.append("latest_outbox:")
    for item in payload.get("latest_outbox_messages", []):
        lines.append(f"  {_format_message_preview_line(item, direction='to')}")
    if not payload.get("latest_outbox_messages"):
        lines.append("  none")
    lines.append("pending_parent_messages:")
    for item in payload.get("pending_parent_messages", []):
        lines.append(f"  {_format_message_preview_line(item, direction='to')}")
    if not payload.get("pending_parent_messages"):
        lines.append("  none")
    return "\n".join(lines)

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

def _format_mrn_step_result(result: MRnStepResult) -> str:
    parts = [
        f"agent_id={result.agent_id}",
        f"iteration={result.iteration}",
        f"action={result.action}",
        f"status={result.status_after}",
        f"reason={result.reason}",
    ]
    if result.workflow_id:
        parts.append(f"workflow_id={result.workflow_id}")
    if result.report_path:
        parts.append(f"report={Path(result.report_path).name}")
    if result.message_id:
        parts.append(f"message_id={result.message_id}")
    if result.parent_request:
        parts.append(f"parent_request={_compact_text(result.parent_request, limit=80)}")
    if result.error:
        parts.append(f"error={result.error}")
    return "step " + " | ".join(parts)

def _format_mrn_run_result(result: MRnRunResult) -> str:
    return "run " + " | ".join([
        f"agent_id={result.agent_id}",
        f"run_id={result.run_id}",
        f"steps={result.steps_completed}",
        f"stopped_reason={result.stopped_reason}",
        f"workflows_created={result.workflows_created}",
        f"messages_created={result.messages_created}",
        f"final_run_status={result.final_run_status}",
    ])

def _format_scope_grants(
    agent: PersistentAgent,
    *,
    json_output: bool = False,
) -> str:
    grants = list(agent.scope_grants or [])
    payload = {
        "agent_id": agent.agent_id,
        "security_clearance": agent.security_clearance,
        "scope_roots": list(agent.scope_roots),
        "scope_grants": grants,
    }
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        f"agent_id:            {agent.agent_id}",
        f"security_clearance:  {agent.security_clearance:.2f}",
        "scope_roots:",
    ]
    for item in agent.scope_roots:
        lines.append(f"  {item}")
    if not agent.scope_roots:
        lines.append("  none")
    lines.append("scope_grants:")
    for item in grants:
        lines.append(
            "  "
            + f"path={item.get('path', '-')} "
            + f"granted_by={item.get('granted_by', '-')} "
            + f"timestamp={item.get('timestamp', '-')}"
        )
    if not grants:
        lines.append("  none")
    return "\n".join(lines)

def _cmd_agents(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    rc = _reject_invalid_flag_combination(args)
    if rc is not None:
        return rc
    print(_format_agents(
        scoped_agents.list_visible_agents(caller_agent_id),
        json_output=args.json,
        brief=args.brief,
    ))
    return 0

def _cmd_agent(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    usage = (
        "agent <create <title>|kill <ag-id>|assign <ag-id> <mission-file>|"
        "step <ag-id>|run <ag-id> [--steps N] [--max-workflows N] "
        "[--no-confirm-workflows]|<ag-id>|kazi [health]>"
    )
    rc = _reject_invalid_flag_combination(args)
    if rc is not None:
        return rc
    parts = list(args.parts)
    if not parts:
        print(f"error: usage: {usage}", file=sys.stderr)
        return 2
    if parts[0] == "create":
        title = " ".join(parts[1:]).strip()
        if not title:
            print("error: usage: agent create <title>", file=sys.stderr)
            return 2
        try:
            agent = scoped_agents.create_child_agent(caller_agent_id, title)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(agent.agent_id)
        return 0
    if parts[0] == "kill":
        if len(parts) != 2:
            print("error: usage: agent kill <ag-id>", file=sys.stderr)
            return 2
        try:
            agent = scoped_agents.terminate_agent(caller_agent_id, parts[1])
        except (ValueError, AgentScopeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(agent.agent_id)
        return 0
    if parts[0] == "assign":
        return _cmd_agent_assign(
            argparse.Namespace(agent_id=parts[1] if len(parts) > 1 else None, mission_file=parts[2] if len(parts) > 2 else None),
            store,
            caller_agent_id,
            scoped_agents,
        ) if len(parts) == 3 else _print_agent_usage_error("agent assign <ag-id> <mission-file>")
    if parts[0] == "step":
        return _cmd_agent_step(
            argparse.Namespace(agent_id=parts[1] if len(parts) > 1 else None, workflow_compiler=getattr(args, "workflow_compiler", None)),
            store,
            caller_agent_id,
            scoped_agents,
        ) if len(parts) == 2 else _print_agent_usage_error("agent step <ag-id>")
    if parts[0] == "run":
        if len(parts) < 2:
            return _print_agent_usage_error(
                "agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
            )
        return _cmd_agent_run(
            argparse.Namespace(
                agent_id=parts[1],
                steps=getattr(args, "steps", 3),
                max_workflows=getattr(args, "max_workflows", 2),
                no_stop_on_waiting=getattr(args, "no_stop_on_waiting", False),
                stop_on_idle=getattr(args, "stop_on_idle", False),
                no_stop_on_workflow_running=getattr(args, "no_stop_on_workflow_running", False),
                allow_action=getattr(args, "allow_action", []),
                no_confirm_workflows=getattr(args, "no_confirm_workflows", False),
                max_runtime_s=getattr(args, "max_runtime_s", None),
                workflow_compiler=getattr(args, "workflow_compiler", None),
                message_store=getattr(args, "message_store", None),
            ),
            store,
            caller_agent_id,
            scoped_agents,
        )
    target = parts[0]
    if not target.startswith("ag-"):
        action = parts[1] if len(parts) > 1 else None
        try:
            if action == "health":
                print(_format_runtime_agent_health(target, json_output=args.json))
            else:
                print(_format_runtime_agent(
                    target,
                    json_output=args.json,
                    brief=args.brief,
                ))
        except ValueError:
            print(f"error: agent not found: {target}", file=sys.stderr)
            return 2
        return 0
    if len(parts) != 1:
        print("error: usage: agent <ag-id>", file=sys.stderr)
        return 2
    try:
        agent = _runtime_access_for(
            store,
            scoped_agents,
            getattr(args, "message_store", None),
        ).read_agent(
            target,
            caller_agent_id=caller_agent_id,
        )
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_agent(
        agent,
        json_output=args.json,
        brief=args.brief,
    ))
    return 0

def _print_agent_usage_error(usage: str) -> int:
    print(f"error: usage: {usage}", file=sys.stderr)
    return 2

def _cmd_agent_assign(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    mission_path = Path(args.mission_file)
    if not mission_path.exists():
        print(f"error: mission file not found: {mission_path}", file=sys.stderr)
        return 2
    try:
        mission = mission_path.read_text(encoding="utf-8")
    except OSError:
        print(f"error: mission file not found: {mission_path}", file=sys.stderr)
        return 2
    try:
        agent = scoped_agents.assign_mission(caller_agent_id, args.agent_id, mission)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(agent.agent_id)
    return 0

def _cmd_agent_step(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    runner = MRnStepRunner(
        workflow_store=store,
        scoped_agent_store=scoped_agents,
        message_store=getattr(args, "message_store", None),
        workflow_compiler=getattr(args, "workflow_compiler", None),
    )
    try:
        result = runner.step(args.agent_id, caller_agent_id=caller_agent_id)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_mrn_step_result(result))
    return 0

def _cmd_agent_run(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    runner = MRnRunRunner(
        workflow_store=store,
        scoped_agent_store=scoped_agents,
        message_store=getattr(args, "message_store", None),
        workflow_compiler=getattr(args, "workflow_compiler", None),
    )
    policy = MRnRunPolicy(
        max_steps=args.steps,
        max_workflows_created=args.max_workflows,
        stop_on_waiting=not args.no_stop_on_waiting,
        stop_on_idle=bool(args.stop_on_idle),
        stop_on_workflow_running=not args.no_stop_on_workflow_running,
        allowed_actions=list(args.allow_action) if args.allow_action else None,
        require_confirmation_for_workflows=not args.no_confirm_workflows,
        max_runtime_s=args.max_runtime_s,
    )
    try:
        result = runner.run(args.agent_id, policy, caller_agent_id=caller_agent_id)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_mrn_run_result(result))
    return 0

def _cmd_agent_grant_scope(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    try:
        scoped_agents.get_visible_agent(caller_agent_id, args.agent_id)
        grant = scoped_agents.grant_scope(
            caller_agent_id,
            args.agent_id,
            args.path,
            reason=args.reason,
        )
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(grant.path)
    return 0

def _cmd_agent_revoke_scope(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    try:
        scoped_agents.get_visible_agent(caller_agent_id, args.agent_id)
        path = scoped_agents.revoke_scope(caller_agent_id, args.agent_id, args.path)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0

def _cmd_agent_scopes(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store
    try:
        agent = scoped_agents.get_visible_agent(caller_agent_id, args.agent_id)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_scope_grants(agent, json_output=args.json))
    return 0
