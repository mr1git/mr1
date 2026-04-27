"""
Deterministic workflow inspection and submission CLI.

`python -m mr1.workflow_cli <cmd>`

The CLI NEVER instantiates a scheduler. `submit` writes a workflow
directory to disk and exits; the MR1-owned scheduler picks it up on its
next tick. Read commands load state directly from the store and
pretty-print. `compile-workflow` is the one CLI command that may invoke
the workflow compiler agent.

Sub-commands:
    submit <path>                      write workflow spec to the store
    compile-workflow <path>           compile workflow request text into an envelope
    workflows                          list all workflows
    workflow <wf_id>                   show one workflow's tasks
    task <task_id>                     show one task's detail
    jobs                               list live tasks across all workflows
    events <wf_id> [--since T]         show events for a workflow
                   [--until T]
                   [--task TASK_ID]
                   [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from mr1.agents import AgentRegistry, default_agent_registry, run_agent_health
from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.dataflow import Artifact, ResolvedTaskInput, TaskOutput
from mr1.messages import MessageStore, PersistentMessage
from mr1.mrn_loop import MRnStepResult, MRnStepRunner
from mr1.mrn_run import MRnRunPolicy, MRnRunResult, MRnRunRunner
from mr1.scoped_agents import AgentScopeError, PersistentAgent, PersistentAgentStore
from mr1.scheduler import (
    WatcherTriggerError,
    WorkflowSpecError,
    append_workflow_on_disk,
    cancel_task_on_disk,
    cancel_workflow_on_disk,
    insert_workflow_on_disk,
    replace_workflow_on_disk,
    rerun_task_on_disk,
    submit_spec_to_disk,
    trigger_watcher_on_disk,
)
from mr1.tools import ToolRegistry, default_tool_registry
from mr1.workflow_compiler import WorkflowCompilerClient, WorkflowCompilerFailure
from mr1.workflow_schema import (
    WorkflowSchemaRegistry,
    default_workflow_schema_registry,
)
from mr1.workflow_models import (
    Provenance,
    Task,
    TaskStatus,
    Workflow,
    WorkflowStatus,
)
from mr1.workflow_store import WorkflowStore


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _short_ts(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    # "2026-04-20T14:30:05.123456+00:00" → "2026-04-20 14:30:05"
    return iso.replace("T", " ")[:19]


def _format_workflows_table(workflows: list[Workflow]) -> str:
    if not workflows:
        return "No workflows."
    rows = [("WORKFLOW_ID", "STATUS", "TITLE", "TASKS", "CREATED")]
    for wf in workflows:
        rows.append((
            wf.workflow_id,
            wf.status.value,
            wf.title[:40],
            str(len(wf.tasks)),
            _short_ts(wf.created_at),
        ))
    return _render_table(rows)


def _format_workflow_detail(wf: Workflow) -> str:
    lines = [
        f"workflow: {wf.workflow_id}",
        f"title:    {wf.title}",
        f"status:   {wf.status.value}",
        f"created:  {_short_ts(wf.created_at)}  by {wf.created_by.id if wf.created_by else '-'}",
    ]
    if wf.finished_at:
        lines.append(f"finished: {_short_ts(wf.finished_at)}")
    lines.append("")
    lines.append("tasks:")
    rows = [("LABEL", "TASK_ID", "STATUS", "POLICY", "DEPENDS_ON")]
    for label, tid in wf.label_to_task_id.items():
        task = wf.tasks.get(tid)
        if task is None:
            continue
        dep_labels = [
            _label_for_task_id(wf, parent_id) or parent_id[:10]
            for parent_id in task.depends_on
        ]
        rows.append((
            label,
            tid,
            task.status.value,
            task.dependency_policy,
            ",".join(dep_labels) or "-",
        ))
    lines.append(_render_table(rows, indent="  "))
    return "\n".join(lines)


def _format_task_detail(wf: Workflow, task: Task) -> str:
    lines = [
        f"task:       {task.task_id}",
        f"label:      {task.label}",
        f"title:      {task.title}",
        f"workflow:   {task.workflow_id} ({wf.title})",
        f"status:     {task.status.value}",
        f"kind:       {task.task_kind}",
        f"agent:      {task.agent_type or '-'}",
        f"dependency_policy: {task.dependency_policy}",
        f"depends_on: {', '.join(task.depends_on) or '-'}",
        f"created:    {_short_ts(task.created_at)}",
        f"started:    {_short_ts(task.started_at)}",
        f"finished:   {_short_ts(task.finished_at)}",
        f"pid:        {task.pid if task.pid is not None else '-'}",
        f"exit_code:  {task.exit_code if task.exit_code is not None else '-'}",
        f"attempts:   {task.attempt_count}",
        f"current:    {task.current_attempt or '-'}",
    ]
    if task.last_error:
        lines.append(f"last_error: {task.last_error}")
    if task.last_error_type:
        lines.append(f"error_type: {task.last_error_type}")
    if task.result_summary:
        lines.append(f"summary:    {task.result_summary[:200]}")
    if task.log_stdout_path:
        lines.append(f"stdout:     {task.log_stdout_path}")
    if task.log_stderr_path:
        lines.append(f"stderr:     {task.log_stderr_path}")
    if task.result_path:
        lines.append(f"result:     {task.result_path}")
    if task.output_path:
        lines.append(f"output:     {task.output_path}")
    if task.inputs_path:
        lines.append(f"inputs:     {task.inputs_path}")
    if task.materialized_prompt_path:
        lines.append(f"prompt:     {task.materialized_prompt_path}")
    if task.dataflow_error:
        lines.append(f"dataflow:   {task.dataflow_error}")
    lines.append(
        f"run_if:     {json.dumps(task.run_if, sort_keys=True) if task.run_if is not None else '-'}"
    )
    lines.append(
        "condition_result: "
        + (
            json.dumps(task.condition_result, sort_keys=True)
            if task.condition_result is not None else "-"
        )
    )
    lines.append(f"skip_reason: {task.skip_reason or '-'}")
    if task.status is TaskStatus.BLOCKED:
        lines.append(f"blocked_by:     {', '.join(task.blocked_by) or '-'}")
        lines.append(f"blocked_reason: {task.blocked_reason or '-'}")
        lines.append(f"blocked_at:     {_short_ts(task.blocked_at)}")
    if task.task_kind == "watcher":
        lines.extend([
            f"watcher:       {task.watcher_type or '-'}",
            f"watch_started: {_short_ts(task.watch_started_at)}",
            f"watch_done:    {_short_ts(task.watch_satisfied_at)}",
            f"last_checked:  {_short_ts(task.last_checked_at)}",
            f"last_result:   {(task.last_check_result or {}).get('message', '-')}",
            f"condition:     {json.dumps(task.condition, sort_keys=True) if task.condition is not None else '-'}",
        ])
    if task.task_kind == "tool":
        tool_config = json.dumps(task.tool_config, sort_keys=True)
        if len(tool_config) > 200:
            tool_config = tool_config[:197] + "..."
        lines.extend([
            f"tool:         {task.tool_type or '-'}",
            f"tool_config:  {tool_config}",
            f"tool_started: {_short_ts(task.tool_started_at)}",
            f"tool_done:    {_short_ts(task.tool_finished_at)}",
            f"tool_error:   {task.tool_error or '-'}",
        ])
    if task.attempts:
        lines.append("attempt_history:")
        rows = [("ATTEMPT", "STATUS", "STARTED", "FINISHED", "ERROR_TYPE")]
        for attempt in task.attempts:
            rows.append((
                str(attempt.attempt_id),
                attempt.status.value,
                _short_ts(attempt.started_at),
                _short_ts(attempt.finished_at),
                attempt.error_type or "-",
            ))
        lines.append(_render_table(rows, indent="  "))
    return "\n".join(lines)


def _format_jobs(workflows: list[Workflow]) -> str:
    live = {TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.WAITING}
    rows = [("WORKFLOW_ID", "TASK_ID", "LABEL", "STATUS", "PID")]
    for wf in workflows:
        for task in wf.tasks.values():
            if task.status not in live:
                continue
            rows.append((
                wf.workflow_id,
                task.task_id,
                task.label,
                task.status.value,
                str(task.pid) if task.pid is not None else "-",
            ))
    if len(rows) == 1:
        return "No live tasks."
    return _render_table(rows)


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


def _format_watchers(workflows: list[Workflow]) -> str:
    rows = [(
        "WORKFLOW_ID",
        "TASK_ID",
        "LABEL",
        "WATCHER",
        "STATUS",
        "LAST_CHECKED",
        "LAST_RESULT",
    )]
    for wf in workflows:
        for task in wf.tasks.values():
            if task.task_kind != "watcher" or task.is_terminal():
                continue
            last_result = (task.last_check_result or {}).get("message")
            rows.append((
                wf.workflow_id,
                task.task_id,
                task.label,
                task.watcher_type or "-",
                task.status.value,
                _short_ts(task.last_checked_at),
                (last_result or "-")[:60],
            ))
    if len(rows) == 1:
        return "No active watchers."
    return _render_table(rows)


def _format_result(task: Task, output: Optional[TaskOutput]) -> str:
    if output is None:
        return f"No normalized output for task: {task.task_id}"
    lines = [
        f"task:       {task.task_id}",
        f"label:      {task.label}",
        f"status:     {output.status}",
        f"summary:    {output.summary or '-'}",
        "text:",
        output.text or "",
        "",
        "data:",
        json.dumps(output.data, indent=2, sort_keys=True),
        "",
        "metrics:",
        json.dumps(output.metrics, indent=2, sort_keys=True),
        "",
        "artifacts:",
    ]
    if output.artifacts:
        rows = [("NAME", "KIND", "PATH")]
        for artifact in output.artifacts:
            rows.append((artifact.name, artifact.kind, artifact.path))
        lines.append(_render_table(rows, indent="  "))
    else:
        lines.append("  none")
    return "\n".join(lines)


def _format_inputs(task: Task, inputs: Optional[list[ResolvedTaskInput]]) -> str:
    if not inputs:
        return f"No materialized inputs for task: {task.task_id}"
    lines = [f"task:       {task.task_id}", f"label:      {task.label}", "inputs:"]
    for item in inputs:
        lines.extend([
            f"  - name:   {item.name}",
            f"    source: {item.source}",
            f"    type:   {item.resolved_type}",
            f"    value:  {_format_inline_value(item)}",
        ])
    return "\n".join(lines)


def _format_artifacts(workflow: Workflow) -> str:
    artifacts: list[tuple[str, Artifact]] = []
    for task in workflow.tasks.values():
        for artifact in task.artifacts:
            artifacts.append((task.label, artifact))
    if not artifacts:
        return f"No artifacts registered in workflow: {workflow.workflow_id}"
    rows = [("TASK", "NAME", "KIND", "PATH")]
    for label, artifact in artifacts:
        rows.append((label, artifact.name, artifact.kind, artifact.path))
    return _render_table(rows)


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


def _describe_schema_section(
    section: Optional[str],
    registry: Optional[WorkflowSchemaRegistry] = None,
) -> dict[str, Any]:
    active_registry = registry or default_workflow_schema_registry()
    if section is None:
        return active_registry.describe_all()
    if section == "workflow":
        return active_registry.describe_workflow()
    if section == "task":
        return active_registry.describe_task()
    if section == "inputs":
        return active_registry.describe_inputs()
    if section == "refs":
        return active_registry.describe_references()
    if section == "task-kinds":
        return active_registry.describe_task_kinds()
    raise ValueError(f"schema section not found: {section}")


def _brief_schema_view(section: Optional[str], description: dict[str, Any]) -> dict[str, Any]:
    if section is None:
        return {
            "workflow": description["workflow"]["summary"],
            "task": description["task"]["summary"],
            "inputs": {
                "summary": description["inputs"]["summary"],
                "item_shape": description["inputs"]["item_shape"],
                "rules": [
                    "inputs must be a list of objects",
                    "each input object must include non-empty name and from",
                    "inputs must NEVER be strings",
                ],
            },
            "refs": description["refs"]["summary"],
            "task-kinds": description["task-kinds"]["summary"],
        }
    if section == "inputs":
        return {
            "summary": description["summary"],
            "item_shape": description["item_shape"],
            "rules": [
                "inputs must be a list of objects",
                "each input object must include non-empty name and from",
                "inputs must NEVER be strings",
                "inputs must reference upstream dependencies or ancestors",
            ],
        }
    keys = ("summary", "required", "shape", "fields", "supported_patterns", "agent", "tool", "watcher")
    return {key: description[key] for key in keys if key in description}


def _format_schema(
    section: Optional[str] = None,
    registry: Optional[WorkflowSchemaRegistry] = None,
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    description = _describe_schema_section(section, registry=registry)
    view = _brief_schema_view(section, description) if brief else description
    return json.dumps(view, indent=2, sort_keys=True)


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
    result = run_agent_health(agent_name, registry=active_registry)
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
) -> dict[str, Any]:
    payload = agent.to_dict()
    last_run = agent.last_run or {}
    payload["latest_run_id"] = last_run.get("run_id")
    payload["latest_run_stopped_reason"] = last_run.get("stopped_reason")
    payload["latest_run_step_count"] = last_run.get("step_count")
    payload["latest_run_at"] = last_run.get("finished_at")
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
    json_output: bool = False,
    brief: bool = False,
) -> str:
    payload = _persistent_agent_payload(agent, reports=reports, message_store=message_store)
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
        f"agent_id:     {agent.agent_id}",
        f"type:         {agent.agent_type}",
        f"title:        {agent.title}",
        f"status:       {agent.status}",
        f"mission:      {_compact_text(agent.mission)}",
        f"mode:         {agent.mode}",
        f"run_status:   {agent.run_status}",
        f"iteration:    {agent.current_iteration}",
        f"last_step_at: {_short_ts(agent.last_step_at)}",
        f"last_action:  {_summarize_last_action(agent.last_action)}",
        f"latest_run:   {payload.get('latest_run_id') or '-'}",
        f"run_reason:   {payload.get('latest_run_stopped_reason') or '-'}",
        f"run_steps:    {payload.get('latest_run_step_count') or 0}",
        f"run_at:       {_short_ts(payload.get('latest_run_at'))}",
        f"parent_req:   {_compact_text(agent.parent_request)}",
        f"tree_level:   {agent.tree_level}",
        f"parent:       {agent.parent_agent_id or '-'}",
        f"created_at:   {agent.created_at}",
        f"workflows:    {len(agent.owned_workflow_ids)}",
        f"unread_inbox: {payload.get('unread_inbox_count', 0)}",
    ]
    if agent.owned_workflow_ids:
        lines.append(f"owned_ids:    {', '.join(agent.owned_workflow_ids)}")
    lines.append("reports:")
    for path in reports or []:
        lines.append(f"  {path.name}")
    if not reports:
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
    return "\n".join(lines)


def _compact_text(text: Optional[str], *, limit: int = 120) -> str:
    if not text:
        return "-"
    normalized = " ".join(text.split())
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


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
    if json_output:
        return json.dumps(message.to_dict(), indent=2, sort_keys=True)
    return "\n".join([
        f"message_id:   {message.message_id}",
        f"from:         {message.from_agent_id}",
        f"to:           {message.to_agent_id}",
        f"kind:         {message.kind}",
        f"subject:      {message.subject}",
        f"status:       {message.status}",
        f"workflow_id:  {message.workflow_id or '-'}",
        f"task_id:      {message.task_id or '-'}",
        f"created_at:   {message.created_at}",
        f"read_at:      {message.read_at or '-'}",
        f"archived_at:  {message.archived_at or '-'}",
        "body:",
        message.body,
    ])


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
    message = message_store.get_message(message_id)
    if message is None:
        raise ValueError(f"message not found: {message_id}")
    if not message_store.can_agent_access_message(caller_agent_id, message):
        raise AgentScopeError("access denied: message not in agent scope")
    return message


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


def _brief_description(description: dict[str, Any]) -> dict[str, str]:
    return {
        "name": description["name"],
        "type": description["type"],
        "description": description["description"],
    }


def _select_description_view(
    description: dict[str, Any],
    *,
    example_only: bool,
    brief: bool,
) -> dict[str, Any]:
    if example_only and brief:
        raise ValueError("invalid flag combination")
    if example_only:
        examples = list(description.get("examples") or [])
        return examples[0] if examples else {}
    if brief:
        return _brief_description(description)
    return description


def _format_description_text(description: dict[str, Any]) -> str:
    lines = [
        f"name:         {description['name']}",
        f"type:         {description['type']}",
        f"description:  {description['description']}",
        "inputs:",
        json.dumps(description.get("inputs", {}), indent=2, sort_keys=True),
        "outputs:",
        json.dumps(description.get("outputs", {}), indent=2, sort_keys=True),
        "config_schema:",
        json.dumps(description.get("config_schema", {}), indent=2, sort_keys=True),
    ]
    if "runtime" in description:
        lines.extend([
            "runtime:",
            json.dumps(description.get("runtime", {}), indent=2, sort_keys=True),
        ])
    if "workflow_task_allowed" in description:
        lines.append(
            f"workflow_task_allowed: {bool(description.get('workflow_task_allowed'))}"
        )
    lines.extend([
        "examples:",
        json.dumps(description.get("examples", []), indent=2, sort_keys=True),
    ])
    return "\n".join(lines)


def _config_shape_for_tool(registry: ToolRegistry, tool_type: str) -> str:
    for tool in registry.list_tools():
        if tool.tool_type == tool_type:
            return tool.config_shape
    return "-"


def _render_table(rows: list[tuple[str, ...]], indent: str = "") -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for row in rows:
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        out.append(indent + "  ".join(cells).rstrip())
    return "\n".join(out)


def _label_for_task_id(wf: Workflow, task_id: str) -> Optional[str]:
    for label, tid in wf.label_to_task_id.items():
        if tid == task_id:
            return label
    return None


# ---------------------------------------------------------------------------
# Store-level helpers
# ---------------------------------------------------------------------------


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
    workflows = []
    for workflow in store.list_workflows():
        workflow = scoped_agents.normalize_workflow_ownership(workflow)
        if scoped_agents.can_agent_access_workflow(caller_agent_id, workflow):
            workflows.append(workflow)
    return workflows


def _load_scoped_workflow(
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


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def _cmd_submit(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    spec, error = _load_json_file(args.path)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        wf_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            store,
            owner_agent_id=caller_agent_id,
            caller_agent_id=caller_agent_id,
            scoped_agent_store=scoped_agents,
            tool_registry=default_tool_registry(),
        )
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(wf_id)
    return 0


def _cmd_compile_workflow(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    request_text, error = _load_text_file(args.path)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    owner_agent_id = args.owner or caller_agent_id

    def _submitter(spec: dict[str, Any], submit_caller_agent_id: str, submit_owner_agent_id: str) -> tuple[str, str]:
        wf_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            store,
            owner_agent_id=submit_owner_agent_id,
            caller_agent_id=submit_caller_agent_id,
            scoped_agent_store=scoped_agents,
            tool_registry=default_tool_registry(),
        )
        return wf_id, wf_id

    client = WorkflowCompilerClient(
        compiler=getattr(args, "workflow_compiler", None),
        scoped_agent_store=scoped_agents,
        tool_registry=default_tool_registry(),
        submitter=_submitter if args.submit else None,
    )
    try:
        result = client.compile(
            request=request_text,
            context="CLI compile-workflow request",
            caller_agent_id=caller_agent_id,
            owner_agent_id=owner_agent_id,
            mode="submit_if_valid" if args.submit else "preview_only",
        )
    except (WorkflowCompilerFailure, WorkflowSpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_rerun(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
        task_id = rerun_task_on_disk(
            store,
            args.workflow_id,
            args.task_label_or_id,
            agent_id="cli",
        )
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(task_id)
    return 0


def _cmd_cancel_task(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    visible = _visible_workflows(store, scoped_agents, caller_agent_id)
    wf, task = _find_workflow_for_task(store, args.task_id, workflows=visible)
    if wf is None or task is None:
        print(f"error: task not found: {args.task_id}", file=sys.stderr)
        return 2
    try:
        task_id = cancel_task_on_disk(store, task.task_id, agent_id="cli")
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(task_id)
    return 0


def _cmd_cancel_workflow(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
        workflow_id = cancel_workflow_on_disk(
            store,
            args.workflow_id,
            agent_id="cli",
        )
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(workflow_id)
    return 0


def _cmd_append_workflow(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    spec, error = _load_json_file(args.path)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
        workflow_id = append_workflow_on_disk(
            store,
            args.workflow_id,
            spec,
            agent_id="cli",
        )
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(workflow_id)
    return 0


def _cmd_insert_workflow(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    spec, error = _load_json_file(args.path)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
        workflow_id = insert_workflow_on_disk(
            store,
            args.workflow_id,
            args.after_task,
            spec,
            agent_id="cli",
        )
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(workflow_id)
    return 0


def _cmd_replace_workflow(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    spec, error = _load_json_file(args.path)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
        workflow_id = replace_workflow_on_disk(
            store,
            args.workflow_id,
            args.task_label_or_id,
            spec,
            agent_id="cli",
        )
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.rerun:
        from mr1.scheduler import Scheduler

        scheduler = Scheduler(
            store,
            auto_tick=False,
            agent_id="cli",
            scoped_agent_store=scoped_agents,
        )
        try:
            scheduler.tick()
        finally:
            scheduler.shutdown()
    print(workflow_id)
    return 0


def _cmd_workflows(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    print(_format_workflows_table(_visible_workflows(store, scoped_agents, caller_agent_id)))
    return 0


def _cmd_workflow(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    try:
        wf = _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_workflow_detail(wf))
    return 0


def _cmd_task(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    wf, task = _find_workflow_for_task(
        store,
        args.task_id,
        workflows=_visible_workflows(store, scoped_agents, caller_agent_id),
    )
    if wf is None or task is None:
        print(f"error: task not found: {args.task_id}", file=sys.stderr)
        return 2
    print(_format_task_detail(wf, task))
    return 0


def _cmd_jobs(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    print(_format_jobs(_visible_workflows(store, scoped_agents, caller_agent_id)))
    return 0


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


def _cmd_watchers(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    print(_format_watchers(_visible_workflows(store, scoped_agents, caller_agent_id)))
    return 0


def _reject_invalid_flag_combination(args: argparse.Namespace) -> Optional[int]:
    if getattr(args, "example", False) and getattr(args, "brief", False):
        print("error: invalid flag combination", file=sys.stderr)
        return 2
    return None


def _cmd_capabilities(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
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
    scoped_agents: PersistentAgentStore,
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


def _cmd_tools(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    rc = _reject_invalid_flag_combination(args)
    if rc is not None:
        return rc
    print(_format_tools(json_output=args.json, brief=args.brief))
    return 0


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


def _cmd_schema(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    try:
        print(_format_schema(
            args.section,
            json_output=args.json,
            brief=args.brief,
        ))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_tool(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
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
        agent = scoped_agents.get_visible_agent(caller_agent_id, target)
    except (ValueError, AgentScopeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_agent(
        agent,
        reports=scoped_agents.list_reports(agent.agent_id),
        message_store=getattr(args, "message_store", None),
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
    del store, scoped_agents
    message_store = getattr(args, "message_store")
    try:
        message = _require_message(message_store, args.message_id, caller_agent_id)
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
        message = message_store.mark_read(args.message_id)
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


def _cmd_result(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    wf, task = _find_workflow_for_task(
        store,
        args.task_id,
        workflows=_visible_workflows(store, scoped_agents, caller_agent_id),
    )
    if wf is None or task is None:
        print(f"error: task not found: {args.task_id}", file=sys.stderr)
        return 2
    print(_format_result(task, store.load_task_output(wf.workflow_id, task.task_id)))
    return 0


def _cmd_inputs(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    wf, task = _find_workflow_for_task(
        store,
        args.task_id,
        workflows=_visible_workflows(store, scoped_agents, caller_agent_id),
    )
    if wf is None or task is None:
        print(f"error: task not found: {args.task_id}", file=sys.stderr)
        return 2
    print(_format_inputs(task, store.load_task_inputs(wf.workflow_id, task.task_id)))
    return 0


def _cmd_artifacts(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    try:
        wf = _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_artifacts(wf))
    return 0


def _cmd_trigger(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    try:
        _load_scoped_workflow(store, args.workflow_id, scoped_agents, caller_agent_id)
        task_id = trigger_watcher_on_disk(
            store,
            args.workflow_id,
            args.label_or_task_id,
            event_name=args.event_name,
            agent_id="cli",
        )
    except (WatcherTriggerError, WorkflowSpecError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(task_id)
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mr1.workflow_cli",
        description="Submit and inspect MR1 workflows without an LLM in the loop.",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=None,
        help="Override the workflow store root (defaults to mr1/memory/workflows).",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(subparser: argparse.ArgumentParser, *, include_example: bool) -> None:
        subparser.add_argument("--json", action="store_true", dest="json")
        subparser.add_argument("--brief", action="store_true", dest="brief")
        if include_example:
            subparser.add_argument("--example", action="store_true", dest="example")
        else:
            subparser.set_defaults(example=False)

    p_submit = subs.add_parser("submit", help="Write a workflow spec to the store.")
    p_submit.add_argument("path", help="Path to a workflow JSON spec.")
    p_submit.set_defaults(func=_cmd_submit)

    p_compile = subs.add_parser(
        "compile-workflow",
        help="Compile a workflow request file into a validated envelope.",
    )
    p_compile.add_argument("path", help="Path to a text file containing the workflow request.")
    p_compile.add_argument("--owner", default=None)
    p_compile.add_argument("--submit", action="store_true")
    p_compile.set_defaults(func=_cmd_compile_workflow)

    p_rerun = subs.add_parser("rerun", help="Rerun one task in a workflow.")
    p_rerun.add_argument("workflow_id")
    p_rerun.add_argument("task_label_or_id")
    p_rerun.set_defaults(func=_cmd_rerun)

    p_list = subs.add_parser("workflows", help="List all workflows.")
    p_list.set_defaults(func=_cmd_workflows)

    p_wf = subs.add_parser("workflow", help="Show one workflow's detail.")
    p_wf.add_argument("workflow_id")
    p_wf.set_defaults(func=_cmd_workflow)

    p_task = subs.add_parser("task", help="Show one task's detail.")
    p_task.add_argument("task_id")
    p_task.set_defaults(func=_cmd_task)

    p_cancel_task = subs.add_parser("cancel-task", help="Cancel one task.")
    p_cancel_task.add_argument("task_id")
    p_cancel_task.set_defaults(func=_cmd_cancel_task)

    p_cancel_workflow = subs.add_parser("cancel-workflow", help="Cancel one workflow.")
    p_cancel_workflow.add_argument("workflow_id")
    p_cancel_workflow.set_defaults(func=_cmd_cancel_workflow)

    p_jobs = subs.add_parser("jobs", help="List live tasks across all workflows.")
    p_jobs.set_defaults(func=_cmd_jobs)

    p_events = subs.add_parser("events", help="Show events for a workflow.")
    p_events.add_argument("workflow_id")
    p_events.add_argument("--since", default=None)
    p_events.add_argument("--until", default=None)
    p_events.add_argument("--task", default=None, dest="task")
    p_events.add_argument("--limit", type=int, default=None)
    p_events.set_defaults(func=_cmd_events)

    p_watchers = subs.add_parser("watchers", help="List active watcher tasks.")
    p_watchers.set_defaults(func=_cmd_watchers)

    p_capabilities = subs.add_parser("capabilities", help="List registered capabilities.")
    add_common_flags(p_capabilities, include_example=False)
    p_capabilities.set_defaults(func=_cmd_capabilities)

    p_capability = subs.add_parser("capability", help="Show one capability description.")
    p_capability.add_argument("name")
    add_common_flags(p_capability, include_example=True)
    p_capability.set_defaults(func=_cmd_capability)

    p_tools = subs.add_parser("tools", help="List registered deterministic workflow tools.")
    add_common_flags(p_tools, include_example=False)
    p_tools.set_defaults(func=_cmd_tools)

    p_agents = subs.add_parser("agents", help="List persistent scoped agents.")
    add_common_flags(p_agents, include_example=False)
    p_agents.set_defaults(func=_cmd_agents)

    p_schema = subs.add_parser("schema", help="Show workflow schema metadata.")
    p_schema.add_argument("section", nargs="?")
    add_common_flags(p_schema, include_example=False)
    p_schema.set_defaults(func=_cmd_schema)

    p_tool = subs.add_parser("tool", help="Show one tool description.")
    p_tool.add_argument("tool_type")
    add_common_flags(p_tool, include_example=True)
    p_tool.set_defaults(func=_cmd_tool)

    p_agent = subs.add_parser("agent", help="Manage scoped agents or inspect runtime agent profiles.")
    p_agent.add_argument("parts", nargs="+")
    add_common_flags(p_agent, include_example=False)
    p_agent.set_defaults(func=_cmd_agent)

    p_agent_assign = subs.add_parser("agent-assign", help="Assign a mission file to a persistent scoped agent.")
    p_agent_assign.add_argument("agent_id")
    p_agent_assign.add_argument("mission_file")
    p_agent_assign.set_defaults(func=_cmd_agent_assign)

    p_agent_step = subs.add_parser("agent-step", help="Run one bounded MRn step for a persistent scoped agent.")
    p_agent_step.add_argument("agent_id")
    p_agent_step.set_defaults(func=_cmd_agent_step)

    p_agent_run = subs.add_parser("agent-run", help="Run a bounded multi-step MRn run for a persistent scoped agent.")
    p_agent_run.add_argument("agent_id")
    p_agent_run.add_argument("--steps", type=int, default=3)
    p_agent_run.add_argument("--max-workflows", type=int, default=2)
    p_agent_run.add_argument("--no-stop-on-waiting", action="store_true")
    p_agent_run.add_argument("--stop-on-idle", action="store_true")
    p_agent_run.add_argument("--no-stop-on-workflow-running", action="store_true")
    p_agent_run.add_argument("--allow-action", action="append", default=[])
    p_agent_run.add_argument("--no-confirm-workflows", action="store_true")
    p_agent_run.add_argument("--max-runtime-s", type=int, default=None)
    p_agent_run.set_defaults(func=_cmd_agent_run)

    p_inbox = subs.add_parser("inbox", help="List inbox messages for an agent.")
    p_inbox.add_argument("--agent", default=None)
    p_inbox.add_argument("--archived", action="store_true")
    add_common_flags(p_inbox, include_example=False)
    p_inbox.set_defaults(func=_cmd_inbox)

    p_outbox = subs.add_parser("outbox", help="List outbox messages for an agent.")
    p_outbox.add_argument("--agent", default=None)
    p_outbox.add_argument("--archived", action="store_true")
    add_common_flags(p_outbox, include_example=False)
    p_outbox.set_defaults(func=_cmd_outbox)

    p_message = subs.add_parser("message", help="Show one message.")
    p_message.add_argument("message_id")
    add_common_flags(p_message, include_example=False)
    p_message.set_defaults(func=_cmd_message)

    p_message_read = subs.add_parser("message-read", help="Mark one message as read.")
    p_message_read.add_argument("message_id")
    p_message_read.set_defaults(func=_cmd_message_read)

    p_message_archive = subs.add_parser("message-archive", help="Archive one message.")
    p_message_archive.add_argument("message_id")
    p_message_archive.set_defaults(func=_cmd_message_archive)

    p_message_send = subs.add_parser("message-send", help="Send one persistent message.")
    p_message_send.add_argument("to_agent_id")
    p_message_send.add_argument("subject")
    p_message_send.add_argument("body_file")
    p_message_send.add_argument(
        "--kind",
        choices=["report", "question", "alert", "status", "request"],
        default="request",
    )
    p_message_send.set_defaults(func=_cmd_message_send)

    p_result = subs.add_parser("result", help="Show normalized task output.")
    p_result.add_argument("task_id")
    p_result.set_defaults(func=_cmd_result)

    p_inputs = subs.add_parser("inputs", help="Show materialized task inputs.")
    p_inputs.add_argument("task_id")
    p_inputs.set_defaults(func=_cmd_inputs)

    p_artifacts = subs.add_parser("artifacts", help="List artifacts for a workflow.")
    p_artifacts.add_argument("workflow_id")
    p_artifacts.set_defaults(func=_cmd_artifacts)

    p_trigger = subs.add_parser("trigger", help="Trigger a manual_event watcher.")
    p_trigger.add_argument("workflow_id")
    p_trigger.add_argument("label_or_task_id")
    p_trigger.add_argument("event_name", nargs="?")
    p_trigger.set_defaults(func=_cmd_trigger)

    p_append_workflow = subs.add_parser("append-workflow", help="Append task(s) to a workflow.")
    p_append_workflow.add_argument("workflow_id")
    p_append_workflow.add_argument("path")
    p_append_workflow.set_defaults(func=_cmd_append_workflow)

    p_insert_workflow = subs.add_parser("insert-workflow", help="Insert one task after an existing task.")
    p_insert_workflow.add_argument("workflow_id")
    p_insert_workflow.add_argument("after_task")
    p_insert_workflow.add_argument("path")
    p_insert_workflow.set_defaults(func=_cmd_insert_workflow)

    p_replace_workflow = subs.add_parser("replace-workflow", help="Replace one task in a workflow.")
    p_replace_workflow.add_argument("-r", "--rerun", action="store_true")
    p_replace_workflow.add_argument("workflow_id")
    p_replace_workflow.add_argument("task_label_or_id")
    p_replace_workflow.add_argument("path")
    p_replace_workflow.set_defaults(func=_cmd_replace_workflow)

    return parser


def main(
    argv: Optional[list[str]] = None,
    *,
    store: Optional[WorkflowStore] = None,
    caller_agent_id: Optional[str] = None,
    scoped_agent_store: Optional[PersistentAgentStore] = None,
    message_store: Optional[MessageStore] = None,
    workflow_compiler: Optional[Any] = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    active_store = store if store is not None else WorkflowStore(root=args.store_root)
    active_scoped_store = scoped_agent_store or PersistentAgentStore(
        root=active_store.root.parent / "agents"
    )
    active_message_store = message_store or MessageStore(
        root=active_store.root.parent / "messages",
        scoped_agent_store=active_scoped_store,
    )
    resolved_caller_agent_id = caller_agent_id or active_scoped_store.root_agent_id
    setattr(args, "workflow_compiler", workflow_compiler)
    setattr(args, "message_store", active_message_store)
    return args.func(args, active_store, resolved_caller_agent_id, active_scoped_store)


def _format_inline_value(item: ResolvedTaskInput) -> str:
    if item.resolved_type == "artifact":
        return item.artifact_path or "-"
    if item.value is None:
        return "-"
    if isinstance(item.value, str):
        compact = item.value.replace("\n", "\\n")
        return compact[:120] + ("..." if len(compact) > 120 else "")
    return json.dumps(item.value, sort_keys=True)[:120]


if __name__ == "__main__":
    sys.exit(main())
