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
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any, Optional

from mr1.agents import AgentRegistry, default_agent_registry, run_agent_health
from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.capability_policy import (
    CapabilityApprovalDecision,
    CapabilityApprovalRequest,
    CapabilityApprovalStore,
)
from mr1.dataflow import Artifact, ResolvedTaskInput, TaskOutput
from mr1.doctor import create_snapshot, filter_doctor_report, inspect_snapshot, list_snapshots, run_doctor
from mr1.event_log import EventLog, SystemEvent, bind_correlation_id, cli_correlation_id
from mr1.inbox_triage import InboxTriagePolicy, InboxTriageResult, InboxTriageRunner
from mr1.messages import MessageStore, PersistentMessage
from mr1.memory_graph import (
    MemoryGraph,
    MemoryGraphStore,
    file_summary,
    graph_stats,
    project_summary,
    show_node,
    update_graph_from_events,
    workflow_template_summary,
)
from mr1.memory_curator import (
    InsightStore,
    MemoryCurationRun,
    MemoryInsight,
    build_memory_curation_bundle,
    evaluate_memory_curation_due,
    run_memory_curation,
)
from mr1.memory_feedback import (
    InsightFeedback,
    build_memory_maintenance_spec,
    evaluate_memory_feedback_due,
    maintenance_status_payload,
    submit_memory_maintenance_workflow,
    update_insight_feedback,
)
from mr1.memory_queries import (
    list_filtered_insights,
    memory_search,
    memory_graph_agent_summary,
    memory_graph_capabilities,
    memory_graph_failures,
    memory_graph_top_workflows,
    memory_insight_show,
    memory_insights_search,
)
from mr1.memory_retrieval import RetrievalStore, update_memory_retrieval
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
        f"clearance:    {agent.security_clearance:.2f}",
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
    lines.append(f"scope_roots:  {', '.join(agent.scope_roots) or '-'}")
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
    if "risk_score" in description:
        lines.extend([
            f"risk_score:      {description.get('risk_score')}",
            f"direct_allowed:  {bool(description.get('direct_allowed'))}",
            f"workflow_allowed:{bool(description.get('workflow_allowed'))}",
            f"requires_scope:  {bool(description.get('requires_scope'))}",
            f"is_filesystem:   {bool(description.get('is_filesystem'))}",
            f"is_execution:    {bool(description.get('is_execution'))}",
            f"path_arg_fields: {', '.join(description.get('path_arg_fields') or []) or '-'}",
        ])
    lines.extend([
        "examples:",
        json.dumps(description.get("examples", []), indent=2, sort_keys=True),
    ])
    return "\n".join(lines)


def _format_memory_update(result: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(result, indent=2, sort_keys=True)
    rows = [
        ("FIELD", "VALUE"),
        ("processed_events", str(result.get("processed_events", 0))),
        ("last_processed_event_index", str(result.get("last_processed_event_index", 0))),
        ("nodes_created", str(result.get("nodes_created", 0))),
        ("nodes_updated", str(result.get("nodes_updated", 0))),
        ("edges_created", str(result.get("edges_created", 0))),
        ("edges_updated", str(result.get("edges_updated", 0))),
    ]
    return _render_table(rows)


def _format_memory_stats(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        f"nodes: {payload.get('node_count', 0)}",
        f"edges: {payload.get('edge_count', 0)}",
        f"last_processed_event_index: {payload.get('last_processed_event_index', 0)}",
        "",
        "node_types:",
    ]
    node_rows = [("TYPE", "COUNT")]
    for name, count in sorted(dict(payload.get("node_types", {})).items()):
        node_rows.append((name, str(count)))
    lines.append(_render_table(node_rows, indent="  "))
    lines.append("")
    lines.append("edge_types:")
    edge_rows = [("TYPE", "COUNT")]
    for name, count in sorted(dict(payload.get("edge_types", {})).items()):
        edge_rows.append((name, str(count)))
    lines.append(_render_table(edge_rows, indent="  "))
    return "\n".join(lines)


def _format_memory_templates(items: list[dict[str, Any]], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(items, indent=2, sort_keys=True)
    if not items:
        return "No workflow templates."
    rows = [("TEMPLATE_ID", "NAME", "CRED", "SUCCESS", "FAIL", "BLOCKED")]
    for item in items:
        stats = item.get("stats", {})
        rows.append((
            item["node_id"],
            item["name"][:24],
            f"{float(stats.get('credibility_score', 0.0)):.3f}",
            str(int(stats.get("success_count", 0))),
            str(int(stats.get("failure_count", 0))),
            str(int(stats.get("blocked_count", 0))),
        ))
    return _render_table(rows)


def _format_memory_capabilities(items: list[dict[str, Any]], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(items, indent=2, sort_keys=True)
    if not items:
        return "No capabilities."
    rows = [("CAPABILITY_ID", "NAME", "RELIAB", "REQUESTS", "EXEC", "BLOCKED", "FAIL")]
    for item in items:
        stats = item.get("stats", {})
        rows.append((
            item["node_id"],
            item["name"][:24],
            f"{float(stats.get('reliability_score', 0.0)):.3f}",
            str(int(stats.get("request_count", 0))),
            str(int(stats.get("execution_count", 0))),
            str(int(stats.get("blocked_count", 0))),
            str(int(stats.get("failure_count", 0))),
        ))
    return _render_table(rows)


def _format_memory_failures(items: list[dict[str, Any]], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(items, indent=2, sort_keys=True)
    if not items:
        return "No failure modes."
    rows = [("FAILURE_ID", "NAME", "COUNT", "STATUS", "TYPE")]
    for item in items:
        stats = item.get("stats", {})
        metadata = item.get("metadata", {})
        rows.append((
            item["node_id"],
            item["name"][:32],
            str(int(stats.get("occurrence_count", 0))),
            str(metadata.get("status") or "-"),
            str(metadata.get("error_type") or "-"),
        ))
    return _render_table(rows)


def _format_memory_detail(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(payload, indent=2, sort_keys=True)


def _format_memory_retrieval_stats(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        f"retrieval_ready: {bool(payload.get('retrieval_ready'))}",
        f"document_count: {int(payload.get('document_count', 0))}",
        f"schema_version: {payload.get('schema_version') or '-'}",
        f"updated_at: {payload.get('updated_at') or '-'}",
        "",
        "doc_types:",
    ]
    doc_rows = [("TYPE", "COUNT")]
    for name, count in sorted(dict(payload.get("doc_type_counts", {})).items()):
        doc_rows.append((name, str(count)))
    lines.append(_render_table(doc_rows, indent="  "))
    lines.append("")
    lines.append("source_counts:")
    source_rows = [("SOURCE", "COUNT")]
    for name, count in sorted(dict(payload.get("source_counts", {})).items()):
        source_rows.append((name, str(count)))
    lines.append(_render_table(source_rows, indent="  "))
    return "\n".join(lines)


def _format_memory_retrieval_search(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    items = list(payload.get("items", []))
    if not items:
        return "No retrieval documents."
    rows = [("DOC_ID", "TYPE", "SCORE", "TITLE", "SUMMARY")]
    for item in items:
        rows.append((
            str(item.get("doc_id") or ""),
            str(item.get("doc_type") or ""),
            f"{float(item.get('score', 0.0)):.2f}",
            str(item.get("title") or "")[:36],
            str(item.get("summary") or "")[:56],
        ))
    return _render_table(rows)


def _format_memory_curation_due(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    rows = [
        ("FIELD", "VALUE"),
        ("due", str(bool(payload.get("due")))),
        ("latest_event_index", str(payload.get("latest_event_index", 0))),
        ("last_curated_event_index", str(payload.get("last_curated_event_index", 0))),
        ("important_event_count", str(payload.get("important_event_count", 0))),
        ("important_event_types", ", ".join(payload.get("important_event_types", [])) or "-"),
        ("suggested_event_window", json.dumps(payload.get("suggested_event_window", []))),
    ]
    return _render_table(rows)


def _format_compile_memory_summary(result: dict[str, Any] | Any) -> str:
    if hasattr(result, "compiled_with_memory"):
        compiled_with_memory = bool(result.compiled_with_memory)
        memory_tools_used = list(result.memory_tools_used or [])
        memory_refs_used = list(result.envelope.memory_refs_used)
        memory_context = result.memory_context_summary
        warnings = list(result.memory_ref_warnings or [])
    else:
        compiled_with_memory = bool(result.get("compiled_with_memory"))
        memory_tools_used = list(result.get("memory_tools_used", []))
        memory_refs_used = list(result.get("memory_refs_used", []))
        memory_context = str(result.get("memory_context_summary") or "")
        warnings = list(result.get("memory_ref_warnings", []))
    lines = [
        f"memory_enabled: {compiled_with_memory}",
        f"memory_tools_used: {', '.join(memory_tools_used) if memory_tools_used else '-'}",
        f"memory_refs_used: {', '.join(memory_refs_used) if memory_refs_used else '-'}",
    ]
    if memory_context:
        lines.append(f"memory_context_summary: {memory_context}")
    if warnings:
        lines.append(f"memory_ref_warnings: {', '.join(warnings)}")
    return "\n".join(lines)


def _format_memory_insights(
    items: list[MemoryInsight],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in items]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not items:
        return "No insights."
    rows = [("INSIGHT_ID", "TYPE", "SEVERITY", "STATUS", "CONF", "UPDATED", "TITLE")]
    for item in items:
        rows.append((
            item.insight_id,
            item.insight_type,
            item.severity,
            item.status,
            f"{float(item.confidence):.2f}",
            _short_ts(item.updated_at),
            item.title[:48],
        ))
    return _render_table(rows)


def _format_memory_insight(
    item: MemoryInsight,
    *,
    json_output: bool = False,
) -> str:
    payload = item.to_dict()
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(payload, indent=2, sort_keys=True)


def _format_memory_curation_runs(
    runs: list[MemoryCurationRun],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in runs]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not runs:
        return "No curation runs."
    rows = [("RUN_ID", "STATUS", "WINDOW", "OUTPUTS", "ERRORS", "STARTED")]
    for item in runs:
        rows.append((
            item.run_id,
            item.status,
            f"{item.event_start_index}-{item.event_end_index}",
            str(len(item.output_insight_ids)),
            str(len(item.errors)),
            _short_ts(item.started_at),
        ))
    return _render_table(rows)


def _format_memory_feedback(
    items: list[InsightFeedback],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in items]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not items:
        return "No feedback."
    rows = [("FEEDBACK_ID", "INSIGHT_ID", "OUTCOME", "DELTA", "EVENT_TYPE", "CREATED")]
    for item in items:
        rows.append((
            item.feedback_id,
            item.insight_id,
            item.outcome,
            f"{float(item.confidence_delta):+.2f}",
            str(item.metadata.get("event_type") or "-"),
            _short_ts(item.created_at),
        ))
    return _render_table(rows)


def _format_memory_insight_effectiveness(
    items: list[MemoryInsight],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in items]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not items:
        return "No active or stale insights."
    rows = [("INSIGHT_ID", "STATUS", "CONF", "USED", "POS", "NEG", "NEU", "EFF", "UPDATED")]
    for item in items:
        stats = dict(item.stats)
        rows.append((
            item.insight_id,
            item.status,
            f"{float(item.confidence):.2f}",
            str(int(stats.get("used_count", 0))),
            str(int(stats.get("positive_outcome_count", 0))),
            str(int(stats.get("negative_outcome_count", 0))),
            str(int(stats.get("neutral_outcome_count", 0))),
            f"{float(stats.get('effectiveness_score', 0.0)):.2f}",
            _short_ts(item.updated_at),
        ))
    return _render_table(rows)


def _format_doctor_report(
    report: Any,
    *,
    json_output: bool = False,
    errors_only: bool = False,
) -> str:
    if json_output:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)
    display_report = filter_doctor_report(report, errors_only=errors_only)
    if not display_report.checks:
        return "No matching doctor checks."
    rows = [("STATUS", "CATEGORY", "CHECK", "SUMMARY")]
    for item in display_report.checks:
        rows.append((
            item.status.upper(),
            item.category,
            item.title,
            item.summary,
        ))
    recommendations: list[str] = []
    seen: set[str] = set()
    for item in display_report.checks:
        for recommendation in item.recommendations:
            if recommendation in seen:
                continue
            seen.add(recommendation)
            recommendations.append(recommendation)
    lines = [_render_table(rows)]
    if recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
    return "\n".join(lines)


def _ordered_snapshot_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in (
        "snapshot_id",
        "created_at",
        "source_root",
        "doctor_status_at_creation",
        "latest_event_index",
        "file_count",
        "total_bytes",
        "included_paths",
        "errors",
    ):
        if key in payload:
            ordered[key] = payload[key]
    for key in sorted(set(payload) - set(ordered)):
        ordered[key] = payload[key]
    return ordered


def _format_snapshot_manifest(payload: dict[str, Any], *, json_output: bool = False) -> str:
    ordered = _ordered_snapshot_manifest(payload)
    if json_output:
        return json.dumps(ordered, indent=2, sort_keys=True)
    lines = [
        f"snapshot_id:               {ordered.get('snapshot_id', '-')}",
        f"created_at:                {ordered.get('created_at', '-')}",
        f"source_root:               {ordered.get('source_root', '-')}",
        f"doctor_status_at_creation: {ordered.get('doctor_status_at_creation', '-')}",
        f"latest_event_index:        {ordered.get('latest_event_index', '-')}",
        f"file_count:                {ordered.get('file_count', '-')}",
        f"total_bytes:               {ordered.get('total_bytes', '-')}",
        "included_paths:",
    ]
    included = ordered.get("included_paths") or []
    if included:
        for item in included:
            lines.append(f"  - {item}")
    else:
        lines.append("  -")
    lines.append("errors:")
    errors = ordered.get("errors") or []
    if errors:
        for item in errors:
            lines.append(f"  - {item}")
    else:
        lines.append("  -")
    return "\n".join(lines)


def _format_snapshot_list(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No snapshots."
    rows = [("SNAPSHOT_ID", "CREATED_AT", "DOCTOR_STATUS", "FILES", "BYTES")]
    for item in items:
        rows.append((
            str(item.get("snapshot_id", "-")),
            str(item.get("created_at", "-")),
            str(item.get("doctor_status_at_creation", "-")),
            str(item.get("file_count", "-")),
            str(item.get("total_bytes", "-")),
        ))
    return _render_table(rows)


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
    events = _timeline_for(store).list_events()
    return [
        event for event in events
        if _event_visible(
            event,
            store=store,
            scoped_agents=scoped_agents,
            message_store=message_store,
            caller_agent_id=caller_agent_id,
        )
    ]


def _require_visible_approval(
    approval_store: CapabilityApprovalStore,
    approval_request_id: str,
    scoped_agents: PersistentAgentStore,
    caller_agent_id: str,
) -> CapabilityApprovalRequest:
    approval = approval_store.require(approval_request_id)
    visible = {
        item.approval_request_id
        for item in _visible_approvals(approval_store, scoped_agents, caller_agent_id)
    }
    if approval.approval_request_id not in visible:
        raise AgentScopeError("access denied: approval not in agent scope")
    return approval


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
    if bool(args.use_memory) and bool(args.no_memory):
        print("error: invalid flag combination", file=sys.stderr)
        return 2
    owner_agent_id = args.owner or caller_agent_id

    def _submitter(
        spec: dict[str, Any],
        submit_caller_agent_id: str,
        submit_owner_agent_id: str,
        workflow_metadata: Optional[dict[str, Any]],
    ) -> tuple[str, str]:
        wf_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            store,
            owner_agent_id=submit_owner_agent_id,
            caller_agent_id=submit_caller_agent_id,
            workflow_metadata=workflow_metadata,
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
            use_memory=True if args.use_memory else False if args.no_memory else None,
            memory_limit=args.memory_limit,
        )
    except (WorkflowCompilerFailure, WorkflowSpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.show_memory:
        print(_format_compile_memory_summary(result), file=sys.stderr)
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


def _cmd_capability_call(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
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


def _cmd_approvals_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    approvals = _visible_approvals(_approval_store_for(store), scoped_agents, caller_agent_id)
    print(_format_approvals_table(approvals, json_output=args.json))
    return 0


def _cmd_approvals_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
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
    scoped_agents: PersistentAgentStore,
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
    print(approval.approval_request_id)
    return 0


def _cmd_capability_audit_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
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
    scoped_agents: PersistentAgentStore,
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


def _cmd_memory_update(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        result = update_graph_from_events(_timeline_for(store), _graph_store_for(store))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_update(result.to_dict(), json_output=args.json))
    return 0


def _cmd_memory_stats(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, cursor = _load_memory_graph(store)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_stats(graph_stats(graph, last_processed_event_index=cursor), json_output=args.json))
    return 0


def _cmd_memory_graph_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = show_node(graph, args.node_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0


def _cmd_memory_top_workflows(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_graph_top_workflows(store.root.parent, limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_templates(payload["items"], json_output=args.json))
    return 0


def _cmd_memory_capabilities(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_graph_capabilities(store.root.parent, limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_capabilities(payload["items"], json_output=args.json))
    return 0


def _cmd_memory_failures(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_graph_failures(store.root.parent, limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_failures(payload["items"], json_output=args.json))
    return 0


def _cmd_memory_agent(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    if not scoped_agents.is_visible(caller_agent_id, args.agent_id):
        print("error: access denied: agent not in scope", file=sys.stderr)
        return 2
    try:
        payload = memory_graph_agent_summary(store.root.parent, agent_id=args.agent_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not payload["found"]:
        print(f"error: agent not found: {args.agent_id}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload["summary"], json_output=args.json))
    return 0


def _cmd_memory_project(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = project_summary(graph, args.project_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0


def _cmd_memory_file(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = file_summary(graph, args.file_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0


def _cmd_memory_workflow_template(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = workflow_template_summary(graph, args.template_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0


def _cmd_memory_curation_due(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = evaluate_memory_curation_due(
            _timeline_for(store),
            _insight_store_for(store),
        ).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_curation_due(payload, json_output=args.json))
    return 0


def _cmd_memory_curate(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        run = run_memory_curation(
            event_log=_timeline_for(store),
            graph_store=_graph_store_for(store),
            insight_store=_insight_store_for(store),
            trigger_reason="memory_curate_cli",
            persist_not_due=True,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(run.to_dict(), json_output=args.json))
    return 0


def _load_insights_filtered(
    store: WorkflowStore,
    *,
    insight_types: Optional[set[str]] = None,
    include_statuses: Optional[set[str]] = None,
) -> list[MemoryInsight]:
    return list_filtered_insights(
        store.root.parent,
        insight_types=insight_types,
        include_statuses=include_statuses,
    )


def _cmd_memory_insights_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(store)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0


def _cmd_memory_insights_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_insight_show(store.root.parent, insight_id=args.insight_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not payload["found"]:
        print(f"error: insight not found: {args.insight_id}", file=sys.stderr)
        return 2
    insight = MemoryInsight.from_dict(dict(payload["insight"]))
    print(_format_memory_insight(insight, json_output=args.json))
    return 0


def _cmd_memory_insights_recommendations(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(
            store,
            insight_types={"workflow_recommendation", "scope_recommendation", "system_design_lesson"},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0


def _cmd_memory_insights_friction(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(
            store,
            insight_types={"capability_friction", "approval_friction"},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0


def _cmd_memory_insights_failures(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(
            store,
            insight_types={"failure_pattern"},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0


def _cmd_memory_insights_effectiveness(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(store, include_statuses={"active", "stale"})
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insight_effectiveness(items, json_output=args.json))
    return 0


def _cmd_memory_curation_runs(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        runs = _insight_store_for(store).load_runs(limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_curation_runs(runs, json_output=args.json))
    return 0


def _cmd_memory_feedback_due(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = evaluate_memory_feedback_due(
            _timeline_for(store),
            _insight_store_for(store),
            store,
        ).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0


def _cmd_memory_feedback_update(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = update_insight_feedback(
            _timeline_for(store),
            _insight_store_for(store),
            store,
        ).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0


def _cmd_memory_retrieval_update(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = update_memory_retrieval(store.root.parent).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0


def _cmd_memory_retrieval_stats(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    retrieval_store = RetrievalStore(store.root.parent)
    payload: dict[str, Any] = {
        "retrieval_ready": False,
        "document_count": 0,
        "schema_version": None,
        "updated_at": None,
        "doc_type_counts": {},
        "source_counts": {},
    }
    try:
        manifest = retrieval_store.load_manifest() if retrieval_store.manifest_path.exists() else {}
        documents = retrieval_store.load_documents() if retrieval_store.documents_path.exists() else []
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if manifest or documents:
        doc_type_counts: dict[str, int] = {}
        for item in documents:
            doc_type_counts[item.doc_type] = doc_type_counts.get(item.doc_type, 0) + 1
        payload = {
            "retrieval_ready": retrieval_store.exists(),
            "document_count": int(manifest.get("document_count", len(documents))),
            "schema_version": manifest.get("schema_version"),
            "updated_at": manifest.get("updated_at"),
            "doc_type_counts": dict(sorted(doc_type_counts.items())),
            "source_counts": dict(sorted(dict(manifest.get("source_counts", {})).items())),
        }
    print(_format_memory_retrieval_stats(payload, json_output=args.json))
    return 0


def _cmd_memory_retrieval_search(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_search(
            store.root.parent,
            query=args.query,
            limit=args.limit,
            types=list(args.types or []),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_retrieval_search(payload, json_output=args.json))
    return 0


def _cmd_memory_feedback_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _insight_store_for(store).load_feedback(limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_feedback(items, json_output=args.json))
    return 0


def _cmd_memory_feedback_insight(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _insight_store_for(store).load_feedback(limit=args.limit, insight_id=args.insight_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_feedback(items, json_output=args.json))
    return 0


def _cmd_doctor(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        report = run_doctor(_runtime_root_for(store), categories=list(args.categories or []))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_doctor_report(report, json_output=args.json, errors_only=bool(args.errors_only)))
    return 0


def _cmd_snapshot_create(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        manifest = create_snapshot(
            _runtime_root_for(store),
            fail_on_error=bool(args.fail_on_error),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_snapshot_manifest(manifest))
    return 0


def _cmd_snapshot_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del args, caller_agent_id, scoped_agents
    print(_format_snapshot_list(list_snapshots(_runtime_root_for(store))))
    return 0


def _cmd_snapshot_inspect(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = inspect_snapshot(_runtime_root_for(store), args.snapshot_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_snapshot_manifest(payload, json_output=bool(args.json)))
    return 0


def _cmd_memory_maintenance_spec(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    print(json.dumps(build_memory_maintenance_spec(), indent=2, sort_keys=True))
    return 0


def _cmd_memory_maintenance_run(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id
    try:
        workflow_id = submit_memory_maintenance_workflow(store, scoped_agent_store=scoped_agents)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"workflow_id": workflow_id}, indent=2, sort_keys=True))
    else:
        print(workflow_id)
    return 0


def _cmd_memory_maintenance_status(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    print(_format_memory_detail(maintenance_status_payload(store), json_output=args.json))
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
    p_compile.add_argument("--use-memory", action="store_true")
    p_compile.add_argument("--no-memory", action="store_true")
    p_compile.add_argument("--show-memory", action="store_true")
    p_compile.add_argument("--memory-limit", type=int, default=5)
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

    p_doctor = subs.add_parser("doctor", help="Run deterministic runtime health checks.")
    p_doctor.add_argument("--category", action="append", dest="categories", default=[])
    p_doctor.add_argument("--errors-only", action="store_true")
    add_common_flags(p_doctor, include_example=False)
    p_doctor.set_defaults(func=_cmd_doctor)

    p_snapshot = subs.add_parser("snapshot", help="Create and inspect runtime snapshots.")
    snapshot_subs = p_snapshot.add_subparsers(dest="snapshot_command", required=True)

    p_snapshot_create = snapshot_subs.add_parser("create", help="Create a read-only runtime snapshot.")
    p_snapshot_create.add_argument("--fail-on-error", action="store_true")
    p_snapshot_create.set_defaults(func=_cmd_snapshot_create)

    p_snapshot_list = snapshot_subs.add_parser("list", help="List complete runtime snapshots.")
    p_snapshot_list.set_defaults(func=_cmd_snapshot_list)

    p_snapshot_inspect = snapshot_subs.add_parser("inspect", help="Inspect one runtime snapshot manifest.")
    p_snapshot_inspect.add_argument("snapshot_id")
    p_snapshot_inspect.add_argument("--json", action="store_true", dest="json")
    p_snapshot_inspect.set_defaults(func=_cmd_snapshot_inspect)

    p_timeline = subs.add_parser("timeline", help="Inspect the unified runtime timeline.")
    timeline_subs = p_timeline.add_subparsers(dest="timeline_command", required=True)

    p_timeline_list = timeline_subs.add_parser("list", help="List visible timeline events.")
    p_timeline_list.add_argument("--limit", type=int, default=None)
    add_common_flags(p_timeline_list, include_example=False)
    p_timeline_list.set_defaults(func=_cmd_timeline_list)

    p_timeline_recent = timeline_subs.add_parser("recent", help="Show recent timeline events.")
    p_timeline_recent.add_argument("--limit", type=int, default=20)
    add_common_flags(p_timeline_recent, include_example=False)
    p_timeline_recent.set_defaults(func=_cmd_timeline_recent)

    p_timeline_show = timeline_subs.add_parser("show", help="Show one timeline event.")
    p_timeline_show.add_argument("event_id")
    add_common_flags(p_timeline_show, include_example=False)
    p_timeline_show.set_defaults(func=_cmd_timeline_show)

    p_timeline_trace = timeline_subs.add_parser("trace", help="Trace one correlation id.")
    p_timeline_trace.add_argument("correlation_id")
    add_common_flags(p_timeline_trace, include_example=False)
    p_timeline_trace.set_defaults(func=_cmd_timeline_trace)

    p_timeline_blocked = timeline_subs.add_parser("blocked", help="Show currently blocked timeline items.")
    add_common_flags(p_timeline_blocked, include_example=False)
    p_timeline_blocked.set_defaults(func=_cmd_timeline_blocked)

    p_timeline_approvals = timeline_subs.add_parser("approvals", help="Show approval lifecycle events.")
    add_common_flags(p_timeline_approvals, include_example=False)
    p_timeline_approvals.set_defaults(func=_cmd_timeline_approvals)

    p_timeline_agent = timeline_subs.add_parser("agent", help="Show timeline events related to one agent.")
    p_timeline_agent.add_argument("agent_id")
    add_common_flags(p_timeline_agent, include_example=False)
    p_timeline_agent.set_defaults(func=_cmd_timeline_agent)

    p_timeline_workflow = timeline_subs.add_parser("workflow", help="Show timeline events for one workflow.")
    p_timeline_workflow.add_argument("workflow_id")
    add_common_flags(p_timeline_workflow, include_example=False)
    p_timeline_workflow.set_defaults(func=_cmd_timeline_workflow)

    p_watchers = subs.add_parser("watchers", help="List active watcher tasks.")
    p_watchers.set_defaults(func=_cmd_watchers)

    p_capabilities = subs.add_parser("capabilities", help="List registered capabilities.")
    add_common_flags(p_capabilities, include_example=False)
    p_capabilities.set_defaults(func=_cmd_capabilities)

    p_capability = subs.add_parser("capability", help="Show one capability description.")
    p_capability.add_argument("name")
    add_common_flags(p_capability, include_example=True)
    p_capability.set_defaults(func=_cmd_capability)

    p_capability_call = subs.add_parser(
        "capability-call",
        help="Invoke a direct-callable capability once and print the result.",
    )
    p_capability_call.add_argument("name", help="Capability name.")
    p_capability_call.add_argument("config_file", help="Path to a JSON config file.")
    p_capability_call.set_defaults(func=_cmd_capability_call)

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

    p_agent_grant_scope = subs.add_parser("agent-grant-scope", help="Grant one normalized scope root to an agent.")
    p_agent_grant_scope.add_argument("agent_id")
    p_agent_grant_scope.add_argument("path")
    p_agent_grant_scope.add_argument("--reason", default="scope grant")
    p_agent_grant_scope.set_defaults(func=_cmd_agent_grant_scope)

    p_agent_revoke_scope = subs.add_parser("agent-revoke-scope", help="Revoke one normalized scope root from an agent.")
    p_agent_revoke_scope.add_argument("agent_id")
    p_agent_revoke_scope.add_argument("path")
    p_agent_revoke_scope.set_defaults(func=_cmd_agent_revoke_scope)

    p_agent_scopes = subs.add_parser("agent-scopes", help="Show effective scope roots and grant provenance for an agent.")
    p_agent_scopes.add_argument("agent_id")
    add_common_flags(p_agent_scopes, include_example=False)
    p_agent_scopes.set_defaults(func=_cmd_agent_scopes)

    p_approvals = subs.add_parser("approvals", help="List capability approval requests.")
    approvals_subs = p_approvals.add_subparsers(dest="approvals_command", required=True)

    p_approvals_list = approvals_subs.add_parser("list", help="List visible approval requests.")
    add_common_flags(p_approvals_list, include_example=False)
    p_approvals_list.set_defaults(func=_cmd_approvals_list)

    p_approvals_show = approvals_subs.add_parser("show", help="Show one approval request.")
    p_approvals_show.add_argument("approval_request_id")
    add_common_flags(p_approvals_show, include_example=False)
    p_approvals_show.set_defaults(func=_cmd_approvals_show)

    p_approvals_approve = approvals_subs.add_parser("approve", help="Approve one pending approval request.")
    p_approvals_approve.add_argument("approval_request_id")
    p_approvals_approve.add_argument("--grant-scope", action="store_true")
    p_approvals_approve.add_argument("--reason", default="approved")
    p_approvals_approve.set_defaults(func=_cmd_approvals_decide, decision="approved")

    p_approvals_deny = approvals_subs.add_parser("deny", help="Deny one pending approval request.")
    p_approvals_deny.add_argument("approval_request_id")
    p_approvals_deny.add_argument("--reason", default="denied")
    p_approvals_deny.set_defaults(func=_cmd_approvals_decide, decision="denied", grant_scope=False)

    p_capability_audit = subs.add_parser("capability-audit", help="Inspect indexed capability audit records.")
    capability_audit_subs = p_capability_audit.add_subparsers(dest="capability_audit_command", required=True)

    p_capability_audit_list = capability_audit_subs.add_parser("list", help="List visible capability audits.")
    p_capability_audit_list.add_argument("--agent", dest="agent_id", default=None)
    add_common_flags(p_capability_audit_list, include_example=False)
    p_capability_audit_list.set_defaults(func=_cmd_capability_audit_list)

    p_capability_audit_show = capability_audit_subs.add_parser("show", help="Show one capability audit record.")
    p_capability_audit_show.add_argument("audit_id")
    add_common_flags(p_capability_audit_show, include_example=False)
    p_capability_audit_show.set_defaults(func=_cmd_capability_audit_show)

    p_memory = subs.add_parser("memory", help="Inspect deterministic graph memory.")
    memory_subs = p_memory.add_subparsers(dest="memory_command", required=True)

    p_memory_update = memory_subs.add_parser("update", help="Update graph memory from timeline events.")
    add_common_flags(p_memory_update, include_example=False)
    p_memory_update.set_defaults(func=_cmd_memory_update)

    p_memory_stats = memory_subs.add_parser("stats", help="Show graph memory counts and cursor state.")
    add_common_flags(p_memory_stats, include_example=False)
    p_memory_stats.set_defaults(func=_cmd_memory_stats)

    p_memory_due = memory_subs.add_parser("curation-due", help="Check whether memory curation is due.")
    add_common_flags(p_memory_due, include_example=False)
    p_memory_due.set_defaults(func=_cmd_memory_curation_due)

    p_memory_curate = memory_subs.add_parser("curate", help="Run one bounded memory curation pass.")
    add_common_flags(p_memory_curate, include_example=False)
    p_memory_curate.set_defaults(func=_cmd_memory_curate)

    p_memory_retrieval = memory_subs.add_parser("retrieval", help="Inspect unified retrieval documents.")
    memory_retrieval_subs = p_memory_retrieval.add_subparsers(dest="memory_retrieval_command", required=True)

    p_memory_retrieval_update = memory_retrieval_subs.add_parser("update", help="Rebuild unified retrieval documents.")
    add_common_flags(p_memory_retrieval_update, include_example=False)
    p_memory_retrieval_update.set_defaults(func=_cmd_memory_retrieval_update)

    p_memory_retrieval_stats = memory_retrieval_subs.add_parser("stats", help="Show retrieval manifest and document counts.")
    add_common_flags(p_memory_retrieval_stats, include_example=False)
    p_memory_retrieval_stats.set_defaults(func=_cmd_memory_retrieval_stats)

    p_memory_retrieval_search = memory_retrieval_subs.add_parser("search", help="Search unified retrieval documents.")
    p_memory_retrieval_search.add_argument("query")
    p_memory_retrieval_search.add_argument("--limit", type=int, default=5)
    p_memory_retrieval_search.add_argument("--type", action="append", dest="types", default=[])
    add_common_flags(p_memory_retrieval_search, include_example=False)
    p_memory_retrieval_search.set_defaults(func=_cmd_memory_retrieval_search)

    p_memory_insights = memory_subs.add_parser("insights", help="Inspect curated memory insights.")
    memory_insights_subs = p_memory_insights.add_subparsers(dest="memory_insights_command", required=True)

    p_memory_insights_list = memory_insights_subs.add_parser("list", help="List active insights.")
    add_common_flags(p_memory_insights_list, include_example=False)
    p_memory_insights_list.set_defaults(func=_cmd_memory_insights_list)

    p_memory_insights_show = memory_insights_subs.add_parser("show", help="Show one insight.")
    p_memory_insights_show.add_argument("insight_id")
    add_common_flags(p_memory_insights_show, include_example=False)
    p_memory_insights_show.set_defaults(func=_cmd_memory_insights_show)

    p_memory_insights_recommendations = memory_insights_subs.add_parser(
        "recommendations",
        help="List recommendation-oriented insights.",
    )
    add_common_flags(p_memory_insights_recommendations, include_example=False)
    p_memory_insights_recommendations.set_defaults(func=_cmd_memory_insights_recommendations)

    p_memory_insights_friction = memory_insights_subs.add_parser(
        "friction",
        help="List capability and approval friction insights.",
    )
    add_common_flags(p_memory_insights_friction, include_example=False)
    p_memory_insights_friction.set_defaults(func=_cmd_memory_insights_friction)

    p_memory_insights_failures = memory_insights_subs.add_parser(
        "failures",
        help="List failure-pattern insights.",
    )
    add_common_flags(p_memory_insights_failures, include_example=False)
    p_memory_insights_failures.set_defaults(func=_cmd_memory_insights_failures)

    p_memory_insights_effectiveness = memory_insights_subs.add_parser(
        "effectiveness",
        help="Show active and stale insight effectiveness stats.",
    )
    add_common_flags(p_memory_insights_effectiveness, include_example=False)
    p_memory_insights_effectiveness.set_defaults(func=_cmd_memory_insights_effectiveness)

    p_memory_runs = memory_subs.add_parser("curation-runs", help="List recent memory curation runs.")
    p_memory_runs.add_argument("--limit", type=int, default=20)
    add_common_flags(p_memory_runs, include_example=False)
    p_memory_runs.set_defaults(func=_cmd_memory_curation_runs)

    p_memory_feedback = memory_subs.add_parser("feedback", help="Inspect insight feedback.")
    memory_feedback_subs = p_memory_feedback.add_subparsers(dest="memory_feedback_command", required=True)

    p_memory_feedback_due = memory_feedback_subs.add_parser("due", help="Check whether insight feedback is due.")
    add_common_flags(p_memory_feedback_due, include_example=False)
    p_memory_feedback_due.set_defaults(func=_cmd_memory_feedback_due)

    p_memory_feedback_update = memory_feedback_subs.add_parser("update", help="Run one insight feedback update pass.")
    add_common_flags(p_memory_feedback_update, include_example=False)
    p_memory_feedback_update.set_defaults(func=_cmd_memory_feedback_update)

    p_memory_feedback_list = memory_feedback_subs.add_parser("list", help="List recent insight feedback.")
    p_memory_feedback_list.add_argument("--limit", type=int, default=20)
    add_common_flags(p_memory_feedback_list, include_example=False)
    p_memory_feedback_list.set_defaults(func=_cmd_memory_feedback_list)

    p_memory_feedback_insight = memory_feedback_subs.add_parser("insight", help="List feedback for one insight.")
    p_memory_feedback_insight.add_argument("insight_id")
    p_memory_feedback_insight.add_argument("--limit", type=int, default=20)
    add_common_flags(p_memory_feedback_insight, include_example=False)
    p_memory_feedback_insight.set_defaults(func=_cmd_memory_feedback_insight)

    p_memory_maintenance = memory_subs.add_parser("maintenance", help="Build or submit the memory maintenance workflow.")
    memory_maintenance_subs = p_memory_maintenance.add_subparsers(dest="memory_maintenance_command", required=True)

    p_memory_maintenance_spec = memory_maintenance_subs.add_parser("spec", help="Print the memory maintenance workflow spec.")
    add_common_flags(p_memory_maintenance_spec, include_example=False)
    p_memory_maintenance_spec.set_defaults(func=_cmd_memory_maintenance_spec)

    p_memory_maintenance_run = memory_maintenance_subs.add_parser("run", help="Submit the memory maintenance workflow.")
    add_common_flags(p_memory_maintenance_run, include_example=False)
    p_memory_maintenance_run.set_defaults(func=_cmd_memory_maintenance_run)

    p_memory_maintenance_status = memory_maintenance_subs.add_parser("status", help="Show the latest memory maintenance workflow status.")
    add_common_flags(p_memory_maintenance_status, include_example=False)
    p_memory_maintenance_status.set_defaults(func=_cmd_memory_maintenance_status)

    p_memory_graph = memory_subs.add_parser("graph", help="Inspect graph memory views.")
    memory_graph_subs = p_memory_graph.add_subparsers(dest="memory_graph_command", required=True)

    p_memory_graph_show = memory_graph_subs.add_parser("show", help="Show one graph node and its incident edges.")
    p_memory_graph_show.add_argument("node_id")
    add_common_flags(p_memory_graph_show, include_example=False)
    p_memory_graph_show.set_defaults(func=_cmd_memory_graph_show)

    p_memory_top = memory_graph_subs.add_parser("top-workflows", help="Show top workflow templates.")
    p_memory_top.add_argument("--limit", type=int, default=10)
    add_common_flags(p_memory_top, include_example=False)
    p_memory_top.set_defaults(func=_cmd_memory_top_workflows)

    p_memory_caps = memory_graph_subs.add_parser("capabilities", help="Show capability graph stats.")
    p_memory_caps.add_argument("--limit", type=int, default=None)
    add_common_flags(p_memory_caps, include_example=False)
    p_memory_caps.set_defaults(func=_cmd_memory_capabilities)

    p_memory_failures = memory_graph_subs.add_parser("failures", help="Show common failure modes.")
    p_memory_failures.add_argument("--limit", type=int, default=10)
    add_common_flags(p_memory_failures, include_example=False)
    p_memory_failures.set_defaults(func=_cmd_memory_failures)

    p_memory_agent = memory_graph_subs.add_parser("agent", help="Show one agent graph summary.")
    p_memory_agent.add_argument("agent_id")
    add_common_flags(p_memory_agent, include_example=False)
    p_memory_agent.set_defaults(func=_cmd_memory_agent)

    p_memory_project = memory_graph_subs.add_parser("project", help="Show one project graph summary.")
    p_memory_project.add_argument("project_id")
    add_common_flags(p_memory_project, include_example=False)
    p_memory_project.set_defaults(func=_cmd_memory_project)

    p_memory_file = memory_graph_subs.add_parser("file", help="Show one file graph summary.")
    p_memory_file.add_argument("file_id")
    add_common_flags(p_memory_file, include_example=False)
    p_memory_file.set_defaults(func=_cmd_memory_file)

    p_memory_template = memory_graph_subs.add_parser("workflow-template", help="Show one workflow template summary.")
    p_memory_template.add_argument("template_id")
    add_common_flags(p_memory_template, include_example=False)
    p_memory_template.set_defaults(func=_cmd_memory_workflow_template)

    p_inbox = subs.add_parser("inbox", help="List inbox messages for an agent.")
    p_inbox.add_argument("--agent", default=None)
    p_inbox.add_argument("--archived", action="store_true")
    add_common_flags(p_inbox, include_example=False)
    p_inbox.set_defaults(func=_cmd_inbox)

    p_inbox_triage = subs.add_parser("inbox-triage", help="Run one bounded root inbox triage pass.")
    p_inbox_triage.add_argument("--max-messages", type=int, default=10)
    p_inbox_triage.add_argument("--max-actions", type=int, default=3)
    p_inbox_triage.set_defaults(func=_cmd_inbox_triage)

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
    command_name = " ".join(
        item for item in [
            getattr(args, "command", None),
            getattr(args, "snapshot_command", None),
            getattr(args, "timeline_command", None),
            getattr(args, "approvals_command", None),
            getattr(args, "capability_audit_command", None),
            getattr(args, "memory_command", None),
            getattr(args, "memory_feedback_command", None),
            getattr(args, "memory_insights_command", None),
            getattr(args, "memory_maintenance_command", None),
            getattr(args, "memory_graph_command", None),
            getattr(args, "memory_retrieval_command", None),
        ]
        if item
    )
    with bind_correlation_id(cli_correlation_id(resolved_caller_agent_id, command_name or "cli")):
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
