"""Workflow CLI: workflow/jobs/tasks/result/inputs/artifacts/watchers/schema/trigger commands and formatters.

Includes the `_cmd_submit`, `_cmd_compile_workflow`, `_cmd_workflow*`,
`_cmd_task`, `_cmd_jobs`, `_cmd_result`, `_cmd_inputs`, `_cmd_artifacts`,
`_cmd_watchers`, `_cmd_schema`, `_cmd_trigger`, `_cmd_rerun`,
`_cmd_cancel_*`, and the workflow-mutation `_cmd_append/insert/replace_workflow`
handlers plus their formatters.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.dataflow import Artifact, ResolvedTaskInput, TaskOutput
from mr1.tools import default_tool_registry
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
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore
from mr1.workflow_compiler import WorkflowCompilerClient, WorkflowCompilerFailure
from mr1.workflow_models import (
    Provenance,
    Task,
    TaskStatus,
    Workflow,
    WorkflowStatus,
)
from mr1.workflow_schema import (
    WorkflowSchemaRegistry,
    default_workflow_schema_registry,
)
from mr1.workflow_store import WorkflowStore

from mr1.cli.context import (
    _approval_store_for,
    _find_workflow_for_task,
    _load_json_file,
    _load_scoped_workflow,
    _load_text_file,
    _runtime_access_for,
    _runtime_root_for,
    _timeline_for,
    _visible_workflows,
)
from mr1.cli.formatting import (
    _compact_text,
    _format_schema,
    _label_for_task_id,
    _reject_invalid_flag_combination,
    _render_table,
    _short_ts,
)
from mr1.cli.memory import _format_compile_memory_summary


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

def _cmd_watchers(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    print(_format_watchers(_visible_workflows(store, scoped_agents, caller_agent_id)))
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
