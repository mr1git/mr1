"""
Deterministic workflow inspection and submission CLI.

`python -m mr1.workflow_cli <cmd>`

The CLI NEVER instantiates a scheduler. `submit` writes a workflow
directory to disk and exits; the MR1-owned scheduler picks it up on its
next tick. Read commands load state directly from the store and
pretty-print — zero LLM calls, zero subprocesses.

Sub-commands:
    submit <path>                      write workflow spec to the store
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

from mr1.dataflow import Artifact, ResolvedTaskInput, TaskOutput
from mr1.scheduler import (
    WatcherTriggerError,
    WorkflowSpecError,
    submit_spec_to_disk,
    trigger_watcher_on_disk,
)
from mr1.tools import ToolRegistry, default_tool_registry
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
    rows = [("LABEL", "TASK_ID", "STATUS", "DEPENDS_ON")]
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
        f"depends_on: {', '.join(task.depends_on) or '-'}",
        f"created:    {_short_ts(task.created_at)}",
        f"started:    {_short_ts(task.started_at)}",
        f"finished:   {_short_ts(task.finished_at)}",
        f"pid:        {task.pid if task.pid is not None else '-'}",
        f"exit_code:  {task.exit_code if task.exit_code is not None else '-'}",
    ]
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
    rows = [("TIMESTAMP", "EVENT", "TASK_ID", "MESSAGE")]
    for ev in events:
        rows.append((
            _short_ts(ev.timestamp),
            ev.event_type,
            ev.task_id or "-",
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


def _format_tools(registry: Optional[ToolRegistry] = None) -> str:
    active_registry = registry or default_tool_registry()
    tools = active_registry.list_tools()
    if not tools:
        return "No tools registered."
    rows = [("TOOL", "DESCRIPTION", "CONFIG_SHAPE")]
    for tool in tools:
        rows.append((tool.tool_type, tool.description, tool.config_shape))
    return _render_table(rows)


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
) -> tuple[Optional[Workflow], Optional[Task]]:
    for wf in store.list_workflows():
        task = wf.tasks.get(task_id)
        if task is not None:
            return wf, task
    return None, None


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def _cmd_submit(args: argparse.Namespace, store: WorkflowStore) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: spec file not found: {path}", file=sys.stderr)
        return 2
    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 2
    try:
        wf_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            store,
            tool_registry=default_tool_registry(),
        )
    except WorkflowSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(wf_id)
    return 0


def _cmd_workflows(args: argparse.Namespace, store: WorkflowStore) -> int:
    print(_format_workflows_table(store.list_workflows()))
    return 0


def _cmd_workflow(args: argparse.Namespace, store: WorkflowStore) -> int:
    wf = store.load_workflow(args.workflow_id)
    if wf is None:
        print(f"error: workflow not found: {args.workflow_id}", file=sys.stderr)
        return 2
    print(_format_workflow_detail(wf))
    return 0


def _cmd_task(args: argparse.Namespace, store: WorkflowStore) -> int:
    wf, task = _find_workflow_for_task(store, args.task_id)
    if wf is None or task is None:
        print(f"error: task not found: {args.task_id}", file=sys.stderr)
        return 2
    print(_format_task_detail(wf, task))
    return 0


def _cmd_jobs(args: argparse.Namespace, store: WorkflowStore) -> int:
    print(_format_jobs(store.list_workflows()))
    return 0


def _cmd_events(args: argparse.Namespace, store: WorkflowStore) -> int:
    if store.load_workflow(args.workflow_id) is None:
        print(f"error: workflow not found: {args.workflow_id}", file=sys.stderr)
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


def _cmd_watchers(args: argparse.Namespace, store: WorkflowStore) -> int:
    print(_format_watchers(store.list_workflows()))
    return 0


def _cmd_tools(args: argparse.Namespace, store: WorkflowStore) -> int:
    del args, store
    print(_format_tools())
    return 0


def _cmd_result(args: argparse.Namespace, store: WorkflowStore) -> int:
    wf, task = _find_workflow_for_task(store, args.task_id)
    if wf is None or task is None:
        print(f"error: task not found: {args.task_id}", file=sys.stderr)
        return 2
    print(_format_result(task, store.load_task_output(wf.workflow_id, task.task_id)))
    return 0


def _cmd_inputs(args: argparse.Namespace, store: WorkflowStore) -> int:
    wf, task = _find_workflow_for_task(store, args.task_id)
    if wf is None or task is None:
        print(f"error: task not found: {args.task_id}", file=sys.stderr)
        return 2
    print(_format_inputs(task, store.load_task_inputs(wf.workflow_id, task.task_id)))
    return 0


def _cmd_artifacts(args: argparse.Namespace, store: WorkflowStore) -> int:
    wf = store.load_workflow(args.workflow_id)
    if wf is None:
        print(f"error: workflow not found: {args.workflow_id}", file=sys.stderr)
        return 2
    print(_format_artifacts(wf))
    return 0


def _cmd_trigger(args: argparse.Namespace, store: WorkflowStore) -> int:
    try:
        task_id = trigger_watcher_on_disk(
            store,
            args.workflow_id,
            args.label_or_task_id,
            event_name=args.event_name,
            agent_id="cli",
        )
    except WatcherTriggerError as exc:
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

    p_submit = subs.add_parser("submit", help="Write a workflow spec to the store.")
    p_submit.add_argument("path", help="Path to a workflow JSON spec.")
    p_submit.set_defaults(func=_cmd_submit)

    p_list = subs.add_parser("workflows", help="List all workflows.")
    p_list.set_defaults(func=_cmd_workflows)

    p_wf = subs.add_parser("workflow", help="Show one workflow's detail.")
    p_wf.add_argument("workflow_id")
    p_wf.set_defaults(func=_cmd_workflow)

    p_task = subs.add_parser("task", help="Show one task's detail.")
    p_task.add_argument("task_id")
    p_task.set_defaults(func=_cmd_task)

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

    p_tools = subs.add_parser("tools", help="List registered deterministic workflow tools.")
    p_tools.set_defaults(func=_cmd_tools)

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

    return parser


def main(argv: Optional[list[str]] = None, *, store: Optional[WorkflowStore] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    active_store = store if store is not None else WorkflowStore(root=args.store_root)
    return args.func(args, active_store)


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
