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

from mr1.scheduler import WorkflowSpecError, submit_spec_to_disk
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
    if task.status is TaskStatus.BLOCKED:
        lines.append(f"blocked_by:     {', '.join(task.blocked_by) or '-'}")
        lines.append(f"blocked_reason: {task.blocked_reason or '-'}")
        lines.append(f"blocked_at:     {_short_ts(task.blocked_at)}")
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

    return parser


def main(argv: Optional[list[str]] = None, *, store: Optional[WorkflowStore] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    active_store = store if store is not None else WorkflowStore(root=args.store_root)
    return args.func(args, active_store)


if __name__ == "__main__":
    sys.exit(main())
