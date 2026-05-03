from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from mr1.kazi_runner import RunHandle
from mr1.workflow_models import Task, TaskStatus, Workflow
from mr1.workflow_store import WorkflowStore


def probe_path(path: Optional[str | Path]) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size": None}
    resolved = Path(path)
    exists = resolved.exists()
    size: Optional[int] = None
    if exists:
        try:
            size = resolved.stat().st_size
        except OSError:
            size = None
    return {
        "path": str(resolved),
        "exists": exists,
        "size": size,
    }


def build_scheduler_log_metadata(
    task: Task,
    *,
    store: WorkflowStore,
    handle_present: bool,
    handle: Optional[RunHandle] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    attempt_result_path: Optional[str] = None
    if task.current_attempt > 0:
        attempt_result_path = str(store.task_attempt_result_path(
            task.workflow_id,
            task.task_id,
            task.current_attempt,
        ))
    metadata = {
        "workflow_id": task.workflow_id,
        "attempt_id": task.current_attempt or None,
        "pid": task.pid if task.pid is not None else (handle.pid if handle is not None else None),
        "handle_present": handle_present,
    }
    for prefix, path in (
        ("stdout", task.log_stdout_path),
        ("stderr", task.log_stderr_path),
        ("result", task.result_path),
        ("attempt_result", attempt_result_path),
    ):
        probe = probe_path(path)
        metadata[f"{prefix}_path"] = probe["path"]
        metadata[f"{prefix}_exists"] = probe["exists"]
        metadata[f"{prefix}_size"] = probe["size"]
    if extra:
        metadata.update(extra)
    return metadata


def available_agent_slots(workflow: Workflow, concurrency: int) -> int:
    running_now = sum(
        1
        for task in workflow.tasks.values()
        if task.status is TaskStatus.RUNNING and task.task_kind == "agent"
    )
    return concurrency - running_now


def summarize_task_status(task: Task) -> str:
    summary = f"{task.label}: {task.status.value}"
    if task.result_summary:
        summary += f" | {task.result_summary}"
    return summary
