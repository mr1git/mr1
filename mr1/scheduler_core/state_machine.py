from __future__ import annotations

from mr1.workflow_models import FAILED_TASK_STATUSES, TERMINAL_TASK_STATUSES, Task, TaskStatus, Workflow, WorkflowStatus


def is_terminal_status(status: TaskStatus) -> bool:
    return status in TERMINAL_TASK_STATUSES


def is_success_status(status: TaskStatus) -> bool:
    return status in {TaskStatus.SUCCEEDED, TaskStatus.SKIPPED}


def is_failure_status(status: TaskStatus) -> bool:
    return status in FAILED_TASK_STATUSES


def reopen_workflow(workflow: Workflow) -> None:
    workflow.finished_at = None
    if any(task.status is TaskStatus.RUNNING for task in workflow.tasks.values()):
        workflow.status = WorkflowStatus.RUNNING
    else:
        workflow.status = WorkflowStatus.PENDING


def reset_task_runtime_state(task: Task, *, status: TaskStatus) -> None:
    task.status = status
    task.started_at = None
    task.finished_at = None
    task.pid = None
    task.exit_code = None
    task.last_error = None
    task.last_error_type = None
    task.result_summary = None
    task.log_stdout_path = None
    task.log_stderr_path = None
    task.result_path = None
    task.dataflow_error = None
    task.blocked_by = []
    task.blocked_reason = None
    task.blocked_at = None
    task.skip_reason = None
    task.condition_result = None
    task.watch_started_at = None
    task.watch_satisfied_at = None
    task.last_checked_at = None
    task.last_check_result = None
    task.tool_started_at = None
    task.tool_finished_at = None
    task.tool_error = None
