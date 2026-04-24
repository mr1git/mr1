"""Public MR1 package exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "MR1": ("mr1.mr1", "MR1"),
    "MR1Process": ("mr1.mr1", "MR1Process"),
    "StateManager": ("mr1.mr1", "StateManager"),
    "KaziAsyncRunner": ("mr1.kazi_runner", "KaziAsyncRunner"),
    "KaziBlockingRunner": ("mr1.kazi_runner", "KaziBlockingRunner"),
    "MockRunner": ("mr1.kazi_runner", "MockRunner"),
    "RunHandle": ("mr1.kazi_runner", "RunHandle"),
    "RunResult": ("mr1.kazi_runner", "RunResult"),
    "RunStatus": ("mr1.kazi_runner", "RunStatus"),
    "Runner": ("mr1.kazi_runner", "Runner"),
    "Scheduler": ("mr1.scheduler", "Scheduler"),
    "WorkflowSpecError": ("mr1.scheduler", "WorkflowSpecError"),
    "build_workflow_from_spec": ("mr1.scheduler", "build_workflow_from_spec"),
    "submit_spec_to_disk": ("mr1.scheduler", "submit_spec_to_disk"),
    "validate_spec": ("mr1.scheduler", "validate_spec"),
    "WorkflowEventLog": ("mr1.workflow_events", "WorkflowEventLog"),
    "FAILED_TASK_STATUSES": ("mr1.workflow_models", "FAILED_TASK_STATUSES"),
    "TERMINAL_TASK_STATUSES": ("mr1.workflow_models", "TERMINAL_TASK_STATUSES"),
    "TERMINAL_WORKFLOW_STATUSES": ("mr1.workflow_models", "TERMINAL_WORKFLOW_STATUSES"),
    "Provenance": ("mr1.workflow_models", "Provenance"),
    "Task": ("mr1.workflow_models", "Task"),
    "TaskStatus": ("mr1.workflow_models", "TaskStatus"),
    "Workflow": ("mr1.workflow_models", "Workflow"),
    "WorkflowEvent": ("mr1.workflow_models", "WorkflowEvent"),
    "WorkflowStatus": ("mr1.workflow_models", "WorkflowStatus"),
    "new_task_id": ("mr1.workflow_models", "new_task_id"),
    "new_workflow_id": ("mr1.workflow_models", "new_workflow_id"),
    "WorkflowStore": ("mr1.workflow_store", "WorkflowStore"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
