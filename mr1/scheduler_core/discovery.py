from __future__ import annotations

from typing import Optional

from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_models import Task, Workflow
from mr1.workflow_store import WorkflowStore


def workflow_has_live_handles(workflow: Workflow, handle_task_ids: set[str]) -> bool:
    return any(task.task_id in handle_task_ids for task in workflow.tasks.values())


class WorkflowQueryService:
    def __init__(
        self,
        *,
        store: WorkflowStore,
        scoped_agents: PersistentAgentStore,
    ) -> None:
        self._store = store
        self._scoped_agents = scoped_agents

    def list_workflows(self) -> list[Workflow]:
        return [
            self._scoped_agents.normalize_workflow_ownership(workflow)
            for workflow in self._store.list_workflows()
        ]

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        workflow = self._store.load_workflow(workflow_id)
        if workflow is None:
            return None
        return self._scoped_agents.normalize_workflow_ownership(workflow)

    def get_task(self, task_id: str) -> Optional[Task]:
        for workflow in self._store.list_workflows():
            if task_id in workflow.tasks:
                return workflow.tasks[task_id]
        return None

    def list_watchers(self) -> list[tuple[Workflow, Task]]:
        watchers: list[tuple[Workflow, Task]] = []
        for workflow in self._store.list_workflows():
            for task in workflow.tasks.values():
                if task.task_kind != "watcher" or task.is_terminal():
                    continue
                watchers.append((workflow, task))
        return watchers

    def active_workflows(self, handle_task_ids: set[str]) -> list[Workflow]:
        active: list[Workflow] = []
        for workflow in self._store.list_active_workflows():
            if workflow.is_terminal() and not workflow_has_live_handles(workflow, handle_task_ids):
                continue
            active.append(workflow)
        return active
