from __future__ import annotations

from pathlib import Path
from typing import Optional

from mr1.messages import MessageStore
from mr1.scoped_agents import AgentStore
from mr1.scheduler_core.diagnostics import summarize_task_status
from mr1.workflow_models import Workflow


def build_workflow_report_body(
    workflow: Workflow,
    *,
    owner_title: str,
    report_path: Path,
) -> str:
    body_lines = [
        f"Workflow completed: {workflow.title}",
        "",
        f"workflow_id: {workflow.workflow_id}",
        f"owner_agent_id: {workflow.owner_agent_id}",
        f"owner_agent_title: {owner_title}",
        f"status: {workflow.status.value}",
        f"finished_at: {workflow.finished_at or '-'}",
        "",
        "Task summary:",
    ]
    emitted = False
    for label, task_id in workflow.label_to_task_id.items():
        task = workflow.tasks.get(task_id)
        if task is None:
            continue
        emitted = True
        body_lines.append(f"- {summarize_task_status(task)}")
    if not emitted:
        body_lines.append("- none")
    body_lines.extend([
        "",
        f"Report path: {report_path}",
    ])
    return "\n".join(body_lines).rstrip()


class WorkflowReporter:
    def __init__(
        self,
        *,
        scoped_agents: AgentStore,
        message_store: MessageStore,
    ) -> None:
        self._scoped_agents = scoped_agents
        self._message_store = message_store

    def send_workflow_report_message(self, workflow: Workflow, report_path: Path) -> None:
        workflow = self._scoped_agents.normalize_workflow_ownership(workflow)
        owner = self._scoped_agents.load_agent(workflow.owner_agent_id)
        if owner is None or owner.parent_agent_id is None:
            return
        body = build_workflow_report_body(
            workflow,
            owner_title=workflow.owner_agent_title or owner.title,
            report_path=report_path,
        )
        self._message_store.create_message(
            from_agent_id=owner.agent_id,
            to_agent_id=owner.parent_agent_id,
            kind="report",
            subject=f"Workflow completed: {workflow.title}",
            body=body,
            workflow_id=workflow.workflow_id,
        )
