from __future__ import annotations

from dataclasses import dataclass, field

from mr1.conditions import SUPPORTED_DEPENDENCY_POLICIES
from mr1.workflow_models import FAILED_TASK_STATUSES, Task, TaskStatus, Workflow


@dataclass(frozen=True)
class DependencyGateDecision:
    state: str
    reason: str
    blocked_by: list[str] = field(default_factory=list)


def evaluate_dependency_gate(workflow: Workflow, task: Task) -> DependencyGateDecision:
    if not task.depends_on:
        return DependencyGateDecision("pass", "no dependencies; ready to run")

    policy = task.dependency_policy
    if policy not in SUPPORTED_DEPENDENCY_POLICIES:
        policy = "all_succeeded"

    if policy == "all_succeeded":
        failed_parents = [
            parent_id
            for parent_id in task.depends_on
            if (parent := workflow.tasks.get(parent_id)) is not None
            and parent.status in FAILED_TASK_STATUSES
        ]
        if failed_parents:
            parent_statuses = [
                f"{workflow.tasks[parent_id].label}={workflow.tasks[parent_id].status.value}"
                for parent_id in failed_parents
            ]
            return DependencyGateDecision(
                "block",
                ", ".join(parent_statuses),
                blocked_by=failed_parents,
            )
        skipped_parents = [
            parent_id
            for parent_id in task.depends_on
            if (parent := workflow.tasks.get(parent_id)) is not None
            and parent.status is TaskStatus.SKIPPED
        ]
        if skipped_parents:
            return DependencyGateDecision(
                "skip",
                "dependency branch skipped under all_succeeded",
            )
        parents_ok = all(
            (parent := workflow.tasks.get(parent_id)) is not None
            and parent.status is TaskStatus.SUCCEEDED
            for parent_id in task.depends_on
        )
        if parents_ok:
            return DependencyGateDecision("pass", "all dependencies succeeded")
        return DependencyGateDecision(
            "wait",
            f"waiting on {len(task.depends_on)} dependency(ies)",
        )

    all_terminal = all(
        (parent := workflow.tasks.get(parent_id)) is not None
        and parent.is_terminal()
        for parent_id in task.depends_on
    )
    if not all_terminal:
        return DependencyGateDecision(
            "wait",
            f"waiting on {len(task.depends_on)} dependency(ies)",
        )
    if any(
        (parent := workflow.tasks.get(parent_id)) is not None
        and parent.status is TaskStatus.SUCCEEDED
        for parent_id in task.depends_on
    ):
        return DependencyGateDecision("pass", "dependency policy any_succeeded satisfied")
    return DependencyGateDecision("skip", "no dependency succeeded under any_succeeded")


def status_for_reset(workflow: Workflow, task: Task) -> TaskStatus:
    gate = evaluate_dependency_gate(workflow, task)
    return TaskStatus.READY if gate.state == "pass" else TaskStatus.WAITING


def compute_ancestor_labels(depends_on_by_label: dict[str, list[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def visit(label: str) -> set[str]:
        if label in memo:
            return memo[label]
        ancestors: set[str] = set()
        for dep in depends_on_by_label.get(label, []):
            ancestors.add(dep)
            ancestors.update(visit(dep))
        memo[label] = ancestors
        return ancestors

    return {label: visit(label) for label in depends_on_by_label}
