from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from mr1.dataflow import DataflowError, TaskInputSpec, parse_input_reference, resolve_task_input


SUPPORTED_DEPENDENCY_POLICIES = frozenset({
    "all_succeeded",
    "any_succeeded",
})

SUPPORTED_CONDITION_OPS = frozenset({
    "eq",
    "ne",
    "contains",
    "exists",
    "missing",
    "gt",
    "gte",
    "lt",
    "lte",
    "truthy",
    "falsy",
})

VALUE_REQUIRED_OPS = frozenset({
    "eq",
    "ne",
    "contains",
    "gt",
    "gte",
    "lt",
    "lte",
})


@dataclass
class ConditionEvaluation:
    passed: bool
    actual: Any
    expected: Any
    op: str
    ref: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_dependency_policy(policy: Any, *, task_label: str) -> None:
    if not isinstance(policy, str) or policy not in SUPPORTED_DEPENDENCY_POLICIES:
        raise ValueError(
            f"task '{task_label}': dependency_policy '{policy}' not supported"
        )


def validate_condition(condition: Any, workflow: Any, task: Any) -> None:
    if condition is None:
        return
    label = _task_label(task)
    if not isinstance(condition, dict):
        raise ValueError(f"task '{label}': run_if must be a JSON object")

    ref = condition.get("ref")
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"task '{label}': run_if.ref must be a non-empty string")
    op = condition.get("op")
    if not isinstance(op, str) or not op:
        raise ValueError(f"task '{label}': run_if.op must be a non-empty string")
    if op not in SUPPORTED_CONDITION_OPS:
        raise ValueError(f"task '{label}': run_if.op '{op}' not supported")
    if op in VALUE_REQUIRED_OPS and "value" not in condition:
        raise ValueError(f"task '{label}': run_if.value is required for op '{op}'")

    try:
        parsed = parse_input_reference(ref)
    except DataflowError as exc:
        raise ValueError(f"task '{label}': {exc}") from exc

    labels = _workflow_labels(workflow)
    if parsed.label not in labels:
        raise ValueError(
            f"task '{label}': run_if source label '{parsed.label}' is unknown"
        )
    ancestors = _task_ancestor_labels(workflow, task)
    if parsed.label not in ancestors:
        raise ValueError(
            f"task '{label}': run_if source '{parsed.label}' must be an upstream dependency"
        )


def evaluate_condition(condition: dict[str, Any], workflow: Any, task: Any, store: Any) -> ConditionEvaluation:
    ref = condition["ref"]
    op = condition["op"]
    expected = condition.get("value")
    resolved = resolve_task_input(
        workflow,
        TaskInputSpec(name="_condition", from_ref=ref),
        store,
    )
    actual = None if resolved.resolved_type == "missing" else resolved.value
    metadata = dict(resolved.metadata)
    metadata["resolved_type"] = resolved.resolved_type
    metadata["resolved_task_id"] = resolved.resolved_task_id

    if op == "exists":
        passed = resolved.resolved_type != "missing"
        reason = "reference resolved" if passed else "reference missing"
        return ConditionEvaluation(passed, actual, None, op, ref, reason, metadata)
    if op == "missing":
        passed = resolved.resolved_type == "missing"
        reason = "reference missing" if passed else "reference resolved"
        return ConditionEvaluation(passed, actual, None, op, ref, reason, metadata)
    if resolved.resolved_type == "missing":
        return ConditionEvaluation(
            False,
            None,
            expected,
            op,
            ref,
            "reference missing",
            metadata,
        )

    if op == "eq":
        passed = actual == expected
        reason = "values equal" if passed else "values not equal"
        return ConditionEvaluation(passed, actual, expected, op, ref, reason, metadata)
    if op == "ne":
        passed = actual != expected
        reason = "values differ" if passed else "values equal"
        return ConditionEvaluation(passed, actual, expected, op, ref, reason, metadata)
    if op == "contains":
        return _evaluate_contains(actual, expected, op, ref, metadata)
    if op in {"gt", "gte", "lt", "lte"}:
        return _evaluate_numeric(actual, expected, op, ref, metadata)
    if op == "truthy":
        passed = bool(actual)
        reason = "value is truthy" if passed else "value is falsy"
        return ConditionEvaluation(passed, actual, None, op, ref, reason, metadata)
    if op == "falsy":
        passed = not bool(actual)
        reason = "value is falsy" if passed else "value is truthy"
        return ConditionEvaluation(passed, actual, None, op, ref, reason, metadata)
    return ConditionEvaluation(
        False,
        actual,
        expected,
        op,
        ref,
        "unsupported operator",
        metadata,
    )


def _evaluate_contains(
    actual: Any,
    expected: Any,
    op: str,
    ref: str,
    metadata: dict[str, Any],
) -> ConditionEvaluation:
    if isinstance(actual, str):
        if not isinstance(expected, str):
            return ConditionEvaluation(
                False,
                actual,
                expected,
                op,
                ref,
                "contains on strings requires a string value",
                metadata,
            )
        passed = expected in actual
        reason = "substring found" if passed else "substring not found"
        return ConditionEvaluation(passed, actual, expected, op, ref, reason, metadata)
    if isinstance(actual, list):
        passed = expected in actual
        reason = "value present in list" if passed else "value not present in list"
        return ConditionEvaluation(passed, actual, expected, op, ref, reason, metadata)
    if isinstance(actual, dict):
        try:
            passed = expected in actual
        except TypeError:
            return ConditionEvaluation(
                False,
                actual,
                expected,
                op,
                ref,
                "contains on dicts requires a hashable key",
                metadata,
            )
        reason = "key present in dict" if passed else "key not present in dict"
        return ConditionEvaluation(passed, actual, expected, op, ref, reason, metadata)
    return ConditionEvaluation(
        False,
        actual,
        expected,
        op,
        ref,
        "contains is supported only for strings, lists, and dicts",
        metadata,
    )


def _evaluate_numeric(
    actual: Any,
    expected: Any,
    op: str,
    ref: str,
    metadata: dict[str, Any],
) -> ConditionEvaluation:
    if not _is_number(actual) or not _is_number(expected):
        return ConditionEvaluation(
            False,
            actual,
            expected,
            op,
            ref,
            "numeric comparison requires numeric values",
            metadata,
        )
    if op == "gt":
        passed = actual > expected
    elif op == "gte":
        passed = actual >= expected
    elif op == "lt":
        passed = actual < expected
    else:
        passed = actual <= expected
    reason = "numeric comparison passed" if passed else "numeric comparison failed"
    return ConditionEvaluation(passed, actual, expected, op, ref, reason, metadata)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _task_label(task: Any) -> str:
    if isinstance(task, dict):
        return str(task.get("label") or "<unknown>")
    return str(getattr(task, "label", "<unknown>"))


def _workflow_labels(workflow: Any) -> set[str]:
    if isinstance(workflow, dict):
        labels = set()
        for raw in workflow.get("tasks") or []:
            if isinstance(raw, dict) and isinstance(raw.get("label"), str):
                labels.add(raw["label"])
        return labels
    return set(getattr(workflow, "label_to_task_id", {}).keys())


def _task_ancestor_labels(workflow: Any, task: Any) -> set[str]:
    if isinstance(workflow, dict):
        depends_on_by_label = {
            raw["label"]: list(raw.get("depends_on") or [])
            for raw in (workflow.get("tasks") or [])
            if isinstance(raw, dict) and isinstance(raw.get("label"), str)
        }
        return _ancestor_labels_from_label(
            _task_label(task),
            depends_on_by_label,
        )

    label_to_task_id = dict(getattr(workflow, "label_to_task_id", {}))
    task_map = getattr(workflow, "tasks", {})
    task_id_to_label = {task_id: label for label, task_id in label_to_task_id.items()}
    start_ids = list(getattr(task, "depends_on", []) or [])
    ancestors: set[str] = set()
    stack = list(start_ids)
    while stack:
        parent_id = stack.pop()
        label = task_id_to_label.get(parent_id)
        if label:
            ancestors.add(label)
        parent = task_map.get(parent_id)
        if parent is None:
            continue
        for grandparent_id in getattr(parent, "depends_on", []) or []:
            if grandparent_id not in stack:
                stack.append(grandparent_id)
    return ancestors


def _ancestor_labels_from_label(
    label: str,
    depends_on_by_label: dict[str, list[str]],
    memo: Optional[dict[str, set[str]]] = None,
) -> set[str]:
    cache = memo if memo is not None else {}
    if label in cache:
        return cache[label]
    ancestors: set[str] = set()
    for dep in depends_on_by_label.get(label, []):
        ancestors.add(dep)
        ancestors.update(_ancestor_labels_from_label(dep, depends_on_by_label, cache))
    cache[label] = ancestors
    return ancestors
