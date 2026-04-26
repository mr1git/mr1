from __future__ import annotations

import pytest

from mr1.conditions import evaluate_condition, validate_condition
from mr1.dataflow import TaskOutput
from mr1.scheduler import WorkflowSpecError, build_workflow_from_spec, validate_spec
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="agent", id="MR1")


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


def _workflow_with_output(store: WorkflowStore):
    spec = {
        "title": "Conditions",
        "tasks": [
            {
                "label": "check",
                "title": "Check",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "check",
            },
            {
                "label": "branch",
                "title": "Branch",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check"],
                "prompt": "branch",
            },
        ],
    }
    wf = build_workflow_from_spec(spec, PROV)
    check = wf.task_by_label("check")
    store.write_task_output(
        wf.workflow_id,
        check.task_id,
        TaskOutput(
            task_id=check.task_id,
            workflow_id=wf.workflow_id,
            status="succeeded",
            summary="done",
            text="hello world",
            data={
                "exit_code": 0,
                "message": "hello world",
                "items": ["a", "b"],
                "map": {"alpha": 1},
                "truthy_flag": "yes",
                "falsy_flag": "",
                "non_numeric": "abc",
            },
            metrics={"score": 7},
        ),
    )
    return wf, wf.task_by_label("branch")


class TestConditionEvaluation:
    def test_eq_and_ne(self, store):
        wf, task = _workflow_with_output(store)

        eq_result = evaluate_condition(
            {"ref": "check.result.data.exit_code", "op": "eq", "value": 0},
            wf,
            task,
            store,
        )
        ne_result = evaluate_condition(
            {"ref": "check.result.data.exit_code", "op": "ne", "value": 1},
            wf,
            task,
            store,
        )

        assert eq_result.passed is True
        assert ne_result.passed is True

    def test_contains_for_string_list_and_dict(self, store):
        wf, task = _workflow_with_output(store)

        string_result = evaluate_condition(
            {"ref": "check.result.data.message", "op": "contains", "value": "world"},
            wf,
            task,
            store,
        )
        list_result = evaluate_condition(
            {"ref": "check.result.data.items", "op": "contains", "value": "b"},
            wf,
            task,
            store,
        )
        dict_result = evaluate_condition(
            {"ref": "check.result.data.map", "op": "contains", "value": "alpha"},
            wf,
            task,
            store,
        )

        assert string_result.passed is True
        assert list_result.passed is True
        assert dict_result.passed is True

    def test_exists_and_missing(self, store):
        wf, task = _workflow_with_output(store)

        exists_result = evaluate_condition(
            {"ref": "check.result.data.exit_code", "op": "exists"},
            wf,
            task,
            store,
        )
        missing_result = evaluate_condition(
            {"ref": "check.result.data.nope", "op": "missing"},
            wf,
            task,
            store,
        )

        assert exists_result.passed is True
        assert missing_result.passed is True

    def test_numeric_comparisons(self, store):
        wf, task = _workflow_with_output(store)

        assert evaluate_condition(
            {"ref": "check.result.metrics.score", "op": "gt", "value": 5},
            wf,
            task,
            store,
        ).passed is True
        assert evaluate_condition(
            {"ref": "check.result.metrics.score", "op": "gte", "value": 7},
            wf,
            task,
            store,
        ).passed is True
        assert evaluate_condition(
            {"ref": "check.result.metrics.score", "op": "lt", "value": 8},
            wf,
            task,
            store,
        ).passed is True
        assert evaluate_condition(
            {"ref": "check.result.metrics.score", "op": "lte", "value": 7},
            wf,
            task,
            store,
        ).passed is True

    def test_truthy_and_falsy(self, store):
        wf, task = _workflow_with_output(store)

        truthy_result = evaluate_condition(
            {"ref": "check.result.data.truthy_flag", "op": "truthy"},
            wf,
            task,
            store,
        )
        falsy_result = evaluate_condition(
            {"ref": "check.result.data.falsy_flag", "op": "falsy"},
            wf,
            task,
            store,
        )

        assert truthy_result.passed is True
        assert falsy_result.passed is True

    def test_missing_ref_returns_false_for_non_missing_ops(self, store):
        wf, task = _workflow_with_output(store)

        result = evaluate_condition(
            {"ref": "check.result.data.nope", "op": "eq", "value": 0},
            wf,
            task,
            store,
        )

        assert result.passed is False
        assert result.reason == "reference missing"

    def test_invalid_numeric_comparison_returns_false(self, store):
        wf, task = _workflow_with_output(store)

        result = evaluate_condition(
            {"ref": "check.result.data.non_numeric", "op": "gt", "value": 1},
            wf,
            task,
            store,
        )

        assert result.passed is False
        assert "numeric" in result.reason


class TestConditionValidation:
    def test_invalid_condition_schema_rejected(self):
        spec = {
            "tasks": [
                {
                    "label": "check",
                    "title": "Check",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "check",
                },
                {
                    "label": "branch",
                    "title": "Branch",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "depends_on": ["check"],
                    "prompt": "branch",
                    "run_if": {"ref": "check.result.data.exit_code", "op": "eq"},
                },
            ],
        }

        with pytest.raises(WorkflowSpecError, match="run_if.value"):
            validate_spec(spec)

    def test_invalid_dependency_policy_rejected(self):
        spec = {
            "tasks": [
                {
                    "label": "check",
                    "title": "Check",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "check",
                },
                {
                    "label": "branch",
                    "title": "Branch",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "depends_on": ["check"],
                    "prompt": "branch",
                    "dependency_policy": "first_success",
                },
            ],
        }

        with pytest.raises(WorkflowSpecError, match="dependency_policy"):
            validate_spec(spec)

    def test_non_ancestor_condition_reference_rejected(self):
        workflow = {
            "tasks": [
                {
                    "label": "a",
                    "title": "A",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "a",
                },
                {
                    "label": "b",
                    "title": "B",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "b",
                },
                {
                    "label": "c",
                    "title": "C",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "depends_on": ["a"],
                    "prompt": "c",
                },
            ],
        }
        task = {
            "label": "c",
            "depends_on": ["a"],
        }

        with pytest.raises(ValueError, match="upstream dependency"):
            validate_condition(
                {"ref": "b.result.text", "op": "exists"},
                workflow,
                task,
            )
