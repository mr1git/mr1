"""Spec validation: duplicate labels, unknown deps, cycles, unsupported kinds."""

import pytest

from mr1.scheduler import (
    WorkflowSpecError,
    build_workflow_from_spec,
    submit_spec_to_disk,
    validate_spec,
)
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="user", id="cli")


def _base_task(label, deps=None, task_kind="agent", agent_type="kazi"):
    return {
        "label": label,
        "title": label.upper(),
        "task_kind": task_kind,
        "agent_type": agent_type,
        "prompt": "x",
        "depends_on": deps or [],
    }


class TestValidationRejects:
    def test_non_dict_spec(self):
        with pytest.raises(WorkflowSpecError):
            validate_spec("not a dict")  # type: ignore[arg-type]

    def test_missing_tasks(self):
        with pytest.raises(WorkflowSpecError):
            validate_spec({"title": "no tasks"})

    def test_empty_tasks_list(self):
        with pytest.raises(WorkflowSpecError):
            validate_spec({"title": "empty", "tasks": []})

    def test_missing_label(self):
        with pytest.raises(WorkflowSpecError, match="label"):
            validate_spec({"tasks": [{"title": "no label"}]})

    def test_duplicate_labels(self):
        spec = {"tasks": [_base_task("a"), _base_task("a")]}
        with pytest.raises(WorkflowSpecError, match="duplicate label"):
            validate_spec(spec)

    def test_unknown_dependency_label(self):
        spec = {"tasks": [_base_task("a"), _base_task("b", deps=["ghost"])]}
        with pytest.raises(WorkflowSpecError, match="unknown label"):
            validate_spec(spec)

    def test_cycle(self):
        spec = {
            "tasks": [
                _base_task("a", deps=["b"]),
                _base_task("b", deps=["a"]),
            ],
        }
        with pytest.raises(WorkflowSpecError, match="cycle"):
            validate_spec(spec)

    def test_unsupported_task_kind(self):
        spec = {"tasks": [_base_task("a", task_kind="watcher")]}
        with pytest.raises(WorkflowSpecError, match="task_kind"):
            validate_spec(spec)

    def test_unsupported_agent_type(self):
        spec = {"tasks": [_base_task("a", agent_type="mrn")]}
        with pytest.raises(WorkflowSpecError, match="agent_type"):
            validate_spec(spec)


class TestValidationAccepts:
    def test_simple_linear_dag(self):
        validate_spec({
            "tasks": [
                _base_task("a"),
                _base_task("b", deps=["a"]),
                _base_task("c", deps=["b"]),
            ],
        })


class TestBuildWorkflow:
    def test_resolves_labels_to_task_ids(self):
        spec = {
            "title": "resolve",
            "tasks": [
                _base_task("a"),
                _base_task("b", deps=["a"]),
            ],
        }
        wf = build_workflow_from_spec(spec, PROV)
        assert wf.title == "resolve"
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]
        assert wf.tasks[b_id].depends_on == [a_id]
        assert all(t.task_id.startswith("tk-") for t in wf.tasks.values())
        assert wf.workflow_id.startswith("wf-")


class TestSubmitDoesNotLeakOnFailure:
    def test_invalid_spec_leaves_store_empty(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        spec = {
            "tasks": [
                _base_task("a", deps=["b"]),
                _base_task("b", deps=["a"]),
            ],
        }
        with pytest.raises(WorkflowSpecError):
            submit_spec_to_disk(spec, PROV, store)
        assert store.list_workflows() == []
