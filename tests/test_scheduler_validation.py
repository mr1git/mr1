"""Spec validation: duplicate labels, unknown deps, cycles, supported kinds."""

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
    task = {
        "label": label,
        "title": label.upper(),
        "task_kind": task_kind,
        "prompt": "x",
        "depends_on": deps or [],
    }
    if task_kind == "agent":
        task["agent_type"] = agent_type
    return task


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
        spec = {"tasks": [_base_task("a", task_kind="command")]}
        with pytest.raises(WorkflowSpecError, match="task_kind"):
            validate_spec(spec)

    def test_unsupported_agent_type(self):
        spec = {"tasks": [_base_task("a", agent_type="mrn")]}
        with pytest.raises(WorkflowSpecError, match="agent_type"):
            validate_spec(spec)

    def test_rejects_inputs_non_list(self):
        spec = {"tasks": [{**_base_task("a"), "inputs": "bad"}]}
        with pytest.raises(WorkflowSpecError, match="inputs must be a list"):
            validate_spec(spec)

    def test_rejects_unknown_input_label(self):
        spec = {
            "tasks": [
                _base_task("a"),
                {
                    **_base_task("b", deps=["a"]),
                    "inputs": [{"name": "x", "from": "ghost.result.text"}],
                },
            ],
        }
        with pytest.raises(WorkflowSpecError, match="input source label"):
            validate_spec(spec)

    def test_rejects_non_ancestor_input_reference(self):
        spec = {
            "tasks": [
                _base_task("a"),
                _base_task("b"),
                {
                    **_base_task("c", deps=["a"]),
                    "inputs": [{"name": "x", "from": "b.result.text"}],
                },
            ],
        }
        with pytest.raises(WorkflowSpecError, match="upstream dependency"):
            validate_spec(spec)

    def test_rejects_unsupported_input_root(self):
        spec = {
            "tasks": [
                _base_task("a"),
                {
                    **_base_task("b", deps=["a"]),
                    "inputs": [{"name": "x", "from": "a.nope"}],
                },
            ],
        }
        with pytest.raises(WorkflowSpecError, match="unsupported input reference root"):
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

    def test_accepts_watcher_task(self):
        validate_spec({
            "tasks": [
                {
                    "label": "wait",
                    "title": "Wait",
                    "task_kind": "watcher",
                    "watcher_type": "manual_event",
                    "watch_config": {"event": "approved"},
                },
                _base_task("run", deps=["wait"]),
            ],
        })

    def test_accepts_tool_task(self):
        validate_spec({
            "tasks": [
                {
                    "label": "read_notes",
                    "title": "Read notes",
                    "task_kind": "tool",
                    "tool_type": "read_file",
                    "tool_config": {"path": "notes.txt"},
                },
                _base_task("run", deps=["read_notes"]),
            ],
        })

    def test_accepts_transitive_input_source(self):
        validate_spec({
            "tasks": [
                _base_task("a"),
                _base_task("b", deps=["a"]),
                {
                    **_base_task("c", deps=["b"]),
                    "inputs": [{"name": "upstream", "from": "a.result.text"}],
                },
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

    def test_parses_inputs(self):
        spec = {
            "tasks": [
                _base_task("a"),
                {
                    **_base_task("b", deps=["a"]),
                    "inputs": [{"name": "upstream", "from": "a.result.text"}],
                },
            ],
        }
        wf = build_workflow_from_spec(spec, PROV)
        b = wf.task_by_label("b")
        assert len(b.inputs) == 1
        assert b.inputs[0].name == "upstream"
        assert b.inputs[0].from_ref == "a.result.text"


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
