from __future__ import annotations

import json

import pytest

from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler, validate_spec
from mr1.workflow_authoring import WorkflowAuthoringService
from mr1.workflow_store import WorkflowStore


class FakeCompiler:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, prompt: str) -> str:
        self.prompts.append((system_prompt, prompt))
        if not self._responses:
            raise AssertionError("no compiler responses left")
        return self._responses.pop(0)


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def scheduler(store):
    sched = Scheduler(store, MockRunner(), auto_tick=False, agent_id="scheduler")
    try:
        yield sched
    finally:
        sched.shutdown()


def _single_branch_spec() -> dict:
    return {
        "title": "Conditional branch",
        "tasks": [
            {
                "label": "check",
                "title": "Check",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {"argv": ["python3", "--version"]},
            },
            {
                "label": "success_path",
                "title": "Success path",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check"],
                "run_if": {
                    "ref": "check.result.text",
                    "op": "contains",
                    "value": "Python",
                },
                "prompt": "Handle the success case.",
            },
            {
                "label": "final",
                "title": "Final",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["success_path"],
                "prompt": "Summarize the branch result.",
            },
        ],
    }


def _join_spec() -> dict:
    return {
        "title": "Branch join",
        "tasks": [
            {
                "label": "check",
                "title": "Check",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {"argv": ["python3", "--version"]},
            },
            {
                "label": "success_path",
                "title": "Success path",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check"],
                "run_if": {
                    "ref": "check.result.text",
                    "op": "contains",
                    "value": "Python",
                },
                "prompt": "Handle the success case.",
            },
            {
                "label": "failure_path",
                "title": "Failure path",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check"],
                "run_if": {
                    "ref": "check.result.text",
                    "op": "missing",
                },
                "prompt": "Handle the failure case.",
            },
            {
                "label": "final",
                "title": "Final",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["success_path", "failure_path"],
                "dependency_policy": "any_succeeded",
                "inputs": [
                    {"name": "selected_text", "from": "success_path.result.text"},
                ],
                "prompt": "Summarize the winning branch.",
            },
        ],
    }


def _mixed_dependency_spec() -> dict:
    spec = _single_branch_spec()
    spec["tasks"][2]["depends_on"] = ["check", "success_path"]
    return spec


def _collision_spec() -> dict:
    return {
        "title": "Collision join",
        "tasks": [
            {
                "label": "source",
                "title": "Source",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {"argv": ["python3", "--version"]},
            },
            {
                "label": "check-contains-one",
                "title": "Dash label",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["source"],
                "run_if": {
                    "ref": "source.result.text",
                    "op": "contains",
                    "value": "Python",
                },
                "prompt": "dash",
            },
            {
                "label": "check_contains_one",
                "title": "Snake label",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["source"],
                "run_if": {
                    "ref": "source.result.text",
                    "op": "contains",
                    "value": "Python",
                },
                "prompt": "snake",
            },
            {
                "label": "final",
                "title": "Final",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check-contains-one", "check_contains_one"],
                "dependency_policy": "any_succeeded",
                "prompt": "final",
            },
        ],
    }


def _plain_spec() -> dict:
    return {
        "title": "Plain workflow",
        "tasks": [
            {
                "label": "read_notes",
                "title": "Read notes",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {"argv": ["python3", "--version"]},
            },
            {
                "label": "summarize",
                "title": "Summarize",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["read_notes"],
                "inputs": [{"name": "notes", "from": "read_notes.result.text"}],
                "prompt": "Summarize the notes.",
            },
        ],
    }


def _service_for_spec(scheduler, store, spec: dict) -> tuple[WorkflowAuthoringService, FakeCompiler]:
    compiler = FakeCompiler(json.dumps(spec))
    return WorkflowAuthoringService(scheduler, store, compiler=compiler), compiler


class TestBranchAwareCompilation:
    def test_branching_workflow_generates_additional_inputs(self, scheduler, store):
        service, _ = _service_for_spec(scheduler, store, _single_branch_spec())

        spec = service.generate_spec("compile branch workflow")
        final = spec["tasks"][2]

        assert final["inputs"] == [
            {"name": "success_path_status", "from": "success_path.status"},
            {"name": "success_path_condition", "from": "success_path.condition_result"},
            {"name": "success_path_skip_reason", "from": "success_path.skip_reason"},
        ]
        validate_spec(spec)

    def test_join_node_receives_branch_context_and_preserves_existing_inputs(self, scheduler, store):
        service, _ = _service_for_spec(scheduler, store, _join_spec())

        spec = service.generate_spec("compile join workflow")
        final = spec["tasks"][3]

        assert final["inputs"] == [
            {"name": "selected_text", "from": "success_path.result.text"},
            {"name": "success_path_status", "from": "success_path.status"},
            {"name": "success_path_condition", "from": "success_path.condition_result"},
            {"name": "success_path_skip_reason", "from": "success_path.skip_reason"},
            {"name": "failure_path_status", "from": "failure_path.status"},
            {"name": "failure_path_condition", "from": "failure_path.condition_result"},
            {"name": "failure_path_skip_reason", "from": "failure_path.skip_reason"},
        ]
        validate_spec(spec)

    def test_only_conditional_dependencies_get_condition_and_skip_reason(self, scheduler, store):
        service, _ = _service_for_spec(scheduler, store, _mixed_dependency_spec())

        spec = service.generate_spec("compile mixed branch workflow")
        final_inputs = spec["tasks"][2]["inputs"]
        names = {item["name"] for item in final_inputs}

        assert "check_status" in names
        assert "check_condition" not in names
        assert "check_skip_reason" not in names
        assert "success_path_condition" in names
        assert "success_path_skip_reason" in names

    def test_input_naming_is_deterministic_and_collision_safe(self, scheduler, store):
        service, _ = _service_for_spec(scheduler, store, _collision_spec())

        spec = service.generate_spec("compile collision workflow")
        final_inputs = spec["tasks"][3]["inputs"]
        names = [item["name"] for item in final_inputs]

        assert "check_contains_one_status" in names
        assert "check_contains_one_condition" in names
        assert "check_contains_one_skip_reason" in names
        assert "check_contains_one_2_status" in names
        assert "check_contains_one_2_condition" in names
        assert "check_contains_one_2_skip_reason" in names

    def test_duplicate_reserved_inputs_are_replaced_with_canonical_bindings(self, scheduler, store):
        spec = _join_spec()
        spec["tasks"][3]["inputs"] = [
            {"name": "success_path_status", "from": "wrong.result.text"},
            {"name": "success_path_status", "from": "success_path.status"},
            {"name": "success_path_condition", "from": "wrong.result.text"},
            {"name": "selected_text", "from": "success_path.result.text"},
        ]
        service, _ = _service_for_spec(scheduler, store, spec)

        normalized = service.generate_spec("compile dedup workflow")
        final_inputs = normalized["tasks"][3]["inputs"]

        assert final_inputs.count({"name": "success_path_status", "from": "success_path.status"}) == 1
        assert final_inputs.count({"name": "success_path_condition", "from": "success_path.condition_result"}) == 1
        assert {"name": "success_path_status", "from": "wrong.result.text"} not in final_inputs
        assert {"name": "success_path_condition", "from": "wrong.result.text"} not in final_inputs

    def test_non_branch_workflows_remain_unchanged(self, scheduler, store):
        original = _plain_spec()
        service, _ = _service_for_spec(scheduler, store, original)

        spec = service.generate_spec("compile plain workflow")

        assert spec == original

    def test_prompt_augmentation_is_added_once(self, scheduler, store):
        service, _ = _service_for_spec(scheduler, store, _join_spec())

        spec = service.generate_spec("compile prompt workflow")
        final_prompt = spec["tasks"][3]["prompt"]
        assert final_prompt.count("You are summarizing a conditional workflow.") == 1

        result = service.validate_and_maybe_fix(spec)
        assert result.ok is True
        assert result.spec is not None
        assert result.spec["tasks"][3]["prompt"].count("You are summarizing a conditional workflow.") == 1

    def test_schema_metadata_exposes_branch_context_references(self, scheduler, store):
        service, compiler = _service_for_spec(scheduler, store, _plain_spec())

        service.generate_spec("compile plain workflow")
        system_prompt, _ = compiler.prompts[0]

        assert "<label>.status" in system_prompt
        assert "<label>.condition_result" in system_prompt
        assert "<label>.skip_reason" in system_prompt
        assert "success_path_condition" in system_prompt
