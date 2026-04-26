"""Tests for Phase 5 workflow authoring."""

from __future__ import annotations

import json

import pytest

from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler, validate_spec
from mr1.workflow_authoring import WorkflowAuthoringService
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="agent", id="MR1")


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


def _three_step_spec(tmp_path) -> dict:
    return {
        "title": "Read and summarize",
        "tasks": [
            {
                "label": "read_notes",
                "title": "Read notes",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": str(tmp_path / "notes.txt")},
            },
            {
                "label": "python_version",
                "title": "Python version",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {"argv": ["python3", "--version"]},
            },
            {
                "label": "summarize",
                "title": "Summarize",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["read_notes", "python_version"],
                "inputs": [
                    {"name": "notes", "from": "read_notes.result.text"},
                    {"name": "version", "from": "python_version.result.data.stdout"},
                ],
                "prompt": "Summarize the notes and Python version.",
            },
        ],
    }


def _complex_spec(tmp_path) -> dict:
    return {
        "title": "Complex workflow",
        "tasks": [
            {
                "label": "read_notes",
                "title": "Read notes",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": str(tmp_path / "notes.txt")},
            },
            {
                "label": "python_version",
                "title": "Python version",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {"argv": ["python3", "--version"]},
            },
            {
                "label": "summarize",
                "title": "Summarize",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["read_notes", "python_version"],
                "inputs": [
                    {"name": "notes", "from": "read_notes.result.text"},
                    {"name": "version", "from": "python_version.result.data.stdout"},
                ],
                "prompt": "Summarize the notes and Python version.",
            },
            {
                "label": "write_report",
                "title": "Write report",
                "task_kind": "tool",
                "tool_type": "write_file",
                "depends_on": ["summarize"],
                "inputs": [
                    {"name": "summary", "from": "summarize.result.text"},
                ],
                "tool_config": {
                    "path": str(tmp_path / "report.txt"),
                    "content": "placeholder",
                    "overwrite": True,
                },
            },
        ],
    }


class TestAuthoringGeneration:
    def test_natural_language_to_valid_workflow_json(self, scheduler, store, tmp_path):
        compiler = FakeCompiler(json.dumps(_three_step_spec(tmp_path)))
        service = WorkflowAuthoringService(scheduler, store, compiler=compiler)

        spec = service.generate_spec("Read a file, check Python version, and summarize")
        validate_spec(spec)

        assert compiler.prompts
        assert "Available workflow schema:" in compiler.prompts[0][0]
        assert "Available capabilities:" in compiler.prompts[0][0]
        assert '"item_shape"' in compiler.prompts[0][0]
        assert "\"from\": \"<label>.<reference>\"" in compiler.prompts[0][0]
        assert "inputs must NEVER be strings" in compiler.prompts[0][0]
        assert '"dependency_policy"' in compiler.prompts[0][0]
        assert '"run_if"' in compiler.prompts[0][0]
        assert '"skipped"' in compiler.prompts[0][0]
        assert '"name": "shell_command"' in compiler.prompts[0][0]
        assert '"argv"' in compiler.prompts[0][0]
        assert "result.data.stdout" in compiler.prompts[0][0]
        assert compiler.prompts[0][1].startswith("Mode: create")
        assert spec["tasks"][0]["task_kind"] == "tool"
        assert spec["tasks"][1]["tool_type"] == "shell_command"
        assert spec["tasks"][2]["task_kind"] == "agent"

    def test_tool_and_agent_composition_and_dataflow(self, scheduler, store, tmp_path):
        compiler = FakeCompiler(json.dumps(_three_step_spec(tmp_path)))
        service = WorkflowAuthoringService(scheduler, store, compiler=compiler)

        spec = service.generate_spec("Read a file, check Python version, and summarize")
        summarize = spec["tasks"][2]

        assert [task["task_kind"] for task in spec["tasks"][:2]] == ["tool", "tool"]
        assert summarize["task_kind"] == "agent"
        assert summarize["depends_on"] == ["read_notes", "python_version"]
        assert summarize["inputs"] == [
            {"name": "notes", "from": "read_notes.result.text"},
            {"name": "version", "from": "python_version.result.data.stdout"},
        ]
        assert "read_notes.result.text" not in summarize["prompt"]

    def test_generated_workflow_uses_object_inputs_not_string_inputs(self, scheduler, store, tmp_path):
        compiler = FakeCompiler(json.dumps(_three_step_spec(tmp_path)))
        service = WorkflowAuthoringService(scheduler, store, compiler=compiler)

        spec = service.generate_spec(
            f"read {tmp_path / 'test_notes.txt'}, run python3 --version, summarize both"
        )

        summarize = spec["tasks"][2]
        assert summarize["inputs"] == [
            {"name": "notes", "from": "read_notes.result.text"},
            {"name": "version", "from": "python_version.result.data.stdout"},
        ]
        assert all(isinstance(item, dict) for item in summarize["inputs"])
        assert not any(isinstance(item, str) for item in summarize["inputs"])


class TestValidationAndCorrection:
    def test_validation_failure_runs_one_correction_pass(self, scheduler, store, tmp_path):
        invalid = _three_step_spec(tmp_path)
        invalid["tasks"][2]["agent_type"] = "mr2"
        corrected = _three_step_spec(tmp_path)
        compiler = FakeCompiler(json.dumps(corrected))
        service = WorkflowAuthoringService(scheduler, store, compiler=compiler)

        result = service.validate_and_maybe_fix(invalid)

        assert result.ok is True
        assert result.corrected is True
        assert result.spec == corrected
        assert len(compiler.prompts) == 1

    def test_validation_failure_after_one_correction_asks_for_clarification(self, scheduler, store, tmp_path):
        invalid = _three_step_spec(tmp_path)
        invalid["tasks"][2]["agent_type"] = "mr2"
        still_invalid = _three_step_spec(tmp_path)
        still_invalid["tasks"][2]["agent_type"] = "mr3"
        compiler = FakeCompiler(json.dumps(still_invalid))
        service = WorkflowAuthoringService(scheduler, store, compiler=compiler)

        result = service.validate_and_maybe_fix(invalid)

        assert result.ok is False
        assert result.corrected is True
        assert "agent_type" in (result.error or "")

    def test_fix_prompt_includes_schema_and_capability_metadata_and_repairs_string_inputs(self, scheduler, store, tmp_path):
        invalid = _three_step_spec(tmp_path)
        invalid["tasks"][2]["inputs"] = ["read_notes.result.text"]
        corrected = _three_step_spec(tmp_path)
        compiler = FakeCompiler(json.dumps(corrected))
        service = WorkflowAuthoringService(scheduler, store, compiler=compiler)

        result = service.validate_and_maybe_fix(invalid)

        assert result.ok is True
        assert result.spec == corrected
        assert len(compiler.prompts) == 1
        system_prompt, fix_prompt = compiler.prompts[0]
        assert "Available workflow schema:" in system_prompt
        assert "Available capabilities:" in system_prompt
        assert "Available workflow schema:" in fix_prompt
        assert "Available capabilities:" in fix_prompt
        assert "Fix only schema/config errors in the workflow JSON below." in fix_prompt
        assert "Preserve user intent." in fix_prompt
        assert '"inputs": [' in fix_prompt
        assert '"read_notes.result.text"' in fix_prompt


class TestPreviewing:
    def test_preview_simple_vs_complex(self, scheduler, store, tmp_path):
        service = WorkflowAuthoringService(scheduler, store, compiler=lambda *_: "{}")

        simple_preview, simple_complexity = service.preview(_three_step_spec(tmp_path))
        complex_preview, complex_complexity = service.preview(_complex_spec(tmp_path))

        assert simple_complexity == "simple"
        assert "This workflow will:" in simple_preview
        assert complex_complexity == "complex"
        assert "Reply with `show json`" in complex_preview


class TestSubmissionAndRewrite:
    def test_safe_in_place_modification(self, scheduler, store, tmp_path):
        service = WorkflowAuthoringService(scheduler, store, compiler=lambda *_: "{}")
        wf_id = scheduler.submit_workflow(_three_step_spec(tmp_path), PROV)
        original = store.load_workflow(wf_id)
        assert original is not None

        modified = _three_step_spec(tmp_path)
        modified["tasks"].append(
            {
                "label": "final_summary",
                "title": "Final summary",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["summarize"],
                "inputs": [{"name": "summary", "from": "summarize.result.text"}],
                "prompt": "Create a final summary.",
            }
        )

        result = service.submit(
            modified,
            created_by=PROV,
            target_workflow_id=wf_id,
        )

        rewritten = store.load_workflow(wf_id)
        assert result.in_place is True
        assert result.workflow_id == wf_id
        assert rewritten is not None
        assert "final_summary" in rewritten.label_to_task_id
        assert rewritten.task_by_label("read_notes").task_id == original.task_by_label("read_notes").task_id

    def test_started_task_change_forces_new_workflow(self, scheduler, store, tmp_path):
        service = WorkflowAuthoringService(scheduler, store, compiler=lambda *_: "{}")
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        wf_id = scheduler.submit_workflow(_three_step_spec(tmp_path), PROV)
        scheduler.tick()
        original = store.load_workflow(wf_id)
        assert original is not None
        assert original.task_by_label("read_notes").status is TaskStatus.SUCCEEDED

        changed = _three_step_spec(tmp_path)
        changed["tasks"][0]["tool_config"]["path"] = str(tmp_path / "other.txt")

        result = service.submit(
            changed,
            created_by=PROV,
            target_workflow_id=wf_id,
        )

        assert result.in_place is False
        assert result.workflow_id != wf_id
