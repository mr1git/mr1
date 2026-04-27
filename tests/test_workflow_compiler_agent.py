from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mr1.kazi_runner import MockRunner
from mr1.mr1 import MR1, MR1Process, StateManager
from mr1.scoped_agents import PersistentAgentStore
from mr1.scheduler import Scheduler, WorkflowSpecError, submit_spec_to_disk
from mr1.workflow_compiler import WorkflowCompilerClient, WorkflowCompilerFailure
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


def _spec(tmp_path) -> dict:
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


def _envelope(
    spec: dict,
    *,
    preview: str = "Read notes, then summarize them.",
    assumptions: list[str] | None = None,
    risks: list[str] | None = None,
    needs_confirmation: bool = False,
    confidence: str = "high",
) -> str:
    return json.dumps({
        "preview": preview,
        "spec": spec,
        "assumptions": assumptions or [],
        "risks": risks or [],
        "needs_confirmation": needs_confirmation,
        "confidence": confidence,
    })


class FakeCompiler:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, prompt: str) -> str:
        self.prompts.append((system_prompt, prompt))
        if not self._responses:
            raise AssertionError("no compiler responses configured")
        return self._responses.pop(0)


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


def test_compiler_output_envelope_parses(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    compiler = FakeCompiler(_envelope(_spec(tmp_path)))
    client = WorkflowCompilerClient(
        compiler=compiler,
        scoped_agent_store=agent_store,
    )

    result = client.compile(
        "read notes and summarize them",
        "test context",
        root.agent_id,
        root.agent_id,
        "preview_only",
    )

    assert result.envelope.preview == "Read notes, then summarize them."
    assert result.envelope.confidence == "high"
    assert result.envelope.spec["tasks"][1]["label"] == "summarize"


def test_valid_spec_passes_validation(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    compiler = FakeCompiler(_envelope(_spec(tmp_path)))
    client = WorkflowCompilerClient(compiler=compiler, scoped_agent_store=agent_store)

    result = client.compile(
        "read notes and summarize them",
        "test context",
        root.agent_id,
        root.agent_id,
        "preview_only",
    )

    assert result.envelope.spec["title"] == "Read and summarize"
    assert len(compiler.prompts) == 1


def test_invalid_spec_triggers_one_correction_pass(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    invalid = _spec(tmp_path)
    invalid["tasks"][1]["agent_type"] = "workflow_compiler"
    corrected = _spec(tmp_path)
    compiler = FakeCompiler(_envelope(invalid), _envelope(corrected))
    client = WorkflowCompilerClient(compiler=compiler, scoped_agent_store=agent_store)

    result = client.compile(
        "read notes and summarize them",
        "test context",
        root.agent_id,
        root.agent_id,
        "preview_only",
    )

    assert result.envelope.spec == corrected
    assert len(compiler.prompts) == 2


def test_second_invalid_response_returns_deterministic_failure(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    invalid = _spec(tmp_path)
    invalid["tasks"][1]["agent_type"] = "workflow_compiler"
    compiler = FakeCompiler(_envelope(invalid), _envelope(invalid))
    client = WorkflowCompilerClient(compiler=compiler, scoped_agent_store=agent_store)

    with pytest.raises(WorkflowCompilerFailure, match="workflow compilation failed:"):
        client.compile(
            "read notes and summarize them",
            "test context",
            root.agent_id,
            root.agent_id,
            "preview_only",
        )


def test_submit_if_valid_submits_through_scoped_owner_path(tmp_path, workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    compiler = FakeCompiler(_envelope(_spec(tmp_path)))

    def _submitter(spec: dict, caller_agent_id: str, owner_agent_id: str) -> tuple[str, str]:
        workflow_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            workflow_store,
            owner_agent_id=owner_agent_id,
            caller_agent_id=caller_agent_id,
            scoped_agent_store=agent_store,
        )
        return workflow_id, workflow_id

    client = WorkflowCompilerClient(
        compiler=compiler,
        scoped_agent_store=agent_store,
        submitter=_submitter,
    )

    result = client.compile(
        "read notes and summarize them",
        "test context",
        root.agent_id,
        child.agent_id,
        "submit_if_valid",
    )

    workflow = workflow_store.load_workflow(result.workflow_id)
    assert workflow is not None
    assert workflow.owner_agent_id == child.agent_id
    assert result.workflow_id in agent_store.require_agent(child.agent_id).owned_workflow_ids


def test_compiler_cannot_submit_outside_caller_scope(tmp_path, workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    compiler = FakeCompiler(_envelope(_spec(tmp_path)))

    def _submitter(spec: dict, caller_agent_id: str, owner_agent_id: str) -> tuple[str, str]:
        workflow_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            workflow_store,
            owner_agent_id=owner_agent_id,
            caller_agent_id=caller_agent_id,
            scoped_agent_store=agent_store,
        )
        return workflow_id, workflow_id

    client = WorkflowCompilerClient(
        compiler=compiler,
        scoped_agent_store=agent_store,
        submitter=_submitter,
    )

    with pytest.raises(WorkflowSpecError, match="access denied: owner agent not in caller scope"):
        client.compile(
            "read notes and summarize them",
            "test context",
            left.agent_id,
            right.agent_id,
            "submit_if_valid",
        )


def test_compiler_preview_is_preserved_and_show_json_returns_spec(tmp_path):
    compiler = FakeCompiler(_envelope(
        _spec(tmp_path),
        preview="Read the notes, then summarize them.",
        assumptions=["notes file exists"],
        risks=["summary may omit nuance"],
        needs_confirmation=True,
        confidence="medium",
    ))
    mr1_instance = MR1(
        workflow_store=WorkflowStore(root=tmp_path / "workflows"),
        workflow_runner=MockRunner(),
        workflow_auto_tick=False,
        workflow_authoring_backend="compiler_agent",
        workflow_compiler=compiler,
    )
    mr1_instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
    mr1_instance._process = MagicMock(spec=MR1Process)
    mr1_instance._process.alive = True

    preview = mr1_instance.step("read notes and summarize them")

    assert "Read the notes, then summarize them." in preview
    assert "Assumptions:" in preview
    assert "Risks:" in preview
    assert "Confidence: medium" in preview

    raw_json = mr1_instance.step("show json")
    assert raw_json == json.dumps(mr1_instance._state.pending_workflow["spec"], indent=2)
