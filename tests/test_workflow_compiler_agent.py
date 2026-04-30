from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mr1.kazi_runner import MockRunner
from mr1.memory_curator import EvidenceRef, InsightStore, MemoryInsight
from mr1.memory_graph import MemoryGraph, MemoryGraphStore, MemoryNode, agent_node_id
from mr1.memory_retrieval import update_memory_retrieval
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
    memory_refs_used: list[str] | None = None,
) -> str:
    return json.dumps({
        "preview": preview,
        "spec": spec,
        "assumptions": assumptions or [],
        "risks": risks or [],
        "needs_confirmation": needs_confirmation,
        "confidence": confidence,
        "memory_refs_used": memory_refs_used or [],
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


def _seed_memory(tmp_path):
    InsightStore(tmp_path / "insights").save_insights({
        "insight:capability_friction:capability:read_file": MemoryInsight(
            insight_id="insight:capability_friction:capability:read_file",
            insight_type="capability_friction",
            title="read_file friction",
            summary="Use narrower read scopes",
            confidence=0.9,
            severity="WARNING",
            recommended_action="consider narrowing scope",
            evidence=[EvidenceRef(source_type="query", source_id="query:test", reason="fixture")],
            related_nodes=["capability:read_file"],
            created_at="2026-04-20T00:00:00+00:00",
            updated_at="2026-04-20T00:00:00+00:00",
            status="active",
            metadata={"capability_id": "capability:read_file"},
        )
    })
    MemoryGraphStore(tmp_path / "graph").save_graph(MemoryGraph(
        nodes={
            agent_node_id("ag-root"): MemoryNode(
                node_id=agent_node_id("ag-root"),
                node_type="Agent",
                name="MR1",
            )
        }
    ))


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

    def _submitter(
        spec: dict,
        caller_agent_id: str,
        owner_agent_id: str,
        workflow_metadata: dict | None,
    ) -> tuple[str, str]:
        workflow_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            workflow_store,
            owner_agent_id=owner_agent_id,
            caller_agent_id=caller_agent_id,
            workflow_metadata=workflow_metadata,
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

    def _submitter(
        spec: dict,
        caller_agent_id: str,
        owner_agent_id: str,
        workflow_metadata: dict | None,
    ) -> tuple[str, str]:
        workflow_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            workflow_store,
            owner_agent_id=owner_agent_id,
            caller_agent_id=caller_agent_id,
            workflow_metadata=workflow_metadata,
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


def test_preview_pending_draft_preserves_memory_fields(tmp_path):
    compiler = FakeCompiler(_envelope(
        _spec(tmp_path),
        needs_confirmation=True,
        memory_refs_used=["insight:capability_friction:capability:read_file"],
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

    mr1_instance.step("read notes and summarize them")

    pending = mr1_instance._state.pending_workflow
    assert pending is not None
    assert pending["memory_refs_used"] == ["insight:capability_friction:capability:read_file"]


def test_compile_includes_memory_payload_when_enabled(tmp_path, agent_store):
    _seed_memory(tmp_path)
    root = agent_store.ensure_root_agent()
    compiler = FakeCompiler(_envelope(_spec(tmp_path)))
    client = WorkflowCompilerClient(compiler=compiler, scoped_agent_store=agent_store)

    result = client.compile(
        "read notes and summarize them",
        "test context",
        root.agent_id,
        root.agent_id,
        "preview_only",
        use_memory=True,
        memory_limit=3,
    )

    payload = json.loads(compiler.prompts[0][1])
    assert payload["memory"]["enabled"] is True
    assert payload["memory"]["memory_limit"] == 3
    assert payload["memory"]["tools_used"] == [
        "memory_search",
        "memory_insights_search",
        "memory_graph_top_workflows",
        "memory_graph_capabilities",
        "memory_graph_failures",
        "memory_graph_agent_summary",
    ]
    assert "memory_search" in payload["memory"]["prefetched_context"]
    assert result.compiled_with_memory is True


def test_compile_omits_memory_payload_when_disabled(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    compiler = FakeCompiler(_envelope(_spec(tmp_path)))
    client = WorkflowCompilerClient(compiler=compiler, scoped_agent_store=agent_store)

    client.compile(
        "read notes and summarize them",
        "test context",
        root.agent_id,
        root.agent_id,
        "preview_only",
        use_memory=False,
    )

    payload = json.loads(compiler.prompts[0][1])
    assert payload["memory"]["enabled"] is False
    assert payload["memory"]["prefetched_context"] == {}


def test_unknown_memory_refs_produce_warnings_without_failure(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    compiler = FakeCompiler(json.dumps({
        "preview": "Read notes, then summarize them.",
        "spec": _spec(tmp_path),
        "assumptions": [],
        "risks": [],
        "needs_confirmation": False,
        "confidence": "high",
        "memory_refs_used": ["insight:missing"],
    }))
    client = WorkflowCompilerClient(compiler=compiler, scoped_agent_store=agent_store)

    result = client.compile(
        "read notes and summarize them",
        "test context",
        root.agent_id,
        root.agent_id,
        "preview_only",
    )

    assert result.envelope.memory_refs_used == ["insight:missing"]
    assert result.memory_ref_warnings == ["unknown memory ref: insight:missing"]


def test_submit_if_valid_persists_workflow_memory_metadata(tmp_path, workflow_store, agent_store):
    _seed_memory(tmp_path)
    update_memory_retrieval(tmp_path)
    root = agent_store.ensure_root_agent()
    compiler = FakeCompiler(json.dumps({
        "preview": "Read notes, then summarize them.",
        "spec": _spec(tmp_path),
        "assumptions": [],
        "risks": [],
        "needs_confirmation": False,
        "confidence": "high",
        "memory_refs_used": ["retrieval:insight:insight:capability_friction:capability:read_file"],
    }))

    def _submitter(
        spec: dict,
        caller_agent_id: str,
        owner_agent_id: str,
        workflow_metadata: dict | None,
    ) -> tuple[str, str]:
        workflow_id = submit_spec_to_disk(
            spec,
            Provenance(type="user", id="cli"),
            workflow_store,
            owner_agent_id=owner_agent_id,
            caller_agent_id=caller_agent_id,
            workflow_metadata=workflow_metadata,
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
        root.agent_id,
        "submit_if_valid",
        use_memory=True,
    )

    workflow = workflow_store.load_workflow(result.workflow_id)
    assert workflow is not None
    assert workflow.metadata["compiled_with_memory"] is True
    assert workflow.metadata["memory_refs_used"] == ["retrieval:insight:insight:capability_friction:capability:read_file"]
    assert workflow.metadata["memory_tools_used"][0] == "memory_search"
