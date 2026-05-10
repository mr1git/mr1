from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.memory_reset import MemoryResetResult, reset_memory
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_store import WorkflowStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_root(tmp_path: Path) -> Path:
    root = tmp_path / "memory"
    root.mkdir()
    return root


def _seed_active(memory_root: Path) -> None:
    active = memory_root / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "mr1_state.json").write_text(
        json.dumps({"decisions": ["old decision"], "tasks": {"t1": "done"}}),
        encoding="utf-8",
    )
    (active / "mr1_context.md").write_text("# Old context", encoding="utf-8")
    turns = active / "mr1_turns"
    turns.mkdir(exist_ok=True)
    (turns / "turn-001").mkdir()
    (turns / "turn-001" / "data.json").write_text("{}", encoding="utf-8")


def _seed_graph(memory_root: Path) -> None:
    graph_dir = memory_root / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "graph.json").write_text('{"nodes":{},"edges":{}}', encoding="utf-8")
    (graph_dir / "cursor.json").write_text('{"last_processed_event_index":42}', encoding="utf-8")


def _seed_insights(memory_root: Path) -> None:
    insights_dir = memory_root / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "insights.json",
        "cursor.json",
        "curation_runs.jsonl",
        "feedback.jsonl",
        "feedback_cursor.json",
    ):
        (insights_dir / name).write_text("{}", encoding="utf-8")


def _seed_rag(memory_root: Path) -> None:
    chroma = memory_root / "rag" / ".chroma"
    chroma.mkdir(parents=True, exist_ok=True)
    (chroma / "chroma.sqlite3").write_text("fake", encoding="utf-8")

    dumps = memory_root / "dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    (dumps / "distill-20260501.md").write_text("old dump", encoding="utf-8")
    (dumps / "distill-20260502.txt").write_text("old dump 2", encoding="utf-8")

    retrieval = memory_root / "retrieval"
    retrieval.mkdir(parents=True, exist_ok=True)
    (retrieval / "documents.jsonl").write_text("{}", encoding="utf-8")
    (retrieval / "manifest.json").write_text("{}", encoding="utf-8")


def _seed_protected(memory_root: Path) -> None:
    for protected_dir in ("agents", "messages", "workflows", "capability_approvals"):
        d = memory_root / protected_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "record.json").write_text('{"protected": true}', encoding="utf-8")
    events_dir = memory_root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    (events_dir / "events.jsonl").write_text('{"event":"important"}\n', encoding="utf-8")
    snapshots_dir = memory_root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    (snapshots_dir / "snap.json").write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# reset_memory — active scope
# ---------------------------------------------------------------------------

class TestResetActive:
    def test_reset_active_when_files_exist(self, tmp_path):
        root = _make_memory_root(tmp_path)
        _seed_active(root)

        result = reset_memory(root, scope="active")

        assert result.scope == "active"
        state = root / "active" / "mr1_state.json"
        assert not state.exists()
        assert not (root / "active" / "mr1_context.md").exists()
        assert not (root / "active" / "mr1_turns" / "turn-001").exists()
        assert (root / "active" / "mr1_turns").is_dir()
        assert len(result.deleted_paths) > 0
        assert result.recreated_paths == []

    def test_reset_active_when_files_missing(self, tmp_path):
        root = _make_memory_root(tmp_path)

        result = reset_memory(root, scope="active")

        assert result.scope == "active"
        assert not result.warnings
        assert len(result.skipped_paths) > 0
        assert result.recreated_paths == []

    def test_reset_active_returns_required_fields(self, tmp_path):
        root = _make_memory_root(tmp_path)
        _seed_active(root)

        result = reset_memory(root, scope="active")
        d = result.to_dict()

        assert set(d.keys()) == {"scope", "deleted_paths", "recreated_paths", "skipped_paths", "warnings"}
        assert d["scope"] == "active"
        assert isinstance(d["deleted_paths"], list)
        assert isinstance(d["recreated_paths"], list)
        assert isinstance(d["skipped_paths"], list)
        assert isinstance(d["warnings"], list)


# ---------------------------------------------------------------------------
# reset_memory — graph scope
# ---------------------------------------------------------------------------

class TestResetGraph:
    def test_reset_graph_removes_graph_and_cursor(self, tmp_path):
        root = _make_memory_root(tmp_path)
        _seed_graph(root)

        result = reset_memory(root, scope="graph")

        assert result.scope == "graph"
        assert not (root / "graph" / "graph.json").exists()
        assert not (root / "graph" / "cursor.json").exists()
        assert len(result.deleted_paths) == 2

    def test_reset_graph_when_missing(self, tmp_path):
        root = _make_memory_root(tmp_path)

        result = reset_memory(root, scope="graph")

        assert not result.warnings
        assert len(result.skipped_paths) == 2
        assert result.deleted_paths == []


# ---------------------------------------------------------------------------
# reset_memory — insights scope
# ---------------------------------------------------------------------------

class TestResetInsights:
    def test_reset_insights_removes_all_artifacts(self, tmp_path):
        root = _make_memory_root(tmp_path)
        _seed_insights(root)

        result = reset_memory(root, scope="insights")

        assert result.scope == "insights"
        for name in (
            "insights.json",
            "cursor.json",
            "curation_runs.jsonl",
            "feedback.jsonl",
            "feedback_cursor.json",
        ):
            assert not (root / "insights" / name).exists()
        assert len(result.deleted_paths) == 5

    def test_reset_insights_when_missing(self, tmp_path):
        root = _make_memory_root(tmp_path)

        result = reset_memory(root, scope="insights")

        assert not result.warnings
        assert len(result.skipped_paths) == 5
        assert result.deleted_paths == []


# ---------------------------------------------------------------------------
# reset_memory — rag scope
# ---------------------------------------------------------------------------

class TestResetRag:
    def test_reset_rag_removes_chroma_dumps_and_retrieval(self, tmp_path):
        root = _make_memory_root(tmp_path)
        _seed_rag(root)

        result = reset_memory(root, scope="rag")

        assert result.scope == "rag"
        assert not (root / "rag" / ".chroma").exists()
        assert not (root / "dumps" / "distill-20260501.md").exists()
        assert not (root / "dumps" / "distill-20260502.txt").exists()
        assert not (root / "retrieval" / "documents.jsonl").exists()
        assert not (root / "retrieval" / "manifest.json").exists()
        assert len(result.deleted_paths) >= 3

    def test_reset_rag_when_missing(self, tmp_path):
        root = _make_memory_root(tmp_path)

        result = reset_memory(root, scope="rag")

        assert not result.warnings
        assert result.deleted_paths == []


# ---------------------------------------------------------------------------
# reset_memory — all scope
# ---------------------------------------------------------------------------

class TestResetAll:
    def test_reset_all_covers_all_scopes(self, tmp_path):
        root = _make_memory_root(tmp_path)
        _seed_active(root)
        _seed_graph(root)
        _seed_insights(root)
        _seed_rag(root)

        result = reset_memory(root, scope="all")

        assert result.scope == "all"
        assert not (root / "active" / "mr1_state.json").exists()
        assert not (root / "graph" / "graph.json").exists()
        assert not (root / "insights" / "insights.json").exists()
        assert not (root / "rag" / ".chroma").exists()

    def test_reset_all_does_not_touch_protected_paths(self, tmp_path):
        root = _make_memory_root(tmp_path)
        _seed_active(root)
        _seed_graph(root)
        _seed_insights(root)
        _seed_rag(root)
        _seed_protected(root)

        reset_memory(root, scope="all")

        for protected_dir in ("agents", "messages", "workflows", "capability_approvals"):
            assert (root / protected_dir / "record.json").exists()
        assert (root / "events" / "events.jsonl").exists()
        assert (root / "snapshots" / "snap.json").exists()


# ---------------------------------------------------------------------------
# Invalid scope
# ---------------------------------------------------------------------------

def test_invalid_scope_raises_value_error(tmp_path):
    root = _make_memory_root(tmp_path)

    with pytest.raises(ValueError, match="unknown reset scope"):
        reset_memory(root, scope="banana")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "memory" / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "memory" / "agents")


def test_memory_reset_cli_active(tmp_path, store, agent_store, capsys):
    memory_root = store.root.parent
    (memory_root / "active").mkdir(parents=True, exist_ok=True)
    (memory_root / "active" / "mr1_state.json").write_text(
        json.dumps({"decisions": ["stale"], "tasks": {}}), encoding="utf-8"
    )

    rc = workflow_cli.main(
        ["memory", "reset", "active"],
        store=store,
        scoped_agent_store=agent_store,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "memory reset" in out
    assert "active" in out


def test_memory_reset_cli_json_output(tmp_path, store, agent_store):
    memory_root = store.root.parent
    (memory_root / "active").mkdir(parents=True, exist_ok=True)
    (memory_root / "active" / "mr1_state.json").write_text(
        json.dumps({"decisions": [], "tasks": {}}), encoding="utf-8"
    )

    rc = workflow_cli.main(
        ["memory", "reset", "active", "--json"],
        store=store,
        scoped_agent_store=agent_store,
    )

    assert rc == 0


def test_memory_reset_cli_all_empty(tmp_path, store, agent_store, capsys):
    rc = workflow_cli.main(
        ["memory", "reset", "all"],
        store=store,
        scoped_agent_store=agent_store,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "all" in out
