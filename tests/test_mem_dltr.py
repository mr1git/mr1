"""Tests for mr1.mini.mem_dltr"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from mr1.core.logger import Logger
from mr1.mini.mem_dltr import distill, DltrResult, _chunk_text


@pytest.fixture
def tmp_env(tmp_path):
    """Set up a temporary memory directory structure."""
    active_dir = tmp_path / "memory" / "active"
    active_dir.mkdir(parents=True)
    dumps_dir = tmp_path / "memory" / "dumps"
    dumps_dir.mkdir(parents=True)
    return tmp_path, active_dir, dumps_dir


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = _chunk_text("Hello world", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_splits_at_paragraphs(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = _chunk_text(text, chunk_size=20)
        assert len(chunks) >= 2

    def test_empty_text(self):
        assert _chunk_text("") == []

    def test_respects_chunk_size(self):
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = _chunk_text(text, chunk_size=120)
        for chunk in chunks:
            assert len(chunk) <= 120


class TestDistill:
    def test_no_state_file(self, tmp_env):
        tmp_path, active_dir, _ = tmp_env
        logger = Logger(tasks_dir=str(tmp_path / "tasks"))

        with patch("mr1.mini.mem_dltr._ACTIVE_DIR", active_dir):
            result = distill(logger=logger)

        assert result.forgotten == 0
        assert result.dumped == 0

    def test_nothing_to_distill(self, tmp_env):
        tmp_path, active_dir, _ = tmp_env
        logger = Logger(tasks_dir=str(tmp_path / "tasks"))

        state = {
            "session_id": "test123",
            "started_at": "2026-01-01T00:00:00",
            "tasks": {},
            "decisions": [{"timestamp": "2026-01-01", "action": "test", "input_summary": "x"}],
        }
        (active_dir / "mr1_state.json").write_text(json.dumps(state))

        with patch("mr1.mini.mem_dltr._ACTIVE_DIR", active_dir):
            result = distill(logger=logger)

        assert result.forgotten == 0

    def test_distills_old_decisions(self, tmp_env):
        tmp_path, active_dir, dumps_dir = tmp_env
        logger = Logger(tasks_dir=str(tmp_path / "tasks"))

        # Create state with more decisions than the retention window.
        decisions = [
            {"timestamp": f"2026-01-{i+1:02d}", "action": f"act_{i}", "input_summary": f"input {i}"}
            for i in range(30)
        ]
        state = {
            "session_id": "test123",
            "started_at": "2026-01-01T00:00:00",
            "tasks": {},
            "decisions": decisions,
        }
        (active_dir / "mr1_state.json").write_text(json.dumps(state))

        with patch("mr1.mini.mem_dltr._ACTIVE_DIR", active_dir), \
             patch("mr1.mini.mem_dltr._DUMPS_DIR", dumps_dir), \
             patch("mr1.mini.mem_dltr.mem_rtvr.ingest_chunks", return_value=2):
            result = distill(logger=logger)

        assert result.forgotten > 0
        assert result.dump_file is not None
        # Check dump file was created.
        dump_files = list(dumps_dir.glob("*.md"))
        assert len(dump_files) == 1

    def test_distills_completed_tasks(self, tmp_env):
        tmp_path, active_dir, dumps_dir = tmp_env
        logger = Logger(tasks_dir=str(tmp_path / "tasks"))

        state = {
            "session_id": "test123",
            "started_at": "2026-01-01T00:00:00",
            "tasks": {
                "task-done": {
                    "agent_type": "kazi",
                    "status": "completed",
                    "description": "finished task",
                    "started_at": "2026-01-01T00:00:00",
                    "finished_at": "2026-01-01T00:01:00",
                },
                "task-running": {
                    "agent_type": "kami",
                    "status": "running",
                    "description": "still running",
                    "started_at": "2026-01-01T00:00:00",
                },
            },
            "decisions": [],
        }
        (active_dir / "mr1_state.json").write_text(json.dumps(state))

        with patch("mr1.mini.mem_dltr._ACTIVE_DIR", active_dir), \
             patch("mr1.mini.mem_dltr._DUMPS_DIR", dumps_dir), \
             patch("mr1.mini.mem_dltr.mem_rtvr.ingest_chunks", return_value=1):
            result = distill(logger=logger)

        assert result.dumped == 1
        # Verify running task was preserved.
        updated_state = json.loads((active_dir / "mr1_state.json").read_text())
        assert "task-running" in updated_state["tasks"]
        assert "task-done" not in updated_state["tasks"]
