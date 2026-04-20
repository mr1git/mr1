"""Tests for mr1.mini.com_smrzr"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from mr1.core.logger import Logger
from mr1.mini.com_smrzr import (
    summarize,
    SmrzrResult,
    _read_comms,
    _build_summary,
    _extract_highlights,
    _flatten_json_entry,
)


@pytest.fixture
def tmp_logger(tmp_path):
    return Logger(tasks_dir=str(tmp_path / "tasks"))


@pytest.fixture
def task_with_comms(tmp_path):
    """Create a task directory with sample comms."""
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task-001"
    comms_dir = task_dir / "comms"
    comms_dir.mkdir(parents=True)

    # Plain text comm.
    (comms_dir / "001-start.txt").write_text("Agent started working on task.")

    # JSON comm.
    comm_data = {
        "agent_type": "kazi",
        "action": "file_read",
        "result": "success",
        "output": "File contents read.",
        "metadata": {"file": "README.md", "lines": 42},
    }
    (comms_dir / "002-read.json").write_text(json.dumps(comm_data))

    # Markdown comm.
    (comms_dir / "003-done.md").write_text("# Task Complete\nAll steps finished successfully.")

    return tasks_dir, task_dir, comms_dir


class TestFlattenJsonEntry:
    def test_extracts_known_fields(self):
        data = {"agent_type": "kazi", "action": "read", "result": "ok"}
        text = _flatten_json_entry(data)
        assert "agent_type: kazi" in text
        assert "action: read" in text

    def test_includes_metadata(self):
        data = {"action": "test", "metadata": {"key": "value"}}
        text = _flatten_json_entry(data)
        assert "key: value" in text

    def test_fallback_to_json_dump(self):
        data = {"random_field": 123}
        text = _flatten_json_entry(data)
        assert "123" in text


class TestReadComms:
    def test_reads_all_files(self, task_with_comms):
        _, _, comms_dir = task_with_comms
        entries = _read_comms(comms_dir)
        assert len(entries) == 3

    def test_parses_json_entries(self, task_with_comms):
        _, _, comms_dir = task_with_comms
        entries = _read_comms(comms_dir)
        json_entry = [e for e in entries if e["filename"] == "002-read.json"][0]
        assert "data" in json_entry
        assert json_entry["data"]["agent_type"] == "kazi"

    def test_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty_comms"
        empty_dir.mkdir()
        entries = _read_comms(empty_dir)
        assert entries == []

    def test_skips_hidden_files(self, tmp_path):
        comms = tmp_path / "comms"
        comms.mkdir()
        (comms / ".hidden").write_text("secret")
        (comms / "visible.txt").write_text("hello")
        entries = _read_comms(comms)
        assert len(entries) == 1


class TestBuildSummary:
    def test_produces_markdown(self, task_with_comms):
        _, _, comms_dir = task_with_comms
        entries = _read_comms(comms_dir)
        summary = _build_summary("task-001", entries)

        assert "# Task task-001" in summary
        assert "## Timeline" in summary
        assert "## Details" in summary
        assert "Communications: 3" in summary

    def test_truncates_long_entries(self):
        entries = [
            {"filename": "big.txt", "mtime": "2026-01-01T00:00:00", "text": "x" * 2000}
        ]
        summary = _build_summary("t1", entries)
        assert "...(truncated)" in summary


class TestExtractHighlights:
    def test_produces_chunks(self, task_with_comms):
        _, _, comms_dir = task_with_comms
        entries = _read_comms(comms_dir)
        highlights = _extract_highlights("task-001", entries)
        assert len(highlights) > 0
        assert all("id" in h and "text" in h and "source" in h for h in highlights)

    def test_chunk_ids_contain_task_id(self, task_with_comms):
        _, _, comms_dir = task_with_comms
        entries = _read_comms(comms_dir)
        highlights = _extract_highlights("task-001", entries)
        for h in highlights:
            assert "task-001" in h["id"]

    def test_empty_entries(self):
        highlights = _extract_highlights("t1", [])
        assert highlights == []


class TestSummarize:
    def test_no_comms_dir(self, tmp_path, tmp_logger):
        with patch("mr1.mini.com_smrzr._TASKS_DIR", tmp_path / "tasks"):
            result = summarize("nonexistent-task", logger=tmp_logger)

        assert result.comms_read == 0
        assert result.summary_file is None

    def test_full_summarize(self, task_with_comms, tmp_logger):
        tasks_dir, task_dir, _ = task_with_comms

        with patch("mr1.mini.com_smrzr._TASKS_DIR", tasks_dir), \
             patch("mr1.mini.com_smrzr.mem_rtvr.ingest_chunks", return_value=2):
            result = summarize("task-001", logger=tmp_logger)

        assert result.comms_read == 3
        assert result.summary_file is not None
        assert result.comms_clearable is True
        assert result.highlights_ingested == 2

        # Verify summary file exists.
        summary_path = task_dir / "summary.md"
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "task-001" in content
