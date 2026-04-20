"""Tests for mr1.core.logger"""

import json
import tempfile
from pathlib import Path

import pytest
from mr1.core.logger import Logger


@pytest.fixture
def tmp_tasks_dir(tmp_path):
    return str(tmp_path / "tasks")


@pytest.fixture
def logger(tmp_tasks_dir):
    return Logger(tasks_dir=tmp_tasks_dir)


class TestLog:
    def test_basic_log(self, logger, tmp_tasks_dir):
        entry = logger.log("task-001", "kazi", "run", "ok")
        assert entry["task_id"] == "task-001"
        assert entry["agent_type"] == "kazi"
        assert entry["action"] == "run"
        assert entry["result"] == "ok"
        assert "timestamp" in entry

    def test_log_with_metadata(self, logger):
        entry = logger.log("task-001", "kazi", "run", "ok", metadata={"pid": 123})
        assert entry["metadata"]["pid"] == 123

    def test_log_creates_file(self, logger, tmp_tasks_dir):
        logger.log("task-001", "kazi", "run", "ok")
        log_file = Path(tmp_tasks_dir) / "task-001" / "logs" / "kazi.jsonl"
        assert log_file.exists()
        content = log_file.read_text()
        parsed = json.loads(content.strip())
        assert parsed["action"] == "run"

    def test_multiple_entries_append(self, logger, tmp_tasks_dir):
        logger.log("task-001", "kazi", "start", "ok")
        logger.log("task-001", "kazi", "end", "ok")
        log_file = Path(tmp_tasks_dir) / "task-001" / "logs" / "kazi.jsonl"
        lines = [l for l in log_file.read_text().strip().split("\n") if l]
        assert len(lines) == 2


class TestConvenienceMethods:
    def test_log_spawn(self, logger):
        entry = logger.log_spawn("task-001", "kazi", 123, ["claude", "-p", "hi"])
        assert entry["action"] == "spawn"
        assert entry["metadata"]["pid"] == 123

    def test_log_kill(self, logger):
        entry = logger.log_kill("task-001", "kazi", 123, "user_cancel")
        assert entry["action"] == "kill"

    def test_log_denied(self, logger):
        entry = logger.log_denied("task-001", "kazi", "flag not allowed")
        assert entry["result"] == "denied"

    def test_log_exit(self, logger):
        entry = logger.log_exit("task-001", "kazi", 123, 0)
        assert entry["result"] == "ok"
        entry = logger.log_exit("task-001", "kazi", 123, 1)
        assert entry["result"] == "error"


class TestReadLogs:
    def test_read_logs_empty(self, logger):
        assert logger.read_logs("nonexistent") == []

    def test_read_logs_roundtrip(self, logger):
        logger.log("task-001", "kazi", "start", "ok")
        logger.log("task-001", "kazi", "end", "ok")
        entries = logger.read_logs("task-001", "kazi")
        assert len(entries) == 2
        assert entries[0]["action"] == "start"

    def test_read_all_agents(self, logger):
        logger.log("task-001", "kazi", "run", "ok")
        logger.log("task-001", "kami", "delegate", "ok")
        entries = logger.read_logs("task-001")
        assert len(entries) == 2
