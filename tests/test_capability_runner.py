"""Tests for the direct capability invocation layer (Phase 14)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from mr1.capability_runner import CapabilityResult, CapabilityRunner
from mr1.scoped_agents import PersistentAgentStore


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def root_agent_id(agent_store):
    return agent_store.ensure_root_agent().agent_id


@pytest.fixture
def runner(agent_store):
    return CapabilityRunner(scoped_agent_store=agent_store)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_succeeds_and_returns_content(self, tmp_path, runner, root_agent_id):
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")

        result = runner.run_capability("read_file", {"path": str(f)}, root_agent_id)

        assert result.status == "succeeded"
        assert result.error is None
        assert result.output["text"] == "hello world"
        assert result.output["truncated"] is False
        assert result.capability == "read_file"
        assert result.duration_ms >= 0

    def test_missing_file_returns_failed(self, tmp_path, runner, root_agent_id):
        result = runner.run_capability(
            "read_file", {"path": str(tmp_path / "missing.txt")}, root_agent_id
        )

        assert result.status == "failed"
        assert result.error is not None
        assert "does not exist" in result.error

    def test_truncation_flag_set_when_exceeds_max_bytes(self, tmp_path, runner, root_agent_id):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100, encoding="utf-8")

        result = runner.run_capability(
            "read_file", {"path": str(f), "max_bytes": 10}, root_agent_id
        )

        assert result.status == "succeeded"
        assert result.output["truncated"] is True
        assert len(result.output["text"]) == 10


# ---------------------------------------------------------------------------
# file_exists
# ---------------------------------------------------------------------------


class TestFileExists:
    def test_existing_file_returns_succeeded_and_exists_true(
        self, tmp_path, runner, root_agent_id
    ):
        f = tmp_path / "present.txt"
        f.write_text("", encoding="utf-8")

        result = runner.run_capability("file_exists", {"path": str(f)}, root_agent_id)

        assert result.status == "succeeded"
        assert result.output["exists"] is True
        assert result.error is None

    def test_missing_file_returns_succeeded_and_exists_false(
        self, tmp_path, runner, root_agent_id
    ):
        result = runner.run_capability(
            "file_exists", {"path": str(tmp_path / "absent.txt")}, root_agent_id
        )

        assert result.status == "succeeded"
        assert result.output["exists"] is False
        assert result.error is None


# ---------------------------------------------------------------------------
# time_reached
# ---------------------------------------------------------------------------


class TestTimeReached:
    def test_past_time_returns_reached_true(self, runner, root_agent_id):
        result = runner.run_capability(
            "time_reached", {"at": "2000-01-01T00:00:00+00:00"}, root_agent_id
        )

        assert result.status == "succeeded"
        assert result.output["reached"] is True

    def test_future_time_returns_reached_false(self, runner, root_agent_id):
        result = runner.run_capability(
            "time_reached", {"at": "2099-01-01T00:00:00+00:00"}, root_agent_id
        )

        assert result.status == "succeeded"
        assert result.output["reached"] is False
        assert "at" in result.output
        assert "now" in result.output


# ---------------------------------------------------------------------------
# condition_script
# ---------------------------------------------------------------------------


class TestConditionScript:
    def _write_script(self, tmp_path: Path, exit_code: int) -> Path:
        script = tmp_path / "check.py"
        script.write_text(f"import sys; sys.exit({exit_code})", encoding="utf-8")
        return script

    def test_exit_0_returns_succeeded_satisfied(self, tmp_path, runner, root_agent_id):
        script = self._write_script(tmp_path, 0)

        result = runner.run_capability(
            "condition_script", {"path": str(script)}, root_agent_id
        )

        assert result.status == "succeeded"
        assert result.output["satisfied"] is True

    def test_exit_1_returns_succeeded_not_satisfied(self, tmp_path, runner, root_agent_id):
        script = self._write_script(tmp_path, 1)

        result = runner.run_capability(
            "condition_script", {"path": str(script)}, root_agent_id
        )

        assert result.status == "succeeded"
        assert result.output["satisfied"] is False
        assert result.error is None

    def test_exit_2_returns_failed(self, tmp_path, runner, root_agent_id):
        script = self._write_script(tmp_path, 2)

        result = runner.run_capability(
            "condition_script", {"path": str(script)}, root_agent_id
        )

        assert result.status == "failed"
        assert result.error is not None

    def test_timeout_returns_failed(self, tmp_path, runner, root_agent_id):
        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(60)", encoding="utf-8")

        result = runner.run_capability(
            "condition_script", {"path": str(script), "timeout_s": 1}, root_agent_id
        )

        assert result.status == "failed"
        assert result.error is not None
        assert "timed out" in result.error


# ---------------------------------------------------------------------------
# Disallowed direct capabilities
# ---------------------------------------------------------------------------


class TestDisallowedDirectCapabilities:
    @pytest.mark.parametrize("name", ["write_file", "shell_command", "manual_event"])
    def test_raises_not_direct_callable(self, name, runner, root_agent_id):
        with pytest.raises(ValueError, match=f"capability not callable in direct mode: {name}"):
            runner.run_capability(name, {}, root_agent_id)

    def test_unknown_capability_raises(self, runner, root_agent_id):
        with pytest.raises(ValueError, match="capability not found: no_such_cap"):
            runner.run_capability("no_such_cap", {}, root_agent_id)


# ---------------------------------------------------------------------------
# Access denied
# ---------------------------------------------------------------------------


class TestAccessDenied:
    def test_caller_type_not_in_callable_by_raises(self, runner, root_agent_id):
        with patch.object(runner, "_resolve_caller_type", return_value="outsider"):
            with pytest.raises(ValueError, match="access denied"):
                runner.run_capability("read_file", {"path": "/tmp/x"}, root_agent_id)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_log_written_to_capability_calls_jsonl(self, tmp_path, agent_store, root_agent_id):
        f = tmp_path / "note.txt"
        f.write_text("hi", encoding="utf-8")
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        runner.run_capability("read_file", {"path": str(f)}, root_agent_id)

        log_path = agent_store.capability_call_log_path(root_agent_id)
        assert log_path.exists()
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["capability"] == "read_file"
        assert record["agent_id"] == root_agent_id
        assert record["result_status"] == "succeeded"
        assert "timestamp" in record
        assert "duration_ms" in record

    def test_failed_call_still_writes_log(self, tmp_path, agent_store, root_agent_id):
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        runner.run_capability(
            "read_file", {"path": str(tmp_path / "missing.txt")}, root_agent_id
        )

        log_path = agent_store.capability_call_log_path(root_agent_id)
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["result_status"] == "failed"
        assert record["error"] is not None

    def test_multiple_calls_append_lines(self, tmp_path, agent_store, root_agent_id):
        f = tmp_path / "x.txt"
        f.write_text("x", encoding="utf-8")
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        runner.run_capability("read_file", {"path": str(f)}, root_agent_id)
        runner.run_capability("file_exists", {"path": str(f)}, root_agent_id)

        lines = agent_store.capability_call_log_path(root_agent_id).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["capability"] == "read_file"
        assert json.loads(lines[1])["capability"] == "file_exists"


# ---------------------------------------------------------------------------
# Caller type resolution
# ---------------------------------------------------------------------------


class TestCallerTypeResolution:
    def test_root_agent_resolves_to_mr1(self, agent_store, runner, root_agent_id):
        assert runner._resolve_caller_type(root_agent_id) == "mr1"

    def test_child_agent_resolves_to_mrn(self, agent_store, runner, root_agent_id):
        child = agent_store.create_child_agent(root_agent_id, "child")
        assert runner._resolve_caller_type(child.agent_id) == "mrn"
