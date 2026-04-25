from __future__ import annotations

"""Tests for the Runner adapters in mr1.kazi_runner."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from mr1.agents import AgentRuntimeError
from mr1.kazi_runner import (
    KaziAsyncRunner,
    KaziBlockingRunner,
    MockRunner,
    RunStatus,
    _parse_claude_json_envelope,
)
from mr1.workflow_models import Task, TaskStatus
from mr1.workflow_store import WorkflowStore


@dataclass
class _FakeKaziResult:
    task_id: str
    status: str
    output: str
    error: str | None
    duration_s: float
    pid: int | None
    error_type: str | None = None
    payload: dict | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


def _task(task_id="tk-1", wf_id="wf-1", prompt="hello"):
    return Task(
        task_id=task_id,
        workflow_id=wf_id,
        label="a",
        title="Task A",
        task_kind="agent",
        agent_type="kazi",
        prompt=prompt,
        status=TaskStatus.READY,
    )


class TestParseEnvelope:
    def test_parses_result_field(self):
        parsed = _parse_claude_json_envelope(
            '{"result": "done", "is_error": false, "usage": {"output_tokens": 1}}'
        )

        assert parsed["text"] == "done"
        assert parsed["is_error"] is False
        assert parsed["usage"] == {"output_tokens": 1}

    def test_flags_error_envelope(self):
        parsed = _parse_claude_json_envelope(
            '{"result": "boom", "is_error": true}'
        )

        assert parsed["text"] == "boom"
        assert parsed["is_error"] is True

    def test_invalid_json_raises(self):
        with pytest.raises(AgentRuntimeError, match="invalid JSON output"):
            _parse_claude_json_envelope("just text")

    def test_empty_raises(self):
        with pytest.raises(AgentRuntimeError, match="empty JSON output"):
            _parse_claude_json_envelope("")


class TestMockRunner:
    def test_start_records_task(self):
        r = MockRunner()
        handle = r.start(_task())
        assert handle.task_id == "tk-1"
        assert "tk-1" in r.started_task_ids

    def test_poll_without_completion_returns_none(self):
        r = MockRunner()
        handle = r.start(_task())
        assert r.poll(handle) is None

    def test_complete_then_poll_returns_result(self):
        r = MockRunner()
        handle = r.start(_task())
        r.complete("tk-1", RunStatus.SUCCEEDED, summary="ok")
        result = r.poll(handle)
        assert result is not None
        assert result.status is RunStatus.SUCCEEDED
        assert result.summary == "ok"
        # Second poll returns None again (already consumed).
        assert r.poll(handle) is None


class TestKaziBlockingRunner:
    def test_succeeded(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        fake_kazi_run = MagicMock(return_value=_FakeKaziResult(
            task_id="tk-1", status="completed", output="hello",
            error=None, duration_s=0.1, pid=42,
            payload={
                "status": "succeeded",
                "summary": "hello",
                "text": "hello",
                "data": {"raw": {"result": "hello", "is_error": False}, "is_error": False, "metadata": {}},
                "metrics": {"usage": {"output_tokens": 2}},
                "pid": 42,
            },
        ))
        runner = KaziBlockingRunner(store, kazi_run=fake_kazi_run)
        handle = runner.start(_task())
        result = runner.poll(handle)
        assert result is not None
        assert result.status is RunStatus.SUCCEEDED
        assert result.summary == "hello"
        assert result.stdout_path.exists()
        assert result.result_payload["text"] == "hello"
        assert result.result_payload["metrics"]["usage"] == {"output_tokens": 2}
        fake_kazi_run.assert_called_once()

    def test_failed_maps_to_failed_status(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        fake_kazi_run = MagicMock(return_value=_FakeKaziResult(
            task_id="tk-1", status="failed", output="",
            error="exit 1", duration_s=0.1, pid=42,
            error_type="cli_error",
        ))
        runner = KaziBlockingRunner(store, kazi_run=fake_kazi_run)
        handle = runner.start(_task())
        result = runner.poll(handle)
        assert result.status is RunStatus.FAILED
        assert result.error == "exit 1"
        assert result.error_type == "cli_error"

    def test_timeout_maps_to_timed_out(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        fake_kazi_run = MagicMock(return_value=_FakeKaziResult(
            task_id="tk-1", status="timeout", output="",
            error="exceeded 10s", duration_s=10.0, pid=42,
            error_type="timeout",
        ))
        runner = KaziBlockingRunner(store, kazi_run=fake_kazi_run)
        handle = runner.start(_task())
        result = runner.poll(handle)
        assert result.status is RunStatus.TIMED_OUT
        assert result.error_type == "timeout"


class TestKaziAsyncRunner:
    @patch("mr1.kazi_runner.subprocess.Popen")
    def test_successful_claude_run_parses_correctly(self, mock_popen, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        proc = MagicMock()
        proc.pid = 321
        proc.poll.return_value = 0
        mock_popen.return_value = proc
        runner = KaziAsyncRunner(store)

        handle = runner.start(_task())
        handle.payload["stdout_path"].write_text(
            '{"result": "hello", "is_error": false, "usage": {"output_tokens": 4}}',
            encoding="utf-8",
        )
        result = runner.poll(handle)

        assert result is not None
        assert result.status is RunStatus.SUCCEEDED
        assert result.summary == "hello"
        assert result.result_payload["text"] == "hello"
        assert result.result_payload["data"]["raw"]["result"] == "hello"
        assert result.result_payload["metrics"]["usage"] == {"output_tokens": 4}

    @patch("mr1.kazi_runner.subprocess.Popen")
    def test_is_error_true_triggers_failure(self, mock_popen, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        proc = MagicMock()
        proc.pid = 322
        proc.poll.return_value = 0
        mock_popen.return_value = proc
        runner = KaziAsyncRunner(store)

        handle = runner.start(_task())
        handle.payload["stdout_path"].write_text(
            '{"result": "Not logged in. Run claude login.", "is_error": true}',
            encoding="utf-8",
        )
        result = runner.poll(handle)

        assert result is not None
        assert result.status is RunStatus.FAILED
        assert result.error_type == "auth_error"

    @patch("mr1.kazi_runner.subprocess.Popen", side_effect=OSError("No such file or directory: claude"))
    def test_missing_binary_triggers_deterministic_cli_failure(self, _mock_popen, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = KaziAsyncRunner(store)

        handle = runner.start(_task())
        result = runner.poll(handle)

        assert result is not None
        assert result.status is RunStatus.FAILED
        assert result.error_type == "cli_error"
        assert "No such file or directory" in (result.error or "")

    @patch("mr1.kazi_runner.subprocess.Popen")
    def test_invalid_json_triggers_parse_error(self, mock_popen, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        proc = MagicMock()
        proc.pid = 323
        proc.poll.return_value = 0
        mock_popen.return_value = proc
        runner = KaziAsyncRunner(store)

        handle = runner.start(_task())
        handle.payload["stdout_path"].write_text("not json", encoding="utf-8")
        result = runner.poll(handle)

        assert result is not None
        assert result.status is RunStatus.FAILED
        assert result.error_type == "parse_error"

    @patch("mr1.kazi_runner.subprocess.Popen")
    def test_timeout_triggers_timeout_error_type(self, mock_popen, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        proc = MagicMock()
        proc.pid = 324
        proc.poll.return_value = None
        mock_popen.return_value = proc
        runner = KaziAsyncRunner(store)

        task = _task()
        task.timeout_s = 1
        handle = runner.start(task)
        handle.started_monotonic = 0.0
        with patch("mr1.kazi_runner.time.monotonic", return_value=10.0):
            result = runner.poll(handle)

        assert result is not None
        assert result.status is RunStatus.TIMED_OUT
        assert result.error_type == "timeout"
