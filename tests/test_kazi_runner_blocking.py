from __future__ import annotations

"""Tests for the Runner adapters in mr1.kazi_runner."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from mr1.kazi_runner import (
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
        assert _parse_claude_json_envelope('{"result": "done", "is_error": false}') == "done"

    def test_flags_error_envelope(self):
        assert "[KAZI ERROR]" in _parse_claude_json_envelope(
            '{"result": "boom", "is_error": true}'
        )

    def test_passthrough_non_json(self):
        assert _parse_claude_json_envelope("just text") == "just text"

    def test_empty(self):
        assert _parse_claude_json_envelope("") == ""


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
        ))
        runner = KaziBlockingRunner(store, kazi_run=fake_kazi_run)
        handle = runner.start(_task())
        result = runner.poll(handle)
        assert result is not None
        assert result.status is RunStatus.SUCCEEDED
        assert result.summary == "hello"
        assert result.stdout_path.exists()
        fake_kazi_run.assert_called_once()

    def test_failed_maps_to_failed_status(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        fake_kazi_run = MagicMock(return_value=_FakeKaziResult(
            task_id="tk-1", status="failed", output="",
            error="exit 1", duration_s=0.1, pid=42,
        ))
        runner = KaziBlockingRunner(store, kazi_run=fake_kazi_run)
        handle = runner.start(_task())
        result = runner.poll(handle)
        assert result.status is RunStatus.FAILED
        assert result.error == "exit 1"

    def test_timeout_maps_to_timed_out(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        fake_kazi_run = MagicMock(return_value=_FakeKaziResult(
            task_id="tk-1", status="timeout", output="",
            error="exceeded 10s", duration_s=10.0, pid=42,
        ))
        runner = KaziBlockingRunner(store, kazi_run=fake_kazi_run)
        handle = runner.start(_task())
        result = runner.poll(handle)
        assert result.status is RunStatus.TIMED_OUT
