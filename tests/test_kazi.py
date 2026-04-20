"""Tests for mr1.kazi"""

import json
from unittest.mock import patch, MagicMock

import pytest
from mr1.core.logger import Logger
from mr1.core.spawner import Spawner
from mr1.core.dispatcher import Dispatcher
from mr1.kazi import run, KaziResult, _extract_output, _detect_context_exceeded


@pytest.fixture
def tmp_logger(tmp_path):
    return Logger(tasks_dir=str(tmp_path / "tasks"))


@pytest.fixture
def spawner(tmp_logger):
    return Spawner(dispatcher=Dispatcher(), logger=tmp_logger)


class TestKaziResult:
    def test_ok_property(self):
        r = KaziResult("t1", "completed", "out", None, 1.0, 100)
        assert r.ok is True
        r2 = KaziResult("t1", "failed", "", "err", 1.0, 100)
        assert r2.ok is False

    def test_to_dict(self):
        r = KaziResult("t1", "completed", "out", None, 1.234, 100)
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["duration_s"] == 1.23


class TestExtractOutput:
    def test_json_envelope(self):
        raw = json.dumps({"result": "hello world", "is_error": False})
        assert _extract_output(raw) == "hello world"

    def test_json_error(self):
        raw = json.dumps({"result": "oops", "is_error": True})
        assert "[KAZI ERROR]" in _extract_output(raw)

    def test_plain_text_fallback(self):
        assert _extract_output("just text") == "just text"

    def test_empty(self):
        assert _extract_output("") == ""


class TestDetectContextExceeded:
    def test_detects_patterns(self):
        assert _detect_context_exceeded("Error: context window limit reached") is True
        assert _detect_context_exceeded("exceeded max_tokens") is True

    def test_no_false_positive(self):
        assert _detect_context_exceeded("normal error message") is False


class TestRun:
    def test_invalid_context_missing_instructions(self, spawner, tmp_logger):
        result = run({"task_id": "t1"}, spawner=spawner, logger=tmp_logger)
        assert result.status == "invalid"
        assert "instructions" in result.error

    def test_invalid_context_empty_instructions(self, spawner, tmp_logger):
        result = run({"task_id": "t1", "instructions": ""}, spawner=spawner, logger=tmp_logger)
        assert result.status == "invalid"

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_successful_run(self, mock_popen, spawner, tmp_logger):
        output_json = json.dumps({"result": "task done", "is_error": False})
        mock_proc = MagicMock()
        mock_proc.pid = 200
        mock_proc.communicate.return_value = (output_json.encode(), b"")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = run(
            {"task_id": "t1", "instructions": "do something"},
            spawner=spawner,
            logger=tmp_logger,
        )
        assert result.ok is True
        assert result.output == "task done"
        assert result.pid == 200

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_failed_run(self, mock_popen, spawner, tmp_logger):
        mock_proc = MagicMock()
        mock_proc.pid = 201
        mock_proc.communicate.return_value = (b"", b"something went wrong")
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        result = run(
            {"task_id": "t1", "instructions": "fail"},
            spawner=spawner,
            logger=tmp_logger,
        )
        assert result.status == "failed"
        assert result.error is not None

    def test_denied_by_dispatcher(self, spawner, tmp_logger):
        # Kazi can't use Agent tool — dispatcher should reject.
        result = run(
            {"task_id": "t1", "instructions": "test", "allowed_tools": ["Agent"]},
            spawner=spawner,
            logger=tmp_logger,
        )
        assert result.status == "denied"
