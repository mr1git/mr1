"""Tests for mr1.mrn — parameterized manager agent."""

import json
from unittest.mock import patch, MagicMock

import pytest
from mr1.core.logger import Logger
from mr1.core.spawner import Spawner
from mr1.core.dispatcher import Dispatcher
from mr1.mrn import (
    run,
    MRnResult,
    _extract_output,
    _detect_context_exceeded,
    _get_model_for_level,
    _parse_response,
    _build_system_prompt,
)


@pytest.fixture
def tmp_logger(tmp_path):
    return Logger(tasks_dir=str(tmp_path / "tasks"))


@pytest.fixture
def spawner(tmp_logger):
    return Spawner(dispatcher=Dispatcher(), logger=tmp_logger)


class TestMRnResult:
    def test_ok_property(self):
        r = MRnResult("t1", 2, "completed", "out", None, 1.0, 100, [])
        assert r.ok is True
        r2 = MRnResult("t1", 2, "failed", "", "err", 1.0, 100, [])
        assert r2.ok is False

    def test_to_dict(self):
        r = MRnResult("t1", 3, "completed", "out", None, 1.234, 100, [{"sub": 1}])
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["level"] == 3
        assert d["duration_s"] == 1.23
        assert len(d["sub_tasks"]) == 1


class TestExtractOutput:
    def test_json_envelope(self):
        raw = json.dumps({"result": "mrn result", "is_error": False})
        assert _extract_output(raw) == "mrn result"

    def test_json_error(self):
        raw = json.dumps({"result": "oops", "is_error": True})
        assert "[MRn ERROR]" in _extract_output(raw)

    def test_plain_text_fallback(self):
        assert _extract_output("just text") == "just text"

    def test_empty(self):
        assert _extract_output("") == ""

    def test_custom_label(self):
        raw = json.dumps({"result": "fail", "is_error": True})
        assert "[MR2 ERROR]" in _extract_output(raw, label="MR2")


class TestModelForLevel:
    def test_known_levels(self):
        config = {
            "level_models": {1: "model-a", 2: "model-b", 3: "model-c"},
            "default_model": "model-default",
        }
        assert _get_model_for_level(config, 1) == "model-a"
        assert _get_model_for_level(config, 2) == "model-b"
        assert _get_model_for_level(config, 3) == "model-c"

    def test_default_for_unknown_level(self):
        config = {
            "level_models": {2: "model-b"},
            "default_model": "model-default",
        }
        assert _get_model_for_level(config, 5) == "model-default"

    def test_fallback_when_no_default(self):
        config = {"level_models": {}}
        assert _get_model_for_level(config, 99) == "haiku"


class TestParseResponse:
    def test_no_delegation(self):
        text, directive = _parse_response("Just a normal answer.")
        assert text == "Just a normal answer."
        assert directive is None

    def test_kazi_delegation(self):
        raw = 'I will do it. [DELEGATE]{"agent": "kazi", "task": "read file", "context": "none"}[/DELEGATE]'
        text, directive = _parse_response(raw)
        assert text == "I will do it."
        assert directive["agent"] == "kazi"

    def test_mrn_delegation(self):
        raw = '[DELEGATE]{"agent": "mr3", "task": "complex job", "context": "ctx"}[/DELEGATE]'
        text, directive = _parse_response(raw)
        assert directive["agent"] == "mr3"

    def test_invalid_agent_rejected(self):
        raw = '[DELEGATE]{"agent": "rogue", "task": "hack"}[/DELEGATE]'
        _, directive = _parse_response(raw)
        assert directive is None


class TestBuildSystemPrompt:
    def test_can_spawn_next(self):
        prompt = _build_system_prompt(level=2, height_limit=4)
        assert "MR2" in prompt
        assert "MR3" in prompt
        assert "Kazi" in prompt

    def test_at_height_limit(self):
        prompt = _build_system_prompt(level=4, height_limit=4)
        assert "CANNOT spawn manager agents" in prompt
        assert "Kazi" in prompt


class TestRun:
    def test_invalid_context(self, spawner, tmp_logger):
        result = run({"task_id": "t1"}, level=2, spawner=spawner, logger=tmp_logger)
        assert result.status == "invalid"

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_successful_run(self, mock_popen, spawner, tmp_logger):
        output_json = json.dumps({"result": "complex task done", "is_error": False})
        mock_proc = MagicMock()
        mock_proc.pid = 300
        mock_proc.communicate.return_value = (output_json.encode(), b"")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = run(
            {"task_id": "t1", "instructions": "do complex thing"},
            level=2,
            spawner=spawner,
            logger=tmp_logger,
        )
        assert result.ok is True
        assert result.output == "complex task done"
        assert result.level == 2

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_context_exceeded(self, mock_popen, spawner, tmp_logger):
        mock_proc = MagicMock()
        mock_proc.pid = 301
        mock_proc.communicate.return_value = (b"", b"Error: context window limit")
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        result = run(
            {"task_id": "t1", "instructions": "huge task"},
            level=2,
            spawner=spawner,
            logger=tmp_logger,
        )
        assert result.status == "context_exceeded"

    def test_denied_by_dispatcher(self, spawner, tmp_logger):
        # MR2 can't use Agent tool — should be rejected.
        result = run(
            {"task_id": "t1", "instructions": "test", "allowed_tools": ["Agent"]},
            level=2,
            spawner=spawner,
            logger=tmp_logger,
        )
        assert result.status == "denied"

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_level_determines_model(self, mock_popen, spawner, tmp_logger):
        """Verify that the model used varies by level."""
        output_json = json.dumps({"result": "done", "is_error": False})
        mock_proc = MagicMock()
        mock_proc.pid = 400
        mock_proc.communicate.return_value = (output_json.encode(), b"")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        # Level 2 should use Haiku
        run(
            {"task_id": "t1", "instructions": "test"},
            level=2,
            spawner=spawner,
            logger=tmp_logger,
        )
        cmd = mock_popen.call_args[0][0]
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "haiku"

        # Level 3 should use Haiku too
        run(
            {"task_id": "t2", "instructions": "test"},
            level=3,
            spawner=spawner,
            logger=tmp_logger,
        )
        cmd = mock_popen.call_args[0][0]
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "haiku"
