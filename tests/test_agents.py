"""Tests for the Phase 6 agent registry, health checks, CLI, and built-ins."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from mr1 import workflow_cli
from mr1.agents import default_agent_registry, run_agent_health
from mr1.kazi_runner import MockRunner
from mr1.mr1 import MR1, StateManager
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


def _build_mr1(tmp_path):
    mr1 = MR1(
        workflow_store=WorkflowStore(root=tmp_path / "workflows"),
        workflow_runner=MockRunner(),
        workflow_auto_tick=False,
        workflow_compiler=lambda *_: "{}",
    )
    mr1._state = StateManager(state_path=tmp_path / "mr1_state.json")
    return mr1


class TestAgentRegistry:
    def test_lists_kazi(self):
        registry = default_agent_registry()

        assert registry.list_agents() == ["kazi"]

    def test_describe_agent_shape(self):
        description = default_agent_registry().describe_agent("kazi")

        assert description["name"] == "kazi"
        assert description["type"] == "agent"
        assert description["runtime"]["binary"] == "claude"
        assert description["runtime"]["supports_json_output"] is True
        assert description["config_schema"]["timeout_s"]["type"] == "int"
        assert "result.text" in description["outputs"]
        assert len(description["examples"]) == 1

    def test_unknown_agent_is_deterministic(self):
        with pytest.raises(ValueError, match="agent not found: missing"):
            default_agent_registry().describe_agent("missing")


class TestAgentHealth:
    @patch("mr1.agents.subprocess.run")
    @patch("mr1.agents.shutil.which")
    def test_health_success_path(self, mock_which, mock_run):
        mock_which.return_value = "/usr/local/bin/claude"
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                ["/usr/local/bin/claude", "--version"],
                0,
                stdout="claude 1.0.0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                ["/usr/local/bin/claude", "-p", "ping", "--output-format", "json"],
                0,
                stdout=json.dumps({
                    "result": "pong",
                    "is_error": False,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }),
                stderr="",
            ),
        ]

        result = run_agent_health("kazi")

        assert result["status"] == "healthy"
        assert result["checks"]["binary"] == "/usr/local/bin/claude"
        assert result["checks"]["prompt_test"] == "passed"
        assert result["checks"]["json_parse"] == "ok"
        assert result["checks"]["auth"] == "ok"
        assert result["checks"]["config"] == "ok"
        assert result["checks"]["flags"] == "ok"

    @patch("mr1.agents.shutil.which")
    def test_health_failure_missing_binary(self, mock_which):
        mock_which.return_value = None

        result = run_agent_health("kazi")

        assert result["status"] == "unhealthy"
        assert result["checks"]["binary"] == "missing"
        assert result["error"] == "binary not found: claude"

    @patch("mr1.agents.subprocess.run")
    @patch("mr1.agents.shutil.which")
    def test_health_failure_auth_error(self, mock_which, mock_run):
        mock_which.return_value = "/usr/local/bin/claude"
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                ["/usr/local/bin/claude", "--version"],
                0,
                stdout="claude 1.0.0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                ["/usr/local/bin/claude", "-p", "ping", "--output-format", "json"],
                0,
                stdout=json.dumps({"result": "Not logged in. Run claude login.", "is_error": True}),
                stderr="",
            ),
        ]

        result = run_agent_health("kazi")

        assert result["status"] == "unhealthy"
        assert result["checks"]["prompt_test"] == "passed"
        assert result["checks"]["auth"] == "failed"
        assert result["checks"]["json_parse"] == "ok"
        assert "Not logged in" in result["error"]


class TestAgentCli:
    def test_agents_command(self, store, capsys):
        rc = workflow_cli.main(["agents"], store=store)

        assert rc == 0
        out = capsys.readouterr().out
        assert "kazi" in out
        assert "claude" in out

    def test_agent_command(self, store, capsys):
        rc = workflow_cli.main(["agent", "kazi"], store=store)

        assert rc == 0
        out = capsys.readouterr().out
        assert "name:         kazi" in out
        assert '"binary": "claude"' in out

    def test_agent_json_command(self, store, capsys):
        rc = workflow_cli.main(["agent", "kazi", "--json"], store=store)

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "kazi"
        assert payload["runtime"]["binary"] == "claude"

    @patch("mr1.workflow_cli.run_agent_health")
    def test_agent_health_command(self, mock_health, store, capsys):
        mock_health.return_value = {
            "status": "healthy",
            "checks": {
                "binary": "/usr/local/bin/claude",
                "version": "claude 1.0.0",
                "prompt_test": "passed",
                "auth": "ok",
                "json_parse": "ok",
                "config": "ok",
                "flags": "ok",
            },
        }

        rc = workflow_cli.main(["agent", "kazi", "health"], store=store)

        assert rc == 0
        out = capsys.readouterr().out
        assert "agent:       kazi" in out
        assert "status:      healthy" in out
        assert "binary: /usr/local/bin/claude" in out

    def test_unknown_agent_error_is_deterministic(self, store, capsys):
        rc = workflow_cli.main(["agent", "missing"], store=store)

        assert rc == 2
        assert capsys.readouterr().err.strip() == "error: agent not found: missing"


class TestAgentBuiltins:
    def test_agents_builtin(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        output = mr1._handle_builtin("/agents")

        assert "kazi" in output
        assert "claude" in output

    def test_agent_builtin(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        output = mr1._handle_builtin("/agent kazi --json")
        payload = json.loads(output)

        assert payload["name"] == "kazi"
        assert payload["runtime"]["supports_json_output"] is True

    @patch("mr1.workflow_cli.run_agent_health")
    def test_agent_health_builtin(self, mock_health, tmp_path):
        mr1 = _build_mr1(tmp_path)
        mock_health.return_value = {
            "status": "healthy",
            "checks": {
                "binary": "/usr/local/bin/claude",
                "version": "claude 1.0.0",
                "prompt_test": "passed",
                "auth": "ok",
                "json_parse": "ok",
                "config": "ok",
                "flags": "ok",
            },
        }

        output = mr1._handle_builtin("/agent kazi health")

        assert "agent:       kazi" in output
        assert "status:      healthy" in output

    def test_unknown_agent_builtin_is_deterministic(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        assert mr1._handle_builtin("/agent missing") == "agent not found: missing"
