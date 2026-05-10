"""Tests for the Phase 6 agent registry, health checks, CLI, and built-ins."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mr1 import workflow_cli
from mr1.agents import default_agent_registry, run_agent_health
from mr1.kazi_runner import MockRunner
from mr1.mrn_loop import MRnStepResult
from mr1.mrn_run import MRnRunResult
from mr1.mr1 import MR1, StateManager
from mr1.scoped_agents import build_assignment_packet, render_assignment_mission
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
    def test_lists_runtime_profiles(self):
        registry = default_agent_registry()

        assert registry.list_agents() == ["kazi", "memory_curator", "workflow_compiler"]

    def test_describe_agent_shape(self):
        description = default_agent_registry().describe_agent("kazi")

        assert description["name"] == "kazi"
        assert description["type"] == "agent"
        assert description["runtime"]["binary"] == "claude"
        assert description["runtime"]["supports_json_output"] is True
        assert description["workflow_task_allowed"] is True
        assert description["config_schema"]["timeout_s"]["type"] == "int"
        assert "result.text" in description["outputs"]
        assert len(description["examples"]) == 1

    def test_describe_workflow_compiler_shape(self):
        description = default_agent_registry().describe_agent("workflow_compiler")

        assert description["name"] == "workflow_compiler"
        assert description["workflow_task_allowed"] is False
        assert description["runtime"]["supports_json_output"] is True
        assert description["config_schema"]["allowed_tools"]["default"] == []

    def test_describe_memory_curator_shape(self):
        description = default_agent_registry().describe_agent("memory_curator")

        assert description["name"] == "memory_curator"
        assert description["workflow_task_allowed"] is False
        assert description["runtime"]["supports_json_output"] is True
        assert description["config_schema"]["allowed_tools"]["default"] == []

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
        assert "MR1" in out
        assert "mr1" in out

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

        assert "MR1" in output
        assert "mr1" in output

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

    def test_agent_assign_builtin(self, tmp_path):
        mr1 = _build_mr1(tmp_path)
        child = mr1._scoped_agents.create_child_agent(mr1._root_agent_id, "research")
        mission_path = tmp_path / "mission.txt"
        mission_path.write_text("Investigate the repo", encoding="utf-8")

        output = mr1._handle_builtin(f"/agent assign {child.agent_id} {mission_path}")

        assert output == child.agent_id
        reloaded = mr1._scoped_agents.require_agent(child.agent_id)
        assert reloaded.mission == "Investigate the repo"

    def test_runtime_agent_json_builtin_includes_assignment_packet(self, tmp_path):
        mr1 = _build_mr1(tmp_path)
        root = mr1._scoped_agents.ensure_root_agent()
        child = mr1._scoped_agents.create_child_agent(root.agent_id, "MR2", security_clearance=0.9)
        assignment_packet = build_assignment_packet(
            root,
            child.title,
            "Own runtime stabilization for this repo.",
            {
                "assigned_clearance": child.security_clearance,
                "agents": [root.agent_id, child.agent_id],
                "messages": ["msg-1"],
                "workflows": ["wf-1"],
            },
        )
        mr1._scoped_agents.assign_mission(
            root.agent_id,
            child.agent_id,
            render_assignment_mission(assignment_packet) or "",
            assignment_packet=assignment_packet,
        )

        output = mr1._handle_builtin(f"/agent {child.agent_id} --json")
        payload = json.loads(output)

        assert payload["agent_id"] == child.agent_id
        assert payload["assignment_packet"]["full_parent_request"] == "Own runtime stabilization for this repo."
        assert payload["assignment_packet"]["relevant_context"]["agents"] == [root.agent_id, child.agent_id]

    @patch("mr1.mr1.MRnStepRunner.step")
    def test_agent_step_builtin(self, mock_step, tmp_path):
        mr1 = _build_mr1(tmp_path)
        child = mr1._scoped_agents.create_child_agent(mr1._root_agent_id, "research")
        mr1._scoped_agents.assign_mission(mr1._root_agent_id, child.agent_id, "Investigate")
        mock_step.return_value = MRnStepResult(
            agent_id=child.agent_id,
            iteration=1,
            action="idle",
            status_before="idle",
            status_after="idle",
            reason="nothing to do",
            message="agent remains idle",
        )

        output = mr1._handle_builtin(f"/agent step {child.agent_id}")

        assert "action=idle" in output
        assert child.agent_id in output

    @patch("mr1.mr1.MRnRunRunner.run")
    def test_agent_run_builtin(self, mock_run, tmp_path):
        mr1 = _build_mr1(tmp_path)
        child = mr1._scoped_agents.create_child_agent(mr1._root_agent_id, "research")
        mr1._scoped_agents.assign_mission(mr1._root_agent_id, child.agent_id, "Investigate")
        mock_run.return_value = MRnRunResult(
            run_id="run-1",
            agent_id=child.agent_id,
            caller_agent_id=mr1._root_agent_id,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:02+00:00",
            policy={"max_steps": 5},
            steps=[{"iteration": 1}],
            workflows_created=0,
            messages_created=0,
            stopped_reason="max_steps",
            status="completed",
            final_run_status="idle",
        )

        output = mr1._handle_builtin(f"/agent run {child.agent_id} --steps 5")

        assert "run_id=run-1" in output
        assert child.agent_id in output
