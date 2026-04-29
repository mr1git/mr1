"""Tests for the direct capability invocation policy boundary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mr1.capability_runner import CapabilityRunner
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


def _load_audit(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TestReadFile:
    def test_root_read_inside_workspace_is_allowed(self, tmp_path, runner, root_agent_id):
        target = tmp_path / "hello.txt"
        target.write_text("hello world", encoding="utf-8")

        result = runner.run_capability("read_file", {"path": str(target)}, root_agent_id)

        assert result.status == "succeeded"
        assert result.error is None
        assert result.output["text"] == "hello world"
        audit = _load_audit(result.audit_record_path)
        assert audit["decision"]["status"] == "allowed"
        assert audit["decision"]["final_effect"] == "executed"

    def test_missing_file_is_execution_failure_but_still_audited(self, tmp_path, runner, root_agent_id):
        result = runner.run_capability(
            "read_file",
            {"path": str(tmp_path / "missing.txt")},
            root_agent_id,
        )

        assert result.status == "failed"
        assert "does not exist" in (result.error or "")
        audit = _load_audit(result.audit_record_path)
        assert audit["decision"]["status"] == "allowed"
        assert audit["error"] is not None

    def test_child_in_scope_read_is_allowed(self, tmp_path, agent_store, root_agent_id):
        child = agent_store.create_child_agent(root_agent_id, "child")
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        target = allowed_dir / "data.txt"
        target.write_text("ok", encoding="utf-8")
        child = agent_store.require_agent(child.agent_id)
        child.scope_roots = [str(allowed_dir)]
        agent_store.save_agent(child)
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        result = runner.run_capability("read_file", {"path": str(target)}, child.agent_id)

        assert result.status == "succeeded"
        assert result.output["text"] == "ok"

    def test_child_out_of_scope_read_requires_approval(self, tmp_path, agent_store, root_agent_id):
        child = agent_store.create_child_agent(root_agent_id, "child")
        allowed_dir = tmp_path / "allowed"
        denied_dir = tmp_path / "denied"
        allowed_dir.mkdir()
        denied_dir.mkdir()
        target = denied_dir / "secret.txt"
        target.write_text("secret", encoding="utf-8")
        child = agent_store.require_agent(child.agent_id)
        child.scope_roots = [str(allowed_dir)]
        agent_store.save_agent(child)
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        result = runner.run_capability(
            "read_file",
            {"path": str(target)},
            child.agent_id,
            step_id="step-1",
        )

        assert result.status == "requires_approval"
        assert result.output["status"] == "requires_approval"
        assert result.output["reason"] == "outside_actor_scope"
        assert result.output["approval_request_id"]
        inbox = runner._message_store.list_inbox(root_agent_id)
        assert len(inbox) == 1
        audit = _load_audit(result.audit_record_path)
        assert audit["decision"]["status"] == "requires_approval"
        assert audit["execution_result"]["approval_request_id"] == result.output["approval_request_id"]

    def test_identical_same_step_requests_dedupe_approval(self, tmp_path, agent_store, root_agent_id):
        child = agent_store.create_child_agent(root_agent_id, "child")
        target = tmp_path / "secret.txt"
        target.write_text("secret", encoding="utf-8")
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        first = runner.run_capability(
            "read_file",
            {"path": str(target)},
            child.agent_id,
            step_id="step-7",
        )
        second = runner.run_capability(
            "read_file",
            {"path": str(target)},
            child.agent_id,
            step_id="step-7",
        )

        assert first.status == "requires_approval"
        assert second.status == "requires_approval"
        assert first.output["approval_request_id"] == second.output["approval_request_id"]
        inbox = runner._message_store.list_inbox(root_agent_id)
        assert len(inbox) == 1


class TestFileExists:
    def test_existing_file_returns_succeeded_and_exists_true(
        self, tmp_path, runner, root_agent_id
    ):
        f = tmp_path / "present.txt"
        f.write_text("", encoding="utf-8")

        result = runner.run_capability("file_exists", {"path": str(f)}, root_agent_id)

        assert result.status == "succeeded"
        assert result.output["exists"] is True

    def test_missing_file_returns_succeeded_and_exists_false(
        self, tmp_path, runner, root_agent_id
    ):
        result = runner.run_capability(
            "file_exists", {"path": str(tmp_path / "absent.txt")}, root_agent_id
        )

        assert result.status == "succeeded"
        assert result.output["exists"] is False


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


class TestPolicyDenials:
    @pytest.mark.parametrize("name", ["write_file", "shell_command", "manual_event"])
    def test_policy_block_returns_structured_denial(self, name, runner, root_agent_id):
        result = runner.run_capability(name, {}, root_agent_id)

        assert result.status == "denied"
        assert result.output["status"] == "denied"
        assert "reason" in result.output

    def test_condition_script_is_denied_for_root_direct_risk(self, tmp_path, runner, root_agent_id):
        script = tmp_path / "check.py"
        script.write_text("import sys; sys.exit(0)", encoding="utf-8")

        result = runner.run_capability(
            "condition_script",
            {"path": str(script)},
            root_agent_id,
        )

        assert result.status == "denied"
        assert result.output["reason"] == "risk_exceeds_direct_threshold"

    def test_missing_required_arg_is_denied(self, runner, root_agent_id):
        result = runner.run_capability("read_file", {}, root_agent_id)

        assert result.status == "denied"
        assert result.output["reason"] == "missing_required_arg:path"

    def test_access_denied_for_unknown_actor_type(self, runner, root_agent_id):
        with patch.object(runner, "_resolve_caller_type", return_value="outsider"):
            with pytest.raises(ValueError, match="access denied"):
                runner.run_capability("read_file", {"path": "/tmp/x"}, root_agent_id)


class TestScopeNormalization:
    def test_symlink_escape_requires_approval(self, tmp_path, agent_store, root_agent_id):
        child = agent_store.create_child_agent(root_agent_id, "child")
        allowed_dir = tmp_path / "allowed"
        outside_dir = tmp_path / "outside"
        allowed_dir.mkdir()
        outside_dir.mkdir()
        target = outside_dir / "secret.txt"
        target.write_text("secret", encoding="utf-8")
        link = allowed_dir / "linked.txt"
        link.symlink_to(target)
        child = agent_store.require_agent(child.agent_id)
        child.scope_roots = [str(allowed_dir)]
        agent_store.save_agent(child)
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        result = runner.run_capability("read_file", {"path": str(link)}, child.agent_id)

        assert result.status == "requires_approval"


class TestAuditLog:
    def test_audit_record_written_for_success(self, tmp_path, agent_store, root_agent_id):
        f = tmp_path / "note.txt"
        f.write_text("hi", encoding="utf-8")
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        result = runner.run_capability("read_file", {"path": str(f)}, root_agent_id)

        audit = _load_audit(result.audit_record_path)
        assert audit["capability_name"] == "read_file"
        assert audit["decision"]["policy_id"] == "capability-policy-v1"
        assert audit["decision"]["policy_version"] == 1
        assert audit["execution_result"]["text"] == "hi"

    def test_audit_record_written_for_denied(self, agent_store, root_agent_id):
        runner = CapabilityRunner(scoped_agent_store=agent_store)

        result = runner.run_capability("write_file", {"path": "/tmp/x", "content": "x"}, root_agent_id)

        audit = _load_audit(result.audit_record_path)
        assert audit["decision"]["status"] == "denied"
        assert audit["execution_result"]["status"] == "denied"


class TestCallerTypeResolution:
    def test_root_agent_resolves_to_mr1(self, agent_store, runner, root_agent_id):
        assert runner._resolve_caller_type(root_agent_id) == "mr1"

    def test_child_agent_resolves_to_mrn(self, agent_store, runner, root_agent_id):
        child = agent_store.create_child_agent(root_agent_id, "child")
        assert runner._resolve_caller_type(child.agent_id) == "mrn"
