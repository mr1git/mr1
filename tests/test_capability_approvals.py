"""Tests for hierarchical capability approvals and scope control."""

from __future__ import annotations

import json

import pytest

from mr1.capability_policy import CapabilityApprovalDecision, CapabilityApprovalStore
from mr1.capability_runner import CapabilityRunner
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def approval_store(tmp_path):
    return CapabilityApprovalStore(tmp_path / "capability_approvals")


def test_approval_routing_uses_nearest_qualified_ancestor(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    parent = agent_store.create_child_agent(root.agent_id, "parent", security_clearance=0.2)
    child = agent_store.create_child_agent(parent.agent_id, "child", security_clearance=0.1)
    denied_dir = tmp_path / "denied"
    denied_dir.mkdir()
    target = denied_dir / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    result = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-1")

    assert result.status == "requires_approval"
    assert result.approval_request_id is not None
    approval = runner._approval_store.require(result.approval_request_id)
    assert approval.designated_approver_id == parent.agent_id
    assert len(runner._message_store.list_inbox(parent.agent_id)) == 1
    assert runner._message_store.list_inbox(root.agent_id) == []


def test_approval_routing_falls_back_to_root(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    parent = agent_store.create_child_agent(root.agent_id, "parent", security_clearance=0.05)
    child = agent_store.create_child_agent(parent.agent_id, "child", security_clearance=0.05)
    denied_dir = tmp_path / "denied"
    denied_dir.mkdir()
    target = denied_dir / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    result = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-2")

    approval = runner._approval_store.require(result.approval_request_id)
    assert approval.designated_approver_id == root.agent_id
    assert len(runner._message_store.list_inbox(root.agent_id)) == 1


def test_approval_decision_rejected_when_clearance_drops(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    parent = agent_store.create_child_agent(root.agent_id, "parent", security_clearance=0.2)
    child = agent_store.create_child_agent(parent.agent_id, "child", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)
    result = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-3")
    approval_id = result.approval_request_id

    parent_record = agent_store.require_agent(parent.agent_id)
    parent_record.security_clearance = 0.05
    agent_store.save_agent(parent_record)

    with pytest.raises(ValueError, match="insufficient security clearance"):
        runner._approval_store.apply_decision(
            approval_id,
            decision=CapabilityApprovalDecision(
                approval_request_id=approval_id,
                decision="approved",
                decided_by=parent.agent_id,
                reason="approve",
                timestamp=1.0,
                approval_scope="single_use",
            ),
            scoped_agent_store=agent_store,
        )


def test_approval_decision_accepted_when_clearance_is_sufficient(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    parent = agent_store.create_child_agent(root.agent_id, "parent", security_clearance=0.2)
    child = agent_store.create_child_agent(parent.agent_id, "child", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)
    result = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-4")
    approval_id = result.approval_request_id

    approval = runner._approval_store.apply_decision(
        approval_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval_id,
            decision="approved",
            decided_by=parent.agent_id,
            reason="approve",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )

    assert approval.status == "approved"
    assert approval.decision["approval_scope"] == "single_use"


def test_approved_request_executes_on_retry_and_consumes_single_use(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    first = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-5")
    approval_id = first.approval_request_id
    runner._approval_store.apply_decision(
        approval_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval_id,
            decision="approved",
            decided_by=root.agent_id,
            reason="one shot",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )

    second = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-5")

    assert second.status == "succeeded"
    assert second.output["text"] == "secret"
    consumed = runner._approval_store.require(approval_id)
    assert consumed.status == "superseded"
    assert consumed.used_by_audit_id
    third = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-5")
    assert third.status == "requires_approval"
    assert third.approval_request_id == approval_id


def test_approved_request_matches_retry_across_step_ids(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    first = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-a")
    approval_id = first.approval_request_id
    runner._approval_store.apply_decision(
        approval_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval_id,
            decision="approved",
            decided_by=root.agent_id,
            reason="one shot",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )

    second = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-b")

    assert second.status == "succeeded"
    assert second.output["text"] == "secret"


def test_denied_request_remains_blocked_on_retry(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    first = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-6")
    approval_id = first.approval_request_id
    runner._approval_store.apply_decision(
        approval_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval_id,
            decision="denied",
            decided_by=root.agent_id,
            reason="no",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )

    second = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-6")

    assert second.status == "requires_approval"
    assert second.approval_request_id == approval_id
    assert len(runner._message_store.list_inbox(root.agent_id)) == 1


def test_scope_grant_adds_access_and_revoke_removes_it(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    before = runner.run_capability("read_file", {"path": str(target)}, child.agent_id)
    grant = agent_store.grant_scope(root.agent_id, child.agent_id, target, reason="allow")
    after_grant = runner.run_capability("read_file", {"path": str(target)}, child.agent_id)
    revoked = agent_store.revoke_scope(root.agent_id, child.agent_id, target)
    after_revoke = runner.run_capability("read_file", {"path": str(target)}, child.agent_id)

    assert before.status == "requires_approval"
    assert grant.path == str(target.resolve())
    assert after_grant.status == "succeeded"
    assert revoked == str(target.resolve())
    assert after_revoke.status == "requires_approval"


def test_mrn_cannot_grant_scope_it_does_not_have(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child", security_clearance=1.0)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")

    with pytest.raises(AgentScopeError, match="granting agent lacks access"):
        agent_store.grant_scope(child.agent_id, child.agent_id, target, reason="self")
