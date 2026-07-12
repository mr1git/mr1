"""A3 — approvals have a wall-clock deadline, and expiry never grants anything."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mr1.capability_policy import (
    DEFAULT_APPROVAL_TTL_S,
    CapabilityApprovalDecision,
    CapabilityApprovalStore,
    CapabilityRequest,
    PolicyEngine,
    ScopeContext,
    build_approval_request,
    maybe_route_approval_request,
    metadata_for_capability,
)
from mr1.clock import VirtualClock, parse_iso
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore


def _fixture(tmp_path, *, ttl_s=DEFAULT_APPROVAL_TTL_S):
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    runtime_root = tmp_path / "runtime"
    agents = PersistentAgentStore(root=runtime_root / "agents")
    messages = MessageStore(root=runtime_root / "messages", scoped_agent_store=agents)
    store = CapabilityApprovalStore(
        runtime_root / "capability_approvals",
        clock=clock,
        default_ttl_s=ttl_s,
    )
    return clock, runtime_root, agents, messages, store


def _route_shell_approval(tmp_path, *, ttl_s=DEFAULT_APPROVAL_TTL_S):
    clock, runtime_root, agents, messages, store = _fixture(tmp_path, ttl_s=ttl_s)
    root_id = agents.root_agent_id
    metadata = metadata_for_capability("shell_command", "tool")
    request = CapabilityRequest(
        actor_id=root_id,
        actor_type="mr1",
        actor_clearance=0.99,
        invocation_mode="workflow",
        capability_name="shell_command",
        args={"argv": ["pytest"], "cwd": str(tmp_path)},
        scope=ScopeContext(allowed_roots=[tmp_path], workspace_root=tmp_path),
        workflow_id="wf-1",
        task_id="tk-1",
    )
    decision = PolicyEngine().evaluate(request, metadata)
    assert decision.status == "requires_approval"

    approval_id, _created = maybe_route_approval_request(
        build_approval_request(request, metadata, decision),
        approval_store=store,
        message_store=messages,
        scoped_agent_store=agents,
    )
    return clock, runtime_root, agents, store, approval_id


def test_a_routed_approval_carries_a_deadline(tmp_path):
    _clock, _root, _agents, store, approval_id = _route_shell_approval(tmp_path)

    approval = store.require(approval_id)
    assert approval.status == "pending"
    expires = parse_iso(approval.expires_at)
    created = parse_iso(approval.created_at)
    assert (expires - created).total_seconds() == pytest.approx(DEFAULT_APPROVAL_TTL_S)


def test_expiry_is_not_reached_before_the_ttl(tmp_path):
    clock, _root, _agents, store, approval_id = _route_shell_approval(tmp_path, ttl_s=3600)

    clock.advance(3599)
    assert store.expire_stale_requests() == []
    assert store.require(approval_id).status == "pending"


def test_the_sweep_expires_a_stale_request_and_emits_the_event(tmp_path):
    clock, runtime_root, _agents, store, approval_id = _route_shell_approval(tmp_path, ttl_s=3600)

    clock.advance(3601)
    assert store.expire_stale_requests() == [approval_id]

    approval = store.require(approval_id)
    assert approval.status == "expired"

    events = EventLog(runtime_root / "events").filter_events(event_type="approval_expired")
    assert len(events) == 1
    assert events[0].metadata["reason"] == "ttl_expired"
    assert events[0].approval_request_id == approval_id


def test_expiry_is_idempotent(tmp_path):
    clock, _root, _agents, store, _approval_id = _route_shell_approval(tmp_path, ttl_s=60)
    clock.advance(120)

    assert len(store.expire_stale_requests()) == 1
    assert store.expire_stale_requests() == []


def test_an_expired_approval_cannot_be_decided(tmp_path):
    """Fail-closed: expiry removes authority, it never grants it."""
    clock, _root, agents, store, approval_id = _route_shell_approval(tmp_path, ttl_s=60)
    clock.advance(61)
    store.expire_stale_requests()

    decision = CapabilityApprovalDecision(
        approval_request_id=approval_id,
        decision="approved",
        decided_by=agents.root_agent_id,
        reason="too late",
        timestamp=clock.monotonic(),
        approval_scope="single_use",
    )
    with pytest.raises(ValueError, match="not pending"):
        store.apply_decision(approval_id, decision=decision, scoped_agent_store=agents)

    assert store.require(approval_id).status == "expired"


def test_an_expired_approval_does_not_authorize_execution(tmp_path):
    clock, _root, agents, store, approval_id = _route_shell_approval(tmp_path, ttl_s=60)
    metadata = metadata_for_capability("shell_command", "tool")
    request = store.require(approval_id).original_request

    clock.advance(61)
    store.expire_stale_requests()

    decision = PolicyEngine().evaluate(
        request,
        metadata,
        approval_request=store.load(approval_id),
    )
    assert decision.allowed is False
    assert decision.status == "requires_approval"


def test_a_decided_approval_is_not_swept(tmp_path):
    clock, _root, agents, store, approval_id = _route_shell_approval(tmp_path, ttl_s=60)
    store.apply_decision(
        approval_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval_id,
            decision="approved",
            decided_by=agents.root_agent_id,
            reason="ok",
            timestamp=clock.monotonic(),
            approval_scope="single_use",
        ),
        scoped_agent_store=agents,
    )

    clock.advance(10_000)

    assert store.expire_stale_requests() == []
    assert store.require(approval_id).status == "approved"


def test_ttl_can_be_disabled(tmp_path):
    clock, _root, _agents, store, approval_id = _route_shell_approval(tmp_path, ttl_s=None)

    assert store.require(approval_id).expires_at is None
    clock.advance(10_000_000)
    assert store.expire_stale_requests() == []


def test_workflow_cancellation_still_expires_immediately(tmp_path):
    _clock, runtime_root, _agents, store, approval_id = _route_shell_approval(tmp_path)

    expired = store.expire_requests_for_workflow("wf-1", reason="workflow_cancelled")

    assert expired == [approval_id]
    assert store.require(approval_id).status == "expired"
    events = EventLog(runtime_root / "events").filter_events(event_type="approval_expired")
    assert events[0].metadata["reason"] == "workflow_cancelled"


def test_re_requesting_after_expiry_starts_a_fresh_deadline(tmp_path):
    clock, _root, agents, store, approval_id = _route_shell_approval(tmp_path, ttl_s=3600)
    messages = MessageStore(
        root=store._root.parent / "messages",
        scoped_agent_store=agents,
    )
    clock.advance(3601)
    store.expire_stale_requests()
    expired = store.require(approval_id)

    clock.advance(100)
    reopened_id, _ = maybe_route_approval_request(
        expired,
        approval_store=store,
        message_store=messages,
        scoped_agent_store=agents,
    )

    reopened = store.require(reopened_id)
    assert reopened.status == "pending"
    assert parse_iso(reopened.expires_at) > parse_iso(expired.expires_at)


def test_legacy_approvals_without_expires_at_load_and_never_self_expire(tmp_path):
    """Approvals written before A3 must keep working."""
    clock, _root, _agents, store, approval_id = _route_shell_approval(tmp_path)
    payload = store.require(approval_id).to_dict()
    payload.pop("expires_at")
    path = store.approval_path(approval_id)
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")

    approval = store.require(approval_id)
    assert approval.expires_at is None
    clock.advance(10_000_000)
    assert store.expire_stale_requests() == []
