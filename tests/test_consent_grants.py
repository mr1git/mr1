"""A4 — objective-scoped consent grants: the standing-authority mechanism."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mr1.autonomy.consent import ConsentGrant, ConsentGrantError, ConsentGrantStore
from mr1.capability_policy import (
    CapabilityRequest,
    PolicyEngine,
    ScopeContext,
    metadata_for_capability,
)
from mr1.clock import VirtualClock
from mr1.event_log import EventLog
from mr1.scoped_agents import PersistentAgentStore


OBJECTIVE = "obj-genesis"
OTHER_OBJECTIVE = "obj-somethingelse"


def _fixture(tmp_path):
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    runtime_root = tmp_path / "runtime"
    agents = PersistentAgentStore(root=runtime_root / "agents")
    store = ConsentGrantStore(runtime_root, clock=clock, scoped_agent_store=agents)
    return clock, runtime_root, agents, store


def _shell_request(tmp_path, *, objective_id=OBJECTIVE, argv=None, cwd=None):
    return CapabilityRequest(
        actor_id="agent-1",
        actor_type="mr1",
        actor_clearance=0.99,
        invocation_mode="workflow",
        capability_name="shell_command",
        args={"argv": argv or ["pytest", "-q"], "cwd": str(cwd or tmp_path)},
        scope=ScopeContext(allowed_roots=[tmp_path], workspace_root=tmp_path),
        workflow_id="wf-1",
        task_id="tk-1",
        objective_id=objective_id,
    )


def _grant(store, agents, tmp_path, **overrides):
    payload = {
        "objective_id": OBJECTIVE,
        "capability_name": "shell_command",
        "scope_roots": [tmp_path],
        "max_risk": 1.0,
        "granted_by": agents.root_agent_id,
        "ttl_s": 7 * 86_400,
        "arg_predicate": {"argv": {"regex": r"^pytest\b"}},
        "reason": "genesis weekly cycle",
    }
    payload.update(overrides)
    return store.create(**payload)


# -- the model -------------------------------------------------------------


def test_a_grant_must_expire(tmp_path):
    _clock, _root, agents, store = _fixture(tmp_path)
    with pytest.raises(ConsentGrantError, match="positive TTL"):
        _grant(store, agents, tmp_path, ttl_s=0)


def test_a_grant_must_bind_a_scope(tmp_path):
    _clock, _root, agents, store = _fixture(tmp_path)
    with pytest.raises(ConsentGrantError, match="scope root"):
        _grant(store, agents, tmp_path, scope_roots=[])


def test_only_root_may_grant_above_the_autonomous_ceiling(tmp_path):
    _clock, _root, agents, store = _fixture(tmp_path)
    child = agents.create_child_agent(
        agents.root_agent_id,
        "MR2",
        security_clearance=0.99,
    )
    with pytest.raises(ConsentGrantError, match="only the root agent"):
        _grant(store, agents, tmp_path, granted_by=child.agent_id, max_risk=1.0)

    # Below the ceiling, a sufficiently cleared agent may grant.
    grant = _grant(store, agents, tmp_path, granted_by=child.agent_id, max_risk=0.6)
    assert grant.granted_by == child.agent_id


def test_grantor_cannot_exceed_its_own_clearance(tmp_path):
    _clock, _root, agents, store = _fixture(tmp_path)
    child = agents.create_child_agent(
        agents.root_agent_id,
        "MR3",
        security_clearance=0.3,
    )
    with pytest.raises(ConsentGrantError, match="clearance"):
        _grant(store, agents, tmp_path, granted_by=child.agent_id, max_risk=0.9)


def test_create_emits_an_event_and_persists(tmp_path):
    _clock, runtime_root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)

    reloaded = ConsentGrantStore(runtime_root).require(grant.grant_id)
    assert reloaded.grantee_id == OBJECTIVE
    assert reloaded.use_count == 0

    events = EventLog(runtime_root / "events").filter_events(event_type="consent_grant_created")
    assert len(events) == 1
    assert events[0].metadata["grant_id"] == grant.grant_id
    assert events[0].metadata["max_risk"] == 1.0


# -- matching --------------------------------------------------------------


def test_a_matching_grant_authorizes(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")

    assert grant.match_failure(_shell_request(tmp_path), metadata, now=clock.now()) is None
    assert store.match(_shell_request(tmp_path), metadata) is not None


def test_a_grant_cannot_authorize_another_objective(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")

    request = _shell_request(tmp_path, objective_id=OTHER_OBJECTIVE)
    assert grant.match_failure(request, metadata, now=clock.now()) == "grant_belongs_to_another_objective"
    assert store.match(request, metadata) is None


def test_a_request_with_no_objective_matches_nothing(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")

    request = _shell_request(tmp_path, objective_id=None)
    assert grant.match_failure(request, metadata, now=clock.now()) == "request_has_no_objective"
    assert store.match(request, metadata) is None


def test_a_grant_cannot_widen_its_own_scope(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    inside = tmp_path / "project"
    inside.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    grant = _grant(store, agents, tmp_path, scope_roots=[inside])
    metadata = metadata_for_capability("shell_command", "tool")

    assert grant.match_failure(_shell_request(tmp_path, cwd=inside), metadata, now=clock.now()) is None
    assert (
        grant.match_failure(_shell_request(tmp_path, cwd=outside), metadata, now=clock.now())
        == "path_outside_grant_scope:cwd"
    )


def test_a_grant_cannot_authorize_above_its_max_risk(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path, max_risk=0.5)
    metadata = metadata_for_capability("shell_command", "tool")  # risk 1.0

    assert (
        grant.match_failure(_shell_request(tmp_path), metadata, now=clock.now())
        == "risk_exceeds_grant_max_risk"
    )


def test_a_grant_is_capability_specific(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path, capability_name="write_file")
    metadata = metadata_for_capability("shell_command", "tool")

    assert grant.match_failure(_shell_request(tmp_path), metadata, now=clock.now()) == "capability_mismatch"


def test_the_arg_predicate_is_enforced(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")

    allowed = _shell_request(tmp_path, argv=["pytest", "tests/"])
    denied = _shell_request(tmp_path, argv=["rm", "-rf", "/"])

    assert grant.match_failure(allowed, metadata, now=clock.now()) is None
    assert grant.match_failure(denied, metadata, now=clock.now()) == "predicate_rejected:argv"


@pytest.mark.parametrize(
    "spec,argv,expected",
    [
        ({"argv": {"prefix": "pytest"}}, ["pytest", "-q"], True),
        ({"argv": {"prefix": "pytest"}}, ["ruff", "check"], False),
        ({"argv": {"equals": "git status"}}, ["git", "status"], True),
        ({"argv": {"equals": "git status"}}, ["git", "push"], False),
        ({"argv": {"one_of": ["pytest", "ruff check"]}}, ["ruff", "check"], True),
        ({"argv": {"one_of": ["pytest"]}}, ["ruff"], False),
        ({"argv": {"regex": r"^(pytest|ruff)\b"}}, ["ruff", "check"], True),
        ({"argv": {"unknown_op": "x"}}, ["pytest"], False),
        ({"argv": {}}, ["pytest"], False),
        ({"argv": {"regex": "["}}, ["pytest"], False),
    ],
)
def test_predicate_operators(tmp_path, spec, argv, expected):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path, arg_predicate=spec)
    metadata = metadata_for_capability("shell_command", "tool")

    matched = grant.match_failure(_shell_request(tmp_path, argv=argv), metadata, now=clock.now()) is None
    assert matched is expected


def test_a_predicate_on_a_missing_arg_fails_closed(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path, arg_predicate={"nope": {"equals": "x"}})
    metadata = metadata_for_capability("shell_command", "tool")

    assert (
        grant.match_failure(_shell_request(tmp_path), metadata, now=clock.now())
        == "predicate_arg_missing:nope"
    )


# -- lifecycle -------------------------------------------------------------


def test_an_expired_grant_authorizes_nothing(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path, ttl_s=3600)
    metadata = metadata_for_capability("shell_command", "tool")

    clock.advance(3599)
    assert store.match(_shell_request(tmp_path), metadata) is not None

    clock.advance(2)
    assert store.match(_shell_request(tmp_path), metadata) is None
    assert grant.match_failure(_shell_request(tmp_path), metadata, now=clock.now()) == "grant_expired"
    assert store.list_active() == []


def test_a_revoked_grant_authorizes_nothing(tmp_path):
    _clock, runtime_root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")

    store.revoke(grant.grant_id, revoked_by=agents.root_agent_id, reason="changed my mind")

    assert store.match(_shell_request(tmp_path), metadata) is None
    assert store.require(grant.grant_id).status(store._clock.now()) == "revoked"
    events = EventLog(runtime_root / "events").filter_events(event_type="consent_grant_revoked")
    assert events[0].metadata["reason"] == "changed my mind"


def test_revoke_all_removes_every_grant(tmp_path):
    _clock, _root, agents, store = _fixture(tmp_path)
    _grant(store, agents, tmp_path)
    _grant(store, agents, tmp_path, objective_id=OTHER_OBJECTIVE)

    revoked = store.revoke_all(revoked_by=agents.root_agent_id, reason="halt")

    assert len(revoked) == 2
    assert store.list_active() == []


def test_revoke_all_can_target_one_objective(tmp_path):
    _clock, _root, agents, store = _fixture(tmp_path)
    _grant(store, agents, tmp_path)
    keeper = _grant(store, agents, tmp_path, objective_id=OTHER_OBJECTIVE)

    store.revoke_all(revoked_by=agents.root_agent_id, objective_id=OBJECTIVE)

    active = store.list_active()
    assert [item.grant_id for item in active] == [keeper.grant_id]


def test_use_count_tracks_unattended_executions(tmp_path):
    _clock, runtime_root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)

    store.record_use(grant.grant_id, workflow_id="wf-1", task_id="tk-1", audit_id="aud-1")
    store.record_use(grant.grant_id, workflow_id="wf-2", task_id="tk-2", audit_id="aud-2")

    assert store.require(grant.grant_id).use_count == 2
    assert store.unattended_executions()[grant.grant_id] == 2
    events = EventLog(runtime_root / "events").filter_events(event_type="consent_grant_used")
    assert len(events) == 2
    assert events[-1].metadata["use_count"] == 2


def test_expiry_sweep_emits_once(tmp_path):
    clock, runtime_root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path, ttl_s=60)

    assert store.expire_stale() == []
    clock.advance(61)
    assert store.expire_stale() == [grant.grant_id]
    assert store.expire_stale() == []

    events = EventLog(runtime_root / "events").filter_events(event_type="consent_grant_expired")
    assert len(events) == 1


def test_a_malformed_grant_file_authorizes_nothing(tmp_path):
    _clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)
    store.grant_path(grant.grant_id).write_text("{ not json", encoding="utf-8")

    assert store.load(grant.grant_id) is None
    assert store.list_grants() == []
    assert store.match(_shell_request(tmp_path), metadata_for_capability("shell_command", "tool")) is None


def test_a_grant_with_unparseable_expiry_is_rejected(tmp_path):
    with pytest.raises(ConsentGrantError, match="expires_at"):
        ConsentGrant(
            grant_id="grant-x",
            grantee_id=OBJECTIVE,
            capability_name="shell_command",
            scope_roots=["/tmp"],
            max_risk=1.0,
            granted_by="root",
            granted_at="2026-01-01T00:00:00+00:00",
            expires_at="whenever",
        )


# -- the policy engine's second override path ------------------------------


def test_policy_engine_blocks_risk_one_without_a_grant(tmp_path):
    _clock, _root, _agents, _store = _fixture(tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")

    decision = PolicyEngine().evaluate(_shell_request(tmp_path), metadata)

    assert decision.allowed is False
    assert decision.status == "requires_approval"


def test_policy_engine_allows_risk_one_under_a_matching_grant(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    grant = _grant(store, agents, tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")

    decision = PolicyEngine().evaluate(
        _shell_request(tmp_path),
        metadata,
        consent_grants=store.list_active(objective_id=OBJECTIVE),
        now=clock.now(),
    )

    assert decision.allowed is True
    assert decision.reason == "consent_grant"
    assert decision.metadata["consent_grant_id"] == grant.grant_id
    assert decision.metadata["grantee_id"] == OBJECTIVE


def test_policy_engine_ignores_a_grant_for_another_objective(tmp_path):
    clock, _root, agents, store = _fixture(tmp_path)
    _grant(store, agents, tmp_path, objective_id=OTHER_OBJECTIVE)
    metadata = metadata_for_capability("shell_command", "tool")

    decision = PolicyEngine().evaluate(
        _shell_request(tmp_path),
        metadata,
        consent_grants=store.list_grants(),
        now=clock.now(),
    )

    assert decision.allowed is False
    assert decision.status == "requires_approval"


def test_a_grant_does_not_authorize_direct_invocation(tmp_path):
    """Standing consent authorizes governed workflow execution only."""
    clock, _root, agents, store = _fixture(tmp_path)
    _grant(store, agents, tmp_path)
    metadata = metadata_for_capability("shell_command", "tool")
    direct = CapabilityRequest(
        actor_id="agent-1",
        actor_type="mr1",
        actor_clearance=0.99,
        invocation_mode="direct",
        capability_name="shell_command",
        args={"argv": ["pytest"], "cwd": str(tmp_path)},
        scope=ScopeContext(allowed_roots=[tmp_path], workspace_root=tmp_path),
        objective_id=OBJECTIVE,
    )

    decision = PolicyEngine().evaluate(
        direct,
        metadata,
        consent_grants=store.list_active(),
        now=clock.now(),
    )

    assert decision.allowed is False
    assert decision.reason == "capability_not_allowed_in_direct_mode"


def test_a_grant_overrides_actor_scope_only_within_its_own_roots(tmp_path):
    """The grant carries the scope; the agent's own scope no longer has to."""
    clock, _root, agents, store = _fixture(tmp_path)
    granted_dir = tmp_path / "granted"
    granted_dir.mkdir()
    _grant(store, agents, tmp_path, scope_roots=[granted_dir])
    metadata = metadata_for_capability("shell_command", "tool")

    request = CapabilityRequest(
        actor_id="agent-1",
        actor_type="mr1",
        actor_clearance=0.99,
        invocation_mode="workflow",
        capability_name="shell_command",
        args={"argv": ["pytest"], "cwd": str(granted_dir)},
        # The actor's own scope does NOT contain granted_dir.
        scope=ScopeContext(allowed_roots=[tmp_path / "other"], workspace_root=tmp_path),
        workflow_id="wf-1",
        task_id="tk-1",
        objective_id=OBJECTIVE,
    )

    decision = PolicyEngine().evaluate(
        request,
        metadata,
        consent_grants=store.list_active(),
        now=clock.now(),
    )

    assert decision.allowed is True
    assert decision.reason == "consent_grant"


def test_one_off_approval_matching_is_unchanged_by_consent(tmp_path):
    """`_approved_override_matches` keeps its single-use semantics exactly."""
    clock, _root, agents, store = _fixture(tmp_path)
    engine = PolicyEngine()
    metadata = metadata_for_capability("shell_command", "tool")
    request = _shell_request(tmp_path, objective_id=None)

    assert engine._approved_override_matches(None, request=request, metadata=metadata) is False
    decision = engine.evaluate(request, metadata, consent_grants=[], now=clock.now())
    assert decision.status == "requires_approval"
