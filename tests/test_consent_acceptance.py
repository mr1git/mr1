"""
A4 acceptance gate.

The whole point of Phase A, proven end to end through the real scheduler:

  1. a shell workflow under a matching objective grant runs unattended;
  2. the same workflow without a grant blocks for a human;
  3. an expired / revoked / non-matching grant authorizes nothing;
  4. every outcome is visible in the capability audit record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mr1.autonomy.consent import ConsentGrantStore
from mr1.capability_policy import CapabilityApprovalStore
from mr1.clock import VirtualClock
from mr1.event_log import EventLog
from mr1.worker_runner import MockRunner
from mr1.messages import MessageStore
from mr1.scheduler import Scheduler
from mr1.scoped_agents import AgentStore
from mr1.workflow_models import Provenance, TaskStatus, WorkflowStatus
from mr1.workflow_store import WorkflowStore


OBJECTIVE = "obj-genesis"


class _Fixture:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        self.runtime_root = tmp_path / "runtime"
        self.clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.store = WorkflowStore(root=self.runtime_root / "workflows")
        self.agents = AgentStore(root=self.runtime_root / "agents")
        self.messages = MessageStore(
            root=self.runtime_root / "messages",
            scoped_agent_store=self.agents,
        )
        self.consent = ConsentGrantStore(
            self.runtime_root,
            clock=self.clock,
            scoped_agent_store=self.agents,
        )
        self.approvals = CapabilityApprovalStore(
            self.runtime_root / "capability_approvals",
            clock=self.clock,
        )
        self.scheduler = Scheduler(
            self.store,
            MockRunner(),
            auto_tick=False,
            scoped_agent_store=self.agents,
            message_store=self.messages,
            workspace_root=self.workspace,
            clock=self.clock,
            consent_store=self.consent,
        )
        self.scheduler._approval_store = self.approvals
        self.scheduler._capability_gate._approval_store = self.approvals

    def grant(self, **overrides):
        payload = {
            "objective_id": OBJECTIVE,
            "capability_name": "shell_command",
            "scope_roots": [self.workspace],
            "max_risk": 1.0,
            "granted_by": self.agents.root_agent_id,
            "ttl_s": 7 * 86_400,
            "arg_predicate": {"argv": {"regex": r"^echo\b"}},
            "reason": "acceptance gate",
        }
        payload.update(overrides)
        return self.consent.create(**payload)

    def submit(self, *, objective_id=OBJECTIVE, argv=None):
        spec = {
            "title": "shell under consent",
            "tasks": [{
                "label": "run",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {
                    "argv": argv or ["echo", "hello"],
                    "cwd": str(self.workspace),
                },
            }],
        }
        metadata = {"objective_id": objective_id} if objective_id else None
        return self.scheduler.submit_workflow(
            spec,
            Provenance(type="agent", id="supervisor"),
            workflow_metadata=metadata,
        )

    def drive(self, workflow_id, ticks=4):
        for _ in range(ticks):
            self.scheduler.tick()
        return self.store.load_workflow(workflow_id)

    def task(self, workflow):
        return next(iter(workflow.tasks.values()))

    def audit(self, workflow_id, task_id, attempt=1):
        path = (
            self.store.task_attempt_dir(workflow_id, task_id, attempt)
            / "capability_audit.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def events(self, **kwargs):
        return EventLog(self.runtime_root / "events").filter_events(**kwargs)


@pytest.fixture
def fx(tmp_path):
    return _Fixture(tmp_path)


# -- 1. runs unattended under a matching grant -----------------------------


def test_shell_workflow_runs_unattended_under_a_matching_grant(fx):
    grant = fx.grant()

    workflow_id = fx.submit()
    workflow = fx.drive(workflow_id)
    task = fx.task(workflow)

    assert task.status is TaskStatus.SUCCEEDED, task.last_error
    assert workflow.status is WorkflowStatus.SUCCEEDED
    # Nobody was asked.
    assert fx.approvals.list_requests() == []

    audit = fx.audit(workflow_id, task.task_id)
    assert audit["decision"]["allowed"] is True
    assert audit["decision"]["reason"] == "consent_grant"
    assert audit["decision"]["metadata"]["consent_grant_id"] == grant.grant_id
    assert audit["request"]["objective_id"] == OBJECTIVE

    allowed = fx.events(event_type="capability_allowed")
    assert allowed[-1].metadata["consent_grant_id"] == grant.grant_id
    assert allowed[-1].metadata["objective_id"] == OBJECTIVE

    assert fx.consent.require(grant.grant_id).use_count == 1
    assert fx.events(event_type="consent_grant_used")


# -- 2. blocks for a human without one -------------------------------------


def test_the_same_workflow_without_a_grant_blocks_for_approval(fx):
    workflow_id = fx.submit()
    workflow = fx.drive(workflow_id)
    task = fx.task(workflow)

    assert task.status is TaskStatus.BLOCKED
    assert task.last_error_type == "approval_required"

    approvals = fx.approvals.list_requests()
    assert len(approvals) == 1
    assert approvals[0].status == "pending"
    assert approvals[0].capability_name == "shell_command"

    audit = fx.audit(workflow_id, task.task_id)
    assert audit["decision"]["allowed"] is False
    assert audit["decision"]["status"] == "requires_approval"
    assert fx.events(event_type="capability_blocked")


def test_a_workflow_with_no_objective_cannot_ride_a_grant(fx):
    fx.grant()

    workflow_id = fx.submit(objective_id=None)
    task = fx.task(fx.drive(workflow_id))

    assert task.status is TaskStatus.BLOCKED
    assert task.last_error_type == "approval_required"


# -- 3. expired / revoked / non-matching authorize nothing -----------------


def test_an_expired_grant_does_not_authorize(fx):
    fx.grant(ttl_s=3600)
    fx.clock.advance(3601)

    task = fx.task(fx.drive(fx.submit()))

    assert task.status is TaskStatus.BLOCKED
    assert task.last_error_type == "approval_required"


def test_a_revoked_grant_does_not_authorize(fx):
    grant = fx.grant()
    fx.consent.revoke(grant.grant_id, revoked_by=fx.agents.root_agent_id, reason="nope")

    task = fx.task(fx.drive(fx.submit()))

    assert task.status is TaskStatus.BLOCKED
    assert fx.consent.require(grant.grant_id).use_count == 0


def test_a_grant_for_another_objective_does_not_authorize(fx):
    fx.grant(objective_id="obj-other")

    task = fx.task(fx.drive(fx.submit()))

    assert task.status is TaskStatus.BLOCKED


def test_a_command_outside_the_predicate_does_not_authorize(fx):
    fx.grant(arg_predicate={"argv": {"regex": r"^pytest\b"}})

    task = fx.task(fx.drive(fx.submit(argv=["echo", "sneaky"])))

    assert task.status is TaskStatus.BLOCKED


def test_a_path_outside_the_grant_scope_does_not_authorize(fx):
    other = fx.tmp_path / "elsewhere"
    other.mkdir()
    fx.grant(scope_roots=[other])

    task = fx.task(fx.drive(fx.submit()))

    assert task.status is TaskStatus.BLOCKED


# -- 4. revocation takes effect on the next invocation ---------------------


def test_revoking_a_grant_blocks_the_next_run(fx):
    grant = fx.grant()

    first = fx.task(fx.drive(fx.submit()))
    assert first.status is TaskStatus.SUCCEEDED

    fx.consent.revoke(grant.grant_id, revoked_by=fx.agents.root_agent_id, reason="revoked")

    second = fx.task(fx.drive(fx.submit()))
    assert second.status is TaskStatus.BLOCKED
    assert fx.consent.require(grant.grant_id).use_count == 1


def test_halt_revokes_standing_authority(fx):
    from mr1.cli.service import halt_runtime

    fx.grant()
    assert len(fx.consent.list_active()) == 1

    payload = halt_runtime(
        fx.runtime_root,
        reason="emergency",
        requested_by="marwan",
        clock=fx.clock,
    )

    assert len(payload["revoked_grants"]) == 1
    assert fx.consent.list_active() == []

    task = fx.task(fx.drive(fx.submit()))
    assert task.status is TaskStatus.BLOCKED
