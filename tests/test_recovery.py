"""A6 — the failure ladder. Pure, deterministic, no LLM."""

from __future__ import annotations

import random

import pytest

from mr1.autonomy.recovery import (
    FailureClass,
    FailurePolicy,
    FailureSignal,
    RecoveryKind,
    RecoveryState,
    backoff_for,
    classify,
    decide,
)
from mr1.workflow_models import (
    Provenance,
    Task,
    TaskStatus,
    Workflow,
    WorkflowStatus,
)


def _workflow(*tasks, status=WorkflowStatus.FAILED) -> Workflow:
    workflow = Workflow(
        workflow_id="wf-1",
        title="test",
        status=status,
        created_by=Provenance(type="agent", id="supervisor"),
    )
    for task in tasks:
        workflow.tasks[task.task_id] = task
    return workflow


def _task(
    task_id: str,
    status: TaskStatus,
    *,
    error_type=None,
    error=None,
) -> Task:
    return Task(
        task_id=task_id,
        workflow_id="wf-1",
        label=task_id,
        title=task_id,
        task_kind="agent",
        agent_type="kazi",
        prompt="do the thing",
        status=status,
        last_error_type=error_type,
        last_error=error,
    )


# -- classification --------------------------------------------------------


def test_timeout_is_transient():
    workflow = _workflow(_task("t1", TaskStatus.TIMED_OUT, error_type="timeout"))
    signal = classify(workflow)
    assert signal.classification is FailureClass.TRANSIENT


def test_infrastructure_failure_is_transient():
    """A task that died with the runtime must not become a permanent failure."""
    workflow = _workflow(
        _task("t1", TaskStatus.FAILED, error_type="infrastructure_failure", error="run handle lost")
    )
    assert classify(workflow).classification is FailureClass.TRANSIENT


def test_auth_and_internal_errors_are_transient():
    for error_type in ("auth_error", "internal_error", "tool_unavailable"):
        workflow = _workflow(_task("t1", TaskStatus.FAILED, error_type=error_type))
        assert classify(workflow).classification is FailureClass.TRANSIENT, error_type


def test_a_task_that_failed_on_its_merits_is_a_planning_failure():
    workflow = _workflow(
        _task("t1", TaskStatus.FAILED, error_type="unknown", error="command exited 2")
    )
    signal = classify(workflow)
    assert signal.classification is FailureClass.PLANNING
    assert signal.task_id == "t1"


def test_approval_required_is_blocked():
    workflow = _workflow(
        _task("t1", TaskStatus.BLOCKED, error_type="approval_required", error="needs a human")
    )
    signal = classify(workflow)
    assert signal.classification is FailureClass.BLOCKED
    assert signal.reason == "approval_required"


def test_policy_block_is_blocked():
    workflow = _workflow(_task("t1", TaskStatus.FAILED, error_type="policy_block"))
    assert classify(workflow).classification is FailureClass.BLOCKED


def test_cancellation_is_blocked_not_retried():
    assert classify(_workflow(status=WorkflowStatus.CANCELLED)).classification is FailureClass.BLOCKED
    workflow = _workflow(_task("t1", TaskStatus.CANCELLED))
    assert classify(workflow).classification is FailureClass.BLOCKED


def test_blocked_wins_over_every_other_failure():
    """If a human is needed, no amount of retrying by MR1 can help."""
    workflow = _workflow(
        _task("t1", TaskStatus.TIMED_OUT, error_type="timeout"),
        _task("t2", TaskStatus.BLOCKED, error_type="approval_required"),
        _task("t3", TaskStatus.FAILED, error_type="unknown"),
    )
    assert classify(workflow).classification is FailureClass.BLOCKED


def test_a_mixed_transient_and_merit_failure_is_a_planning_failure():
    workflow = _workflow(
        _task("t1", TaskStatus.TIMED_OUT, error_type="timeout"),
        _task("t2", TaskStatus.FAILED, error_type="unknown", error="assertion failed"),
    )
    assert classify(workflow).classification is FailureClass.PLANNING


def test_a_failed_workflow_with_no_failed_task_still_classifies():
    workflow = _workflow(_task("t1", TaskStatus.SUCCEEDED))
    assert classify(workflow).classification is FailureClass.PLANNING


def test_classify_rejects_a_succeeded_workflow():
    with pytest.raises(ValueError):
        classify(_workflow(status=WorkflowStatus.SUCCEEDED))


def test_the_same_failure_has_the_same_signature():
    first = _workflow(_task("t1", TaskStatus.FAILED, error_type="unknown", error="cannot find module foo"))
    second = _workflow(_task("t9", TaskStatus.FAILED, error_type="unknown", error="cannot find module foo"))
    assert classify(first).signature == classify(second).signature


def test_signatures_ignore_volatile_ids_and_numbers():
    first = _workflow(_task("t1", TaskStatus.FAILED, error_type="unknown", error="task tk-111 failed after 3s"))
    second = _workflow(_task("t1", TaskStatus.FAILED, error_type="unknown", error="task tk-999 failed after 8s"))
    assert classify(first).signature == classify(second).signature


def test_different_failures_have_different_signatures():
    first = _workflow(_task("t1", TaskStatus.FAILED, error_type="unknown", error="module not found"))
    second = _workflow(_task("t1", TaskStatus.FAILED, error_type="unknown", error="permission denied"))
    assert classify(first).signature != classify(second).signature


# -- backoff ---------------------------------------------------------------


def test_backoff_is_30s_2m_8m():
    policy = FailurePolicy()
    assert backoff_for(1, policy) == 30.0
    assert backoff_for(2, policy) == 120.0
    assert backoff_for(3, policy) == 480.0


def test_backoff_is_capped():
    policy = FailurePolicy(backoff_max_s=600.0)
    assert backoff_for(10, policy) == 600.0


def test_backoff_jitter_stays_within_the_ratio():
    policy = FailurePolicy()
    rng = random.Random(7)
    for _ in range(50):
        delay = backoff_for(1, policy, rng=rng)
        assert 30.0 <= delay <= 33.0


def test_backoff_without_an_rng_is_deterministic():
    policy = FailurePolicy()
    assert backoff_for(2, policy) == backoff_for(2, policy)


# -- the ladder ------------------------------------------------------------


def _signal(cls=FailureClass.TRANSIENT, *, signature_detail="boom"):
    return FailureSignal(
        classification=cls,
        reason="test",
        error_type="timeout" if cls is FailureClass.TRANSIENT else "unknown",
        detail=signature_detail,
    )


def test_transient_retries_with_backoff_inside_the_budget():
    action = decide(_signal(), RecoveryState(retries_used=0), FailurePolicy())
    assert action.kind is RecoveryKind.RETRY
    assert action.delay_s == 30.0
    assert action.terminal is False

    action = decide(_signal(), RecoveryState(retries_used=2), FailurePolicy())
    assert action.kind is RecoveryKind.RETRY
    assert action.delay_s == 480.0


def test_transient_escalates_to_a_replan_when_retries_run_out():
    action = decide(_signal(), RecoveryState(retries_used=3), FailurePolicy())
    assert action.kind is RecoveryKind.REPLAN
    assert "retry budget exhausted" in action.reason


def test_planning_replans_inside_the_budget():
    action = decide(_signal(FailureClass.PLANNING), RecoveryState(), FailurePolicy())
    assert action.kind is RecoveryKind.REPLAN


def test_planning_quarantines_when_replans_run_out():
    action = decide(
        _signal(FailureClass.PLANNING),
        RecoveryState(replans_used=2),
        FailurePolicy(),
    )
    assert action.kind is RecoveryKind.QUARANTINE
    assert action.escalate is True
    assert action.terminal is True


def test_an_objective_with_a_fallback_gets_one_more_level():
    action = decide(
        _signal(FailureClass.PLANNING),
        RecoveryState(replans_used=2, has_fallback=True),
        FailurePolicy(),
    )
    assert action.kind is RecoveryKind.FALLBACK
    assert action.terminal is False

    exhausted = decide(
        _signal(FailureClass.PLANNING),
        RecoveryState(replans_used=2, fallbacks_used=1, has_fallback=True),
        FailurePolicy(),
    )
    assert exhausted.kind is RecoveryKind.QUARANTINE


def test_blocked_always_escalates_and_never_self_authorizes():
    for state in (
        RecoveryState(),
        RecoveryState(retries_used=0, replans_used=0),
        RecoveryState(has_fallback=True),
    ):
        action = decide(_signal(FailureClass.BLOCKED), state, FailurePolicy())
        assert action.kind is RecoveryKind.ESCALATE
        assert action.escalate is True
        assert action.terminal is True


def test_fatal_quarantines_immediately():
    action = decide(_signal(FailureClass.FATAL), RecoveryState(), FailurePolicy())
    assert action.kind is RecoveryKind.QUARANTINE
    assert action.terminal is True


def test_consecutive_failures_quarantine_regardless_of_class():
    action = decide(
        _signal(FailureClass.TRANSIENT),
        RecoveryState(consecutive_failures=5),
        FailurePolicy(),
    )
    assert action.kind is RecoveryKind.QUARANTINE
    assert "consecutive failures" in action.reason


def test_an_elapsed_runtime_budget_terminates_the_objective():
    action = decide(
        _signal(FailureClass.TRANSIENT),
        RecoveryState(elapsed_s=8 * 86_400),
        FailurePolicy(max_elapsed_s=7 * 86_400),
    )
    assert action.kind is RecoveryKind.QUARANTINE
    assert "runtime budget" in action.reason


def test_the_same_failure_repeated_eventually_terminates():
    signal = _signal(FailureClass.TRANSIENT)
    policy = FailurePolicy(max_identical_failures=3)

    twice = decide(signal, RecoveryState(recent_signatures=[signal.signature]), policy)
    assert twice.kind is RecoveryKind.RETRY

    thrice = decide(
        signal,
        RecoveryState(recent_signatures=[signal.signature, signal.signature]),
        policy,
    )
    assert thrice.kind is RecoveryKind.QUARANTINE
    assert "same failure repeated" in thrice.reason


def test_a_broken_repeat_run_does_not_terminate():
    signal = _signal(FailureClass.TRANSIENT)
    other = _signal(FailureClass.TRANSIENT, signature_detail="different")
    action = decide(
        signal,
        RecoveryState(recent_signatures=[signal.signature, other.signature]),
        FailurePolicy(max_identical_failures=3),
    )
    assert action.kind is RecoveryKind.RETRY


def test_nothing_retries_forever():
    """Drive every class to exhaustion; every path must terminate."""
    policy = FailurePolicy()
    for classification in FailureClass:
        state = RecoveryState()
        signatures: list[str] = []
        terminated = False
        for step in range(20):
            signal = _signal(classification)
            action = decide(
                signal,
                RecoveryState(
                    retries_used=state.retries_used,
                    replans_used=state.replans_used,
                    fallbacks_used=state.fallbacks_used,
                    consecutive_failures=step,
                    recent_signatures=list(signatures),
                ),
                policy,
            )
            signatures.append(signal.signature)
            if action.terminal:
                terminated = True
                break
            state = RecoveryState(
                retries_used=state.retries_used + (1 if action.kind is RecoveryKind.RETRY else 0),
                replans_used=state.replans_used + (1 if action.kind is RecoveryKind.REPLAN else 0),
                fallbacks_used=state.fallbacks_used + (1 if action.kind is RecoveryKind.FALLBACK else 0),
            )
        assert terminated, f"{classification} never reached a terminal action"


def test_policy_round_trips():
    policy = FailurePolicy(max_retries=1, max_replans=0)
    assert FailurePolicy.from_dict(policy.to_dict()) == policy
    assert FailurePolicy.from_dict(None) == FailurePolicy()


def test_policy_rejects_nonsense_budgets():
    with pytest.raises(ValueError):
        FailurePolicy(max_retries=-1).validate()
    with pytest.raises(ValueError):
        FailurePolicy(max_consecutive_failures=0).validate()
