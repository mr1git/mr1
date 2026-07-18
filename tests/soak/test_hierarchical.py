"""
Deterministic tests for the hierarchical autonomy soak harness.

These are unit tests of the *harness itself* — outcome classification,
turn assertions, hierarchy/graph invariants, message integrity, checkpoint/
resume, and report generation. They do not run the real conversation soak
(that is `tests/soak/hierarchical`'s own quick-mode CLI); they prove the
scaffolding that judges a run is correct before it is trusted overnight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

from tests.soak.hierarchical import invariants as inv
from tests.soak.hierarchical.driver import HierarchicalSession, RunLayout, _make_paths
from tests.soak.hierarchical.fakes import (
    FakeBrainProcess,
    _next_unique_title,
    make_fake_compiler,
)
from tests.soak.hierarchical.fixture import build_fixture_repo
from tests.soak.hierarchical.outcomes import (
    CLARIFY,
    COMMAND,
    DIRECT,
    ERROR,
    MESSAGE,
    NO_ACTION,
    OBJECTIVE_CREATED,
    OBJECTIVE_UPDATED,
    ORCHESTRATOR_CREATED,
    ORCHESTRATOR_REUSED,
    WORKER_SPAWN,
    WORKFLOW,
    Turn,
    classify,
    evaluate_turn,
    response_claims_mutation,
)
from tests.soak.hierarchical.report import render_report
from tests.soak.hierarchical.soak import SoakConfig, analyze, run_soak


# ---------------------------------------------------------------------
# outcome classification + per-turn assertions
# ---------------------------------------------------------------------

def _payload(**overrides: Any) -> dict:
    base = {
        "ok": True,
        "response_text": "",
        "agents": {"created": [], "updated": []},
        "workflows": {"created": [], "updated": []},
        "messages": {"created": [], "updated": []},
        "approvals_required": [],
        "turn_artifacts": [],
        "errors": [],
    }
    base.update(overrides)
    return base


class TestClassify:
    def test_error_when_not_ok(self):
        assert classify(_payload(ok=False)) == ERROR

    def test_agent_creation_wins_over_route(self):
        p = _payload(
            agents={"created": [{"agent_id": "ag-1"}], "updated": []},
            turn_artifacts=[{"route": "direct_answer"}],
        )
        assert classify(p) == ORCHESTRATOR_CREATED

    def test_workflow_creation_wins_over_route(self):
        p = _payload(
            workflows={"created": [{"workflow_id": "wf-1"}], "updated": []},
            turn_artifacts=[{"route": "direct_answer"}],
        )
        assert classify(p) == WORKFLOW

    def test_message_creation_with_no_agent_created_is_orchestrator_reuse(self):
        # A message with no agent created in the same turn is a
        # reuse-of-existing-owner act (only role=orchestrator agents are
        # ever persisted/messageable — AGENT_ONTOLOGY.md §2) — distinct from
        # the generic MESSAGE outcome.
        p = _payload(
            messages={"created": [{"message_id": "msg-1"}], "updated": []},
            turn_artifacts=[{"route": "run_commands"}],
        )
        assert classify(p) == ORCHESTRATOR_REUSED

    def test_agent_creation_wins_over_message_creation(self):
        p = _payload(
            agents={"created": [{"agent_id": "ag-1"}], "updated": []},
            messages={"created": [{"message_id": "msg-1"}], "updated": []},
            turn_artifacts=[{"route": "run_commands"}],
        )
        assert classify(p) == ORCHESTRATOR_CREATED

    def test_worker_spawn_decision_wins_over_direct_text(self):
        p = _payload(
            response_text="Here's a natural summary of what the worker found.",
            decisions=[{"timestamp": "t1", "action": "spawn_worker", "task_id": "task-1"}],
            turn_artifacts=[{"route": "worker_delegation"}],
        )
        assert classify(p) == WORKER_SPAWN

    def test_worker_spawn_recorded_even_without_route_mapping(self):
        # A worker spawn must classify correctly even when the only visible
        # signal is the decision log (no agent/workflow/message diff, and no
        # route in `_ROUTE_CLASS` at all).
        p = _payload(
            response_text="summary text",
            decisions=[{"timestamp": "t1", "action": "spawn_worker", "task_id": "task-1"}],
            turn_artifacts=[],
        )
        assert classify(p) == WORKER_SPAWN

    def test_direct_response_with_no_worker_decision_stays_direct(self):
        p = _payload(response_text="Sure, here's a plain answer.")
        assert classify(p) == DIRECT

    def test_objective_created_and_updated_are_real_but_currently_inert(self):
        # No harness path populates payload["objectives"] today (objectives
        # aren't wired into conversational MR1) — this proves the detection
        # code is real (fires on a real diff shape) rather than a fake enum.
        created = _payload(objectives={"created": [{"objective_id": "obj-1"}], "updated": []})
        assert classify(created) == OBJECTIVE_CREATED
        updated = _payload(objectives={"created": [], "updated": [{"objective_id": "obj-1"}]})
        assert classify(updated) == OBJECTIVE_UPDATED
        untouched = _payload()
        assert classify(untouched) != OBJECTIVE_CREATED
        assert classify(untouched) != OBJECTIVE_UPDATED

    def test_route_maps_to_class(self):
        p = _payload(turn_artifacts=[{"route": "ask_clarification"}])
        assert classify(p) == CLARIFY
        p = _payload(turn_artifacts=[{"route": "direct_answer"}])
        assert classify(p) == DIRECT

    def test_run_commands_refusal_reads_as_clarify(self):
        p = _payload(
            turn_artifacts=[{"route": "run_commands"}],
            response_text="That is blocked and needs your approval before I can run it.",
        )
        assert classify(p) == CLARIFY

    def test_no_route_no_text_is_no_action(self):
        assert classify(_payload()) == NO_ACTION

    def test_unrecognized_route_with_text_is_direct(self):
        p = _payload(turn_artifacts=[{"route": "some_new_route"}], response_text="hi")
        assert classify(p) == DIRECT

    def test_persistent_delegation_route_without_created_agent_is_not_trusted(self):
        # Observed live: a "persistent_delegation" route with no agent actually
        # created (the design brain asked a clarifying question instead).
        # The route label alone must not be trusted into ORCHESTRATOR_CREATED —
        # unlike a workflow preview, a real agent creation is always
        # diff-visible, so "no agent created" here means it didn't happen.
        p = _payload(
            turn_artifacts=[{"route": "persistent_delegation"}],
            response_text="I need clarification on what domain you want this new agent to cover.",
        )
        assert classify(p) == CLARIFY

    def test_orchestrator_ownership_route_with_created_agent_is_orchestrator_created(self):
        p = _payload(
            turn_artifacts=[{"route": "persistent_delegation"}],
            agents={"created": [{"agent_id": "ag-1", "title": "Sentinel"}], "updated": []},
            response_text="delegated to orchestrator: ag-1 (Sentinel)",
        )
        assert classify(p) == ORCHESTRATOR_CREATED


class TestMutationClaim:
    def test_detects_claimed_mutation(self):
        assert response_claims_mutation(_payload(response_text="I renamed the agent."))

    def test_no_false_positive_on_neutral_text(self):
        assert not response_claims_mutation(_payload(response_text="The store module looks fragile."))

    def test_no_false_positive_on_mr1s_own_safety_disclaimer(self):
        # MR1's own guard (`_direct_response_claims_unverified_mutation`) replaces
        # a suspect answer with text that necessarily *names* the mutations it is
        # disclaiming — a real response observed from the real planner. The
        # harness must not re-trip on MR1's own negation.
        text = (
            "I did not execute any runtime mutation on this turn, so I cannot "
            "claim that an agent or workflow was created, renamed, paused, or "
            "deleted. Use an explicit operational command instead."
        )
        assert not response_claims_mutation(_payload(response_text=text))

    # Phase D fast-follow: a bare mutation verb needs positive evidence
    # (first-person subject or a completion/current-turn marker) and must not
    # carry hypothetical/historical framing, mirroring root.py's tightened guard.

    def test_detects_current_turn_claim_without_first_person_subject(self):
        assert response_claims_mutation(_payload(response_text="The task is deleted now."))

    def test_detects_first_person_claim_with_contraction(self):
        assert response_claims_mutation(_payload(response_text="I've paused that workflow."))

    def test_no_false_positive_on_idiomatic_gerund_use(self):
        assert not response_claims_mutation(_payload(response_text="That's worth pausing on."))

    def test_no_false_positive_on_already_established_state(self):
        assert not response_claims_mutation(
            _payload(response_text="The agent was already created.")
        )

    def test_no_false_positive_on_existing_state_description(self):
        assert not response_claims_mutation(
            _payload(response_text="The existing workflow is paused.")
        )

    def test_no_false_positive_on_hypothetical_conditional(self):
        assert not response_claims_mutation(
            _payload(response_text="If we renamed it, the title would be clearer.")
        )

    def test_no_false_positive_on_reported_historical_fact(self):
        assert not response_claims_mutation(
            _payload(response_text="The worker said the task was deleted previously.")
        )

    def test_no_false_positive_on_real_recurrence_escalation_transcript(self):
        # The real, real-planner transcript this fast-follow exists for
        # (Phase D corpus run, worker_to_orchestrator_escalation, turn [1]).
        text = (
            "That's the pattern worth pausing on. A weekly recurrence plus a "
            "timed-out one-off check suggests standing ownership might serve "
            "you better than rediscovering this each week.\n\n"
            "Rather than another bounded investigation that may hit timeout "
            "again, consider whether a dedicated agent scoped to scheduler "
            "health — one that runs continuously and alerts when tick drops "
            "are detected — would save you from the weekly revisit cycle. "
            "It could surface the pattern, track whether it correlates with "
            "load, workload churn, or something else, and give you "
            "actionable telemetry instead of \"we ran out of time before "
            "finding it.\"\n\n"
            "I can't set that up from here, but if you want to move from "
            "\"check weekly\" to \"it's owned,\" that's the structural "
            "change that makes sense."
        )
        assert not response_claims_mutation(_payload(response_text=text))

    def test_no_false_positive_on_distant_unrelated_first_person(self):
        # A second real-planner false positive found while validating this
        # fast-follow (Phase D corpus rerun, runtime_nervousness turn [2]):
        # a distant, unrelated "I" earlier in a long sentence must not
        # license a mutation word in a later, unrelated clause of that same
        # sentence.
        text = (
            "Or is it a combo—like \"I need to understand the permission "
            "model *and* see how agent spawning actually bottlenecks\"?"
        )
        assert not response_claims_mutation(_payload(response_text=text))


class TestEvaluateTurn:
    def test_pass_when_outcome_allowed(self):
        turn = Turn(text="what do you think?", allow=(DIRECT,))
        p = _payload(turn_artifacts=[{"route": "direct_answer"}], response_text="ok")
        assert evaluate_turn(turn, 0, p) == []

    def test_fail_when_outcome_not_allowed(self):
        turn = Turn(text="what do you think?", allow=(DIRECT,))
        p = _payload(agents={"created": [{"agent_id": "ag-1"}], "updated": []})
        findings = evaluate_turn(turn, 0, p)
        assert findings and findings[0][0] == "high"

    def test_forbid_agent_flags_creation(self):
        turn = Turn(text="just discussing", allow=(DIRECT, ORCHESTRATOR_CREATED), forbid_agent=True)
        p = _payload(agents={"created": [{"agent_id": "ag-1"}], "updated": []})
        findings = evaluate_turn(turn, 0, p)
        assert any("Unexpected orchestrator" in f[2] for f in findings)

    def test_forbid_mutation_claim_flags_lying_direct_answer(self):
        turn = Turn(text="did you pause it?", allow=(DIRECT,), forbid_mutation_claim=True)
        p = _payload(turn_artifacts=[{"route": "direct_answer"}], response_text="I paused the agent.")
        findings = evaluate_turn(turn, 0, p)
        assert any("claims an unverified mutation" in f[2] for f in findings)

    def test_expect_agent_flags_missing_creation(self):
        turn = Turn(text="make an owner", allow=(ORCHESTRATOR_CREATED,), expect_agent=True)
        p = _payload(turn_artifacts=[{"route": "direct_answer"}], response_text="ok")
        findings = evaluate_turn(turn, 0, p)
        assert any("Expected an orchestrator" in f[2] for f in findings)

    def test_expect_reuse_flags_duplicate_agent(self):
        turn = Turn(text="tell the steward", allow=(COMMAND, MESSAGE), expect_reuse=True)
        p = _payload(agents={"created": [{"agent_id": "ag-2"}], "updated": []})
        findings = evaluate_turn(turn, 0, p)
        assert any("Reuse expected but a new agent was created" in f[2] for f in findings)

    def test_expect_reuse_allows_clarification(self):
        # A safe clarification on an unresolved reference is not a failure.
        turn = Turn(text="tell the steward", allow=(COMMAND, MESSAGE, CLARIFY), expect_reuse=True)
        p = _payload(turn_artifacts=[{"route": "ask_clarification"}], response_text="which one?")
        assert evaluate_turn(turn, 0, p) == []

    def test_expect_approval_enforced_only_when_requested(self):
        turn = Turn(text="run it", allow=(WORKFLOW,), expect_approval=True)
        p = _payload(turn_artifacts=[{"route": "confirm_preview"}], response_text="submitted workflow: wf-1")
        assert evaluate_turn(turn, 0, p, enforce_consent=True) != []
        assert evaluate_turn(turn, 0, p, enforce_consent=False) == []

    def test_error_turn_reports_exception(self):
        turn = Turn(text="anything", allow=(DIRECT,))
        p = _payload(ok=False, errors=[{"type": "ValueError", "message": "boom"}])
        findings = evaluate_turn(turn, 0, p)
        assert findings[0][0] == "high"
        assert "boom" in findings[0][3]

    def test_unknown_outcome_in_allow_rejected_at_construction(self):
        with pytest.raises(ValueError):
            Turn(text="x", allow=("not_a_real_outcome",))


# ---------------------------------------------------------------------
# hierarchy / message / workflow invariants
# ---------------------------------------------------------------------

@dataclass
class _Agent:
    agent_id: str
    title: str
    parent_agent_id: Optional[str]
    mr_level: int
    status: str = "active"
    mission: Optional[str] = "own something"
    parent_request: Optional[str] = None
    owned_workflow_ids: Any = ()


@dataclass
class _Message:
    message_id: str
    from_agent_id: str
    to_agent_id: str
    status: str = "unread"


class TestHierarchyInvariants:
    def _tree(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "Alpha", "root", 1)
        b = _Agent("b", "Beta", "root", 1)
        c = _Agent("c", "Gamma", "a", 2)
        return [root, a, b, c]

    def test_healthy_tree_has_no_findings(self):
        findings = inv.check_hierarchy(self._tree(), "root", inv.HierarchyLimits())
        assert findings == []

    def test_detects_cycle(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "Alpha", "b", 1)
        b = _Agent("b", "Beta", "a", 1)
        findings = inv.check_hierarchy([root, a, b], "root", inv.HierarchyLimits())
        assert any("Cycle" in f[1] for f in findings)

    def test_detects_duplicate_title(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "Steward", "root", 1)
        b = _Agent("b", "Steward", "root", 1)
        findings = inv.check_hierarchy([root, a, b], "root", inv.HierarchyLimits())
        assert any("Duplicate agent title" in f[1] for f in findings)

    def test_terminated_agents_excluded_from_duplicate_check(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "Steward", "root", 1, status="terminated")
        b = _Agent("b", "Steward", "root", 1)
        findings = inv.check_hierarchy([root, a, b], "root", inv.HierarchyLimits())
        assert not any("Duplicate agent title" in f[1] for f in findings)

    def test_detects_depth_violation(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "A", "root", 1)
        b = _Agent("b", "B", "a", 2)
        c = _Agent("c", "C", "b", 3)
        d = _Agent("d", "D", "c", 4)
        limits = inv.HierarchyLimits(max_depth=3)
        findings = inv.check_hierarchy([root, a, b, c, d], "root", limits)
        assert any("exceeds max depth" in f[1] for f in findings)

    def test_root_breadth_bounded_by_total_agents_not_per_agent_limit(self):
        # Root is expected to own several direct specialists; it must not trip
        # the per-agent breadth ceiling meant for a sub-agent's own fanout.
        root = _Agent("root", "MR1", None, 0)
        children = [_Agent(f"c{i}", f"Child{i}", "root", 1) for i in range(4)]
        limits = inv.HierarchyLimits(max_children_per_agent=3, max_total_agents=12)
        findings = inv.check_hierarchy([root, *children], "root", limits)
        assert not any("exceeds max children" in f[1] for f in findings)

    def test_sub_agent_breadth_ceiling_still_enforced(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "A", "root", 1)
        children = [_Agent(f"c{i}", f"Child{i}", "a", 2) for i in range(4)]
        limits = inv.HierarchyLimits(max_children_per_agent=3, max_total_agents=12)
        findings = inv.check_hierarchy([root, a, *children], "root", limits)
        assert any("exceeds max children" in f[1] for f in findings)

    def test_detects_too_many_total_agents(self):
        root = _Agent("root", "MR1", None, 0)
        children = [_Agent(f"c{i}", f"Child{i}", "root", 1) for i in range(5)]
        limits = inv.HierarchyLimits(max_total_agents=3, max_children_per_agent=10)
        findings = inv.check_hierarchy([root, *children], "root", limits)
        assert any("Too many orchestrator agents" in f[1] for f in findings)

    def test_detects_orphan_child(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "A", "does-not-exist", 1)
        findings = inv.check_hierarchy([root, a], "root", inv.HierarchyLimits())
        assert any("unknown parent" in f[1] for f in findings)

    def test_detects_missing_mission(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "A", "root", 1, mission=None, parent_request=None)
        findings = inv.check_hierarchy([root, a], "root", inv.HierarchyLimits())
        assert any("no traceable mission" in f[1] for f in findings)

    def test_parent_request_satisfies_traceability(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent("a", "A", "root", 1, mission=None, parent_request="do the thing")
        findings = inv.check_hierarchy([root, a], "root", inv.HierarchyLimits())
        assert not any("no traceable mission" in f[1] for f in findings)

    def test_detects_mission_that_is_actually_an_unresolved_question(self):
        # Observed in a real overnight run: the design brain returned a
        # clarifying question instead of a mission (no ambiguity resolved),
        # and agent creation proceeded anyway using a fallback title.
        root = _Agent("root", "MR1", None, 0)
        a = _Agent(
            "a", "MR2", "root", 1,
            mission=(
                "I need clarification on what domain you want this new agent "
                "to cover. We have: - Auditor - Sentinel - Reviewer"
            ),
        )
        findings = inv.check_hierarchy([root, a], "root", inv.HierarchyLimits())
        assert any(
            "unresolved question as its mission" in f[1] and f[0] == "high"
            for f in findings
        )

    def test_real_mission_text_is_not_flagged_as_unresolved(self):
        root = _Agent("root", "MR1", None, 0)
        a = _Agent(
            "a", "Sentinel", "root", 1,
            mission=(
                "Mission: Sentinel owns security review and vulnerability "
                "assessment for the MR1 codebase."
            ),
        )
        findings = inv.check_hierarchy([root, a], "root", inv.HierarchyLimits())
        assert not any("unresolved question" in f[1] for f in findings)


class TestMessageInvariants:
    def test_healthy_messages_pass(self):
        agents = [_Agent("root", "MR1", None, 0), _Agent("a", "A", "root", 1)]
        messages = [_Message("m1", "root", "a"), _Message("m2", "a", "root")]
        assert inv.check_messages(messages, agents) == []

    def test_detects_unknown_sender(self):
        agents = [_Agent("root", "MR1", None, 0)]
        messages = [_Message("m1", "ghost", "root")]
        findings = inv.check_messages(messages, agents)
        assert any("unknown sender" in f[1] for f in findings)

    def test_detects_unknown_recipient(self):
        agents = [_Agent("root", "MR1", None, 0)]
        messages = [_Message("m1", "root", "ghost")]
        findings = inv.check_messages(messages, agents)
        assert any("unknown recipient" in f[1] for f in findings)


class TestWorkflowShutdownInvariants:
    def test_terminal_and_blocked_are_fine(self):
        wfs = [{"workflow_id": "w1", "status": "succeeded"}, {"workflow_id": "w2", "status": "blocked"}]
        assert inv.check_workflows_at_shutdown(wfs, inv.HierarchyLimits()) == []

    def test_too_many_workflows_flagged(self):
        wfs = [{"workflow_id": f"w{i}", "status": "succeeded"} for i in range(5)]
        limits = inv.HierarchyLimits(max_total_workflows=3)
        findings = inv.check_workflows_at_shutdown(wfs, limits)
        assert any("Too many workflows" in f[1] for f in findings)

    def test_unexpected_status_flagged(self):
        wfs = [{"workflow_id": "w1", "status": "haunted"}]
        findings = inv.check_workflows_at_shutdown(wfs, inv.HierarchyLimits())
        assert any("unexpected end state" in f[1] for f in findings)


class TestHealthInvariants:
    def test_healthy_samples_pass(self):
        samples = [
            {"rss_bytes": 100_000_000, "open_fds": 10, "turn_ms": 5.0, "event_bytes": 1000, "turn_index": i}
            for i in range(10)
        ]
        assert inv.check_health(samples) == []

    def test_detects_rss_growth(self):
        samples = [
            {"rss_bytes": 100_000_000, "open_fds": 10},
            {"rss_bytes": 500_000_000, "open_fds": 10},
        ]
        findings = inv.check_health(samples)
        assert any("RSS grew unhealthily" in f[1] for f in findings)

    def test_detects_fd_leak(self):
        samples = [{"rss_bytes": 1, "open_fds": 5}, {"rss_bytes": 1, "open_fds": 500}]
        findings = inv.check_health(samples)
        assert any("File descriptors leaked" in f[1] for f in findings)

    def test_too_few_samples_flagged(self):
        findings = inv.check_health([{"rss_bytes": 1, "open_fds": 1}])
        assert any("Too few resource samples" in f[1] for f in findings)

    def test_idle_brain_discipline(self):
        assert inv.check_idle_brain_discipline(0, 100) == []
        findings = inv.check_idle_brain_discipline(3, 100)
        assert any("Brain called on idle" in f[1] for f in findings)


# ---------------------------------------------------------------------
# fakes: title collision, compiler validity
# ---------------------------------------------------------------------

class TestFakes:
    def test_next_unique_title_no_collision(self):
        assert _next_unique_title("TestingSteward", set()) == "TestingSteward"

    def test_next_unique_title_resolves_collision(self):
        used = {"TestingSteward"}
        result = _next_unique_title("TestingSteward", used)
        assert result != "TestingSteward"
        assert result.casefold() not in {u.casefold() for u in used}

    def test_fake_brain_never_repeats_a_title_in_session(self):
        brain = FakeBrainProcess()
        brain.start()
        # Both requests contain "testing", which would collide without guarding.
        r1 = brain.send("User request:\nAGENT_TITLE: make someone to own testing")
        r2 = brain.send("User request:\nAGENT_TITLE: make another, not for testing")
        t1 = r1.split("\n")[0].split(":", 1)[1].strip()
        t2 = r2.split("\n")[0].split(":", 1)[1].strip()
        assert t1 != t2

    def test_fake_direct_answer_never_claims_mutation(self):
        brain = FakeBrainProcess()
        brain.start()
        resp = brain.send("User request:\nwhat do you think about the architecture?")
        assert not response_claims_mutation({"response_text": resp})

    def test_fake_compiler_produces_schema_valid_spec(self, tmp_path):
        from mr1.scheduler import validate_spec

        repo = build_fixture_repo(tmp_path / "fixture")
        compiler = make_fake_compiler(repo)
        raw = compiler("system", "User request:\nrun the tests and report back")
        spec = json.loads(raw)
        validate_spec(spec)  # raises on invalid

    def test_fake_compiler_read_only_variant_also_valid(self, tmp_path):
        from mr1.scheduler import validate_spec

        repo = build_fixture_repo(tmp_path / "fixture")
        compiler = make_fake_compiler(repo)
        raw = compiler("system", "User request:\nsummarize the readme for me")
        spec = json.loads(raw)
        validate_spec(spec)


# ---------------------------------------------------------------------
# fixture repo
# ---------------------------------------------------------------------

class TestFixtureRepo:
    def test_builds_expected_files(self, tmp_path):
        repo = build_fixture_repo(tmp_path / "fixture", git_init=False)
        assert (repo / "fragilekit" / "store.py").exists()
        assert (repo / "fragilekit" / "cache.py").exists()
        assert (repo / "tests" / "test_cache.py").exists()

    def test_idempotent_rebuild(self, tmp_path):
        dest = tmp_path / "fixture"
        build_fixture_repo(dest, git_init=False)
        build_fixture_repo(dest, git_init=False)  # must not raise
        assert (dest / "README.md").exists()


# ---------------------------------------------------------------------
# end-to-end: transcript progression, resume, report generation
# ---------------------------------------------------------------------

@pytest.mark.slow
class TestSoakEndToEnd:
    def test_short_fake_run_progresses_and_reports(self, tmp_path):
        layout = RunLayout(tmp_path / "run")
        config = SoakConfig(planner="fake", do_restart=False, drain_ticks=1)
        result = run_soak(layout, config, max_turns=6, progress=lambda _m: None)
        assert result["counts"]["turns"] == 6
        assert layout.transcript.exists()
        assert layout.samples.exists()
        assert layout.result.exists()

        transcript_lines = layout.transcript.read_text(encoding="utf-8").splitlines()
        assert len(transcript_lines) == 6
        # each transcript line advances the turn index monotonically
        indices = [json.loads(line)["index"] for line in transcript_lines]
        assert indices == sorted(indices)

        report_text = render_report(layout, result)
        assert "Hierarchical Autonomy Soak" in report_text
        assert "Would Marwan have considered" in report_text

    def test_resume_continues_rather_than_restarting(self, tmp_path):
        layout = RunLayout(tmp_path / "run")
        config = SoakConfig(planner="fake", do_restart=False, drain_ticks=1)
        run_soak(layout, config, max_turns=4, progress=lambda _m: None)
        first_len = len(layout.transcript.read_text(encoding="utf-8").splitlines())
        assert first_len == 4

        run_soak(layout, config, resume=True, max_turns=8, progress=lambda _m: None)
        second_len = len(layout.transcript.read_text(encoding="utf-8").splitlines())
        assert second_len == 8

        transcript_lines = layout.transcript.read_text(encoding="utf-8").splitlines()
        indices = [json.loads(line)["index"] for line in transcript_lines]
        assert indices == list(range(8))  # no re-run of already-completed turns

    def test_report_only_mode_reconstructs_from_partial_run(self, tmp_path):
        layout = RunLayout(tmp_path / "run")
        config = SoakConfig(planner="fake", do_restart=False, drain_ticks=1)
        run_soak(layout, config, max_turns=3, progress=lambda _m: None)
        # Simulate an interrupted run: drop the final result.json.
        layout.result.unlink()

        result = analyze(layout)
        assert result["partial"] is True
        assert result["counts"]["turns"] == 3

    def test_mid_conversation_restart_preserves_hierarchy(self, tmp_path):
        layout = RunLayout(tmp_path / "run")
        config = SoakConfig(planner="fake", do_restart=True, drain_ticks=1)
        # Enough turns to pass the persistent_ownership arc (creates an agent)
        # and reach the restart point after "collaboration".
        result = run_soak(layout, config, progress=lambda _m: None)
        assert result["restarted"] is True
        # The agent created before restart must still be visible after it.
        titles = {a["title"] for a in result["agents"]}
        assert "TestingSteward" in titles

    def test_no_high_severity_findings_on_a_full_fake_run(self, tmp_path):
        layout = RunLayout(tmp_path / "run")
        config = SoakConfig(planner="fake", do_restart=True, drain_ticks=4)
        result = run_soak(layout, config, progress=lambda _m: None)
        highs = [
            f for group in result["findings"].values() for f in group if f and f[0] == "high"
        ]
        assert highs == [], highs
        assert result["passed"] is True


class TestFakeWorkerSpawnPlumbing:
    """A bounded-investigation turn under `--planner fake` must be exercised
    deterministically, without ever spawning a real `claude` subprocess
    (HARNESS ROBUSTNESS requirement) — `fake_worker_run` patches
    `mr1.worker.run` the same way `fake_mrn_reasoner` patches the MRn
    reasoner."""

    def test_bounded_investigation_spawns_a_fake_worker_deterministically(self, tmp_path):
        fixture = tmp_path / "fixture"
        build_fixture_repo(fixture)
        paths = _make_paths(tmp_path / "runtime")
        session = HierarchicalSession(paths, planner="fake", fixture_repo=fixture)
        try:
            payload = session.handle_turn(
                "Take a look through this repo.", index=0, drain_ticks=1,
            )
        finally:
            session.shutdown()

        assert payload["ok"] is True
        assert classify(payload) == WORKER_SPAWN
        spawn_decisions = [
            d for d in payload.get("decisions") or [] if d.get("action") == "spawn_worker"
        ]
        assert len(spawn_decisions) == 1
        # never persisted as an AgentRecord
        assert payload["agents"]["created"] == []
