"""
Unit tests for the Phase D partner-behavior QA harness itself — corpus
sanity, judge prompt construction, judge output parsing/degradation, wave
ordering for callback clusters, and report rendering. Same spirit as
`tests/soak/test_hierarchical.py`: these prove the scaffolding that judges a
run is correct before it's trusted on a real, token-spending run.
"""

from __future__ import annotations

import json

import pytest

from tests.behavior_qa import judge as judge_mod
from tests.behavior_qa.corpus import (
    CLUSTERS,
    ConversationTurn,
    EpisodicCluster,
    validate_corpus,
)
from tests.behavior_qa.driver import RunLayout, TurnRecord
from tests.behavior_qa.judge import (
    DIMENSION_KEYS,
    FakeJudge,
    JudgeVerdict,
    build_judge_prompt,
    judge_cluster,
    load_marwan_preferences,
    parse_judge_output,
)
from tests.behavior_qa.metrics import (
    compute_metrics,
    context_isolation,
    digital_twin_score,
    initiative_calibration_by_action_class,
    ownership_judgment_score,
    orchestrator_creation_score,
    worker_utilization,
)
from tests.behavior_qa.report import render_report
from tests.behavior_qa.runner import BehaviorQAConfig, _execution_waves, run_corpus
from tests.soak.hierarchical.outcomes import (
    ALL_OUTCOMES,
    CLARIFY,
    DIRECT,
    ORCHESTRATOR_CREATED,
    ORCHESTRATOR_REUSED,
    WORKER_SPAWN,
    WORKFLOW,
    Turn,
)


# ---------------------------------------------------------------------
# corpus sanity
# ---------------------------------------------------------------------

class TestCorpusSanity:
    def test_real_corpus_validates(self):
        validate_corpus()  # must not raise

    def test_no_duplicate_cluster_names(self):
        names = [c.name for c in CLUSTERS]
        assert len(names) == len(set(names))

    def test_category_b_turns_are_permissive_and_carry_soft_expectations(self):
        for c in CLUSTERS:
            if c.category != "B":
                continue
            for ct in c.turns:
                assert set(ct.turn.allow) == set(ALL_OUTCOMES)
                assert ct.soft_expectations, f"{c.name} turn {ct.text!r} has no soft expectations"

    def test_every_resumes_from_points_at_a_real_cluster(self):
        names = {c.name for c in CLUSTERS}
        for c in CLUSTERS:
            if c.resumes_from is not None:
                assert c.resumes_from in names

    def test_detects_duplicate_name(self):
        a = EpisodicCluster(name="dup", category="A", topic="t", goal="g",
                             turns=[ConversationTurn(Turn(text="hi", allow=(DIRECT,)))])
        b = EpisodicCluster(name="dup", category="A", topic="t", goal="g",
                             turns=[ConversationTurn(Turn(text="hi", allow=(DIRECT,)))])
        with pytest.raises(ValueError, match="duplicate cluster name"):
            validate_corpus([a, b])

    def test_detects_dangling_resumes_from(self):
        a = EpisodicCluster(name="a", category="A", topic="t", goal="g",
                             turns=[ConversationTurn(Turn(text="hi", allow=(DIRECT,)))],
                             resumes_from="ghost")
        with pytest.raises(ValueError, match="unknown cluster"):
            validate_corpus([a])

    def test_detects_resumes_from_cycle(self):
        a = EpisodicCluster(name="a", category="A", topic="t", goal="g",
                             turns=[ConversationTurn(Turn(text="hi", allow=(DIRECT,)))],
                             resumes_from="b")
        b = EpisodicCluster(name="b", category="A", topic="t", goal="g",
                             turns=[ConversationTurn(Turn(text="hi", allow=(DIRECT,)))],
                             resumes_from="a")
        with pytest.raises(ValueError, match="cycle"):
            validate_corpus([a, b])

    def test_category_b_turn_without_soft_expectation_is_rejected(self):
        c = EpisodicCluster(
            name="bad", category="B", topic="t", goal="g",
            turns=[ConversationTurn(Turn(text="hi", allow=tuple(ALL_OUTCOMES)))],
        )
        with pytest.raises(ValueError, match="no soft expectations"):
            validate_corpus([c])


# ---------------------------------------------------------------------
# judge prompt construction
# ---------------------------------------------------------------------

def _turn_record(**overrides) -> TurnRecord:
    base = dict(
        index=0, text="hello", note="", soft_expectations=[],
        response_text="hi there", route="direct_response",
        route_advice={"route": "direct_response", "confidence": 0.9, "reason": "greeting", "signals": {}},
        override_reason="", final_action="direct_response",
        created_agents=[], created_workflows=[], created_messages=[],
        approval_ids=[], ok=True, errors=[], outcome="direct_response", findings=[],
    )
    base.update(overrides)
    return TurnRecord(**base)


class TestJudgePrompt:
    def test_prompt_contains_transcript_and_soft_expectations(self):
        cluster = EpisodicCluster(
            name="c1", category="B", topic="topic-x", goal="goal-x",
            turns=[ConversationTurn(Turn(text="t", allow=tuple(ALL_OUTCOMES)), soft_expectations=("expect-y",))],
        )
        records = [_turn_record(text="I keep thinking about this.",
                                 soft_expectations=["expect-y"])]
        prompt = build_judge_prompt(cluster, records, ["always discuss first"])
        assert "c1" in prompt
        assert "topic-x" in prompt
        assert "goal-x" in prompt
        assert "I keep thinking about this." in prompt
        assert "expect-y" in prompt
        assert "always discuss first" in prompt
        for key in DIMENSION_KEYS:
            assert key in prompt
        assert "partner_score" in prompt
        assert "would_continue_using" in prompt
        assert "would_marwan_have_done_this_himself" in prompt
        assert "surprised_by_action" in prompt

    def test_prompt_notes_a_callback_cluster(self):
        cluster = EpisodicCluster(
            name="c2", category="B", topic="t", goal="g",
            turns=[ConversationTurn(Turn(text="t", allow=tuple(ALL_OUTCOMES)), soft_expectations=("x",))],
            resumes_from="c1",
        )
        prompt = build_judge_prompt(cluster, [_turn_record()], [])
        assert "c1" in prompt
        assert "continues" in prompt


class TestJudgeExplicitRequestCalibration:
    """Phase D.3 follow-up: the real bare_bounded_inspection/own_runtime_testing
    smoke runs were judged TOO_AGGRESSIVE purely for lacking upfront
    discussion — even though both were explicit, bounded requests where the
    desired philosophy says immediate action is correct. Root cause: a
    replayed 'always discuss before delegating' preference memory (predating
    the worker/orchestrator ontology split) was left free to override the
    asymmetric guidance. These tests pin the calibration language that fixes
    this, so it can't silently regress."""

    def test_system_prompt_states_explicit_bounded_requests_need_no_discussion(self):
        prompt = judge_mod.JUDGE_SYSTEM_PROMPT
        assert "I'll have a worker make a bounded pass and report back." in prompt
        assert "I need your permission before I do that." in prompt
        assert "explicit-request calibration" in prompt.lower()

    def test_system_prompt_tells_judge_not_to_let_preferences_override_calibration(self):
        normalized = " ".join(judge_mod.JUDGE_SYSTEM_PROMPT.split())
        assert "does not turn every explicit, bounded worker spawn" in normalized

    def test_structural_judgment_dimension_excludes_discussion_timing(self):
        normalized = " ".join(judge_mod.JUDGE_SYSTEM_PROMPT.split())
        assert "independent of whether a discussion" in normalized

    def test_replayed_preference_gets_an_adjacent_precedence_reminder(self):
        cluster = EpisodicCluster(
            name="c", category="A", topic="t", goal="g",
            turns=[ConversationTurn(Turn(text="t", allow=(DIRECT, CLARIFY)))],
        )
        prompt = build_judge_prompt(
            cluster, [_turn_record()],
            ["Always talk through the problem with Marwan before delegating to Kamis or Kazis."],
        )
        assert "Always talk through the problem" in prompt
        assert "explicit bounded worker spawns or" in prompt


# ---------------------------------------------------------------------
# preference replay
# ---------------------------------------------------------------------

class TestPreferenceReplay:
    def test_missing_memory_dir_returns_empty(self, tmp_path):
        assert load_marwan_preferences(tmp_path / "does-not-exist") == []

    def test_only_feedback_type_is_replayed(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text(
            "- [Feedback one](fb.md) — a behavioral note\n"
            "- [User fact](user.md) — a fact about the user\n"
            "- [Project note](proj.md) — project context\n",
            encoding="utf-8",
        )
        (mem_dir / "fb.md").write_text(
            "---\nname: fb\ndescription: d\nmetadata:\n  type: feedback\n---\n\n"
            "Always discuss before delegating.\n",
            encoding="utf-8",
        )
        (mem_dir / "user.md").write_text(
            "---\nname: user\ndescription: d\nmetadata:\n  type: user\n---\n\n"
            "Marwan is a friend.\n",
            encoding="utf-8",
        )
        (mem_dir / "proj.md").write_text(
            "---\nname: proj\ndescription: d\nmetadata:\n  type: project\n---\n\n"
            "Some project fact.\n",
            encoding="utf-8",
        )
        prefs = load_marwan_preferences(mem_dir)
        assert len(prefs) == 1
        assert "Always discuss before delegating." in prefs[0]

    def test_flat_type_field_is_also_accepted(self, tmp_path):
        # Some real memory files carry `type:` flat rather than nested under
        # `metadata:` — both shapes must work.
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("- [Feedback](fb.md) — note\n", encoding="utf-8")
        (mem_dir / "fb.md").write_text(
            "---\nname: fb\ndescription: d\ntype: feedback\n---\n\nBehave this way.\n",
            encoding="utf-8",
        )
        prefs = load_marwan_preferences(mem_dir)
        assert prefs == ["Behave this way."]

    def test_refuses_to_follow_link_outside_memory_dir(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text(
            "---\nname: x\ndescription: d\nmetadata:\n  type: feedback\n---\n\nleaked\n",
            encoding="utf-8",
        )
        (mem_dir / "MEMORY.md").write_text("- [Leak](../outside.md) — nope\n", encoding="utf-8")
        assert load_marwan_preferences(mem_dir) == []


# ---------------------------------------------------------------------
# judge output parsing
# ---------------------------------------------------------------------

_VALID_VERDICT = {
    "dimensions": {k: 3 for k in DIMENSION_KEYS},
    "behavior": "BALANCED",
    "should_have": ["discussed_only"],
    "marwan_approval": "YES",
    "would_continue_using": "YES",
    "would_marwan_have_done_this_himself": "MAYBE",
    "surprised_by_action": False,
    "partner_score": 7,
    "comments": {"intelligent": "x", "annoying": "", "unnatural": ""},
}


class TestParseJudgeOutput:
    def test_valid_json(self):
        v = parse_judge_output(json.dumps(_VALID_VERDICT))
        assert v.behavior == "BALANCED"
        assert v.partner_score == 7
        assert v.judge_error is None

    def test_json_wrapped_in_prose(self):
        text = "Sure, here you go:\n" + json.dumps(_VALID_VERDICT) + "\nHope that helps!"
        v = parse_judge_output(text)
        assert v.behavior == "BALANCED"

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_judge_output("not json at all")

    def test_invalid_behavior_raises(self):
        bad = dict(_VALID_VERDICT, behavior="SOMETIMES")
        with pytest.raises(ValueError):
            parse_judge_output(json.dumps(bad))

    def test_out_of_range_dimension_raises(self):
        bad = dict(_VALID_VERDICT, dimensions=dict(_VALID_VERDICT["dimensions"], initiative=9))
        with pytest.raises(ValueError):
            parse_judge_output(json.dumps(bad))

    def test_invalid_should_have_raises(self):
        bad = dict(_VALID_VERDICT, should_have=["not_a_real_option"])
        with pytest.raises(ValueError):
            parse_judge_output(json.dumps(bad))


class TestJudgeClusterDegradation:
    def test_degrades_on_call_failure(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise ValueError("subprocess exploded")

        monkeypatch.setattr(judge_mod, "call_judge", _boom)
        cluster = EpisodicCluster(
            name="c", category="B", topic="t", goal="g",
            turns=[ConversationTurn(Turn(text="t", allow=tuple(ALL_OUTCOMES)), soft_expectations=("x",))],
        )
        verdict = judge_cluster(cluster, [_turn_record()], model="sonnet", preferences=[])
        assert verdict.judge_error is not None
        assert "subprocess exploded" in verdict.judge_error
        assert verdict.partner_score is None
        assert all(v is None for v in verdict.dimensions.values())

    def test_degrades_on_bad_json(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "call_judge", lambda *a, **k: "not json")
        cluster = EpisodicCluster(
            name="c", category="A", topic="t", goal="g",
            turns=[ConversationTurn(Turn(text="t", allow=(DIRECT, CLARIFY)))],
        )
        verdict = judge_cluster(cluster, [_turn_record()])
        assert verdict.judge_error is not None

    def test_fake_judge_never_errors(self):
        cluster = EpisodicCluster(
            name="c", category="A", topic="t", goal="g",
            turns=[ConversationTurn(Turn(text="t", allow=(DIRECT, CLARIFY)))],
        )
        verdict = FakeJudge()(cluster, [_turn_record()])
        assert verdict.judge_error is None
        assert verdict.behavior == "BALANCED"


# ---------------------------------------------------------------------
# ontology-aware metrics
# ---------------------------------------------------------------------

def _turn(index, outcome, response_text="some text", **overrides):
    base = dict(
        index=index, text="t", note="", soft_expectations=[],
        response_text=response_text, route="", route_advice={}, override_reason="",
        final_action=None, created_agents=[], created_workflows=[], created_messages=[],
        approval_ids=[], ok=True, errors=[], outcome=outcome, findings=[], worker_spawns=[],
    )
    base.update(overrides)
    return base


def _result(name, category, turns, judge=None, resumes_from=None):
    return {
        "name": name, "category": category, "topic": "t", "goal": "g",
        "resumes_from": resumes_from, "turns": turns,
        "judge": judge or {"dimensions": {}, "behavior": "BALANCED"},
    }


class TestMetrics:
    def test_worker_utilization_counts_only_category_a_bounded_turns(self):
        by_name = {c.name: c for c in CLUSTERS}
        results = [
            _result("investigate_repo", "A", [_turn(0, WORKER_SPAWN)]),
            # Category B turns nominally "allow" everything (including
            # WORKER_SPAWN) but must not count as bounded-investigation-shaped.
            _result("drone_backend", "B", [_turn(0, WORKER_SPAWN)]),
        ]
        score = worker_utilization(list(by_name.values()), results)
        assert score == {"numerator": 1, "denominator": 1, "score": 1.0}

    def test_worker_utilization_counts_workflow_as_good_too(self):
        by_name = {c.name: c for c in CLUSTERS}
        results = [_result("investigate_flaky_tests", "A", [_turn(0, WORKFLOW)])]
        score = worker_utilization(list(by_name.values()), results)
        assert score == {"numerator": 1, "denominator": 1, "score": 1.0}

    def test_worker_utilization_direct_response_counts_against(self):
        by_name = {c.name: c for c in CLUSTERS}
        results = [_result("bare_bounded_inspection", "A", [_turn(0, DIRECT)])]
        score = worker_utilization(list(by_name.values()), results)
        assert score == {"numerator": 0, "denominator": 1, "score": 0.0}

    def test_worker_utilization_none_when_no_bounded_turns_present(self):
        by_name = {c.name: c for c in CLUSTERS}
        results = [_result("fashion_project", "B", [_turn(0, DIRECT)])]
        assert worker_utilization(list(by_name.values()), results) is None

    def test_context_isolation_grounded_followup(self):
        results = [_result("c", "A", [
            _turn(0, WORKER_SPAWN),
            _turn(1, DIRECT, response_text="here is what the worker found"),
        ])]
        assert context_isolation(results) == {"numerator": 1, "denominator": 1, "score": 1.0}

    def test_context_isolation_followup_that_respawns_a_worker_is_not_isolated(self):
        results = [_result("c", "A", [
            _turn(0, WORKER_SPAWN),
            _turn(1, WORKER_SPAWN, response_text="spawned again"),
        ])]
        assert context_isolation(results) == {"numerator": 0, "denominator": 1, "score": 0.0}

    def test_context_isolation_no_followup_turn_is_excluded(self):
        results = [_result("c", "A", [_turn(0, WORKER_SPAWN)])]
        assert context_isolation(results) is None

    def test_orchestrator_creation_score_normalizes_dimensions(self):
        results = [_result(
            "c", "A", [_turn(0, ORCHESTRATOR_CREATED)],
            judge={"dimensions": {"ownership_judgment": 5, "orchestrator_reuse": 3}},
        )]
        # (5-1)/4 = 1.0, (3-1)/4 = 0.5 -> mean 0.75
        assert orchestrator_creation_score(results) == {"clusters": 1, "score": 0.75}

    def test_orchestrator_creation_score_none_without_creation(self):
        results = [_result("c", "A", [_turn(0, DIRECT)])]
        assert orchestrator_creation_score(results) is None

    def test_ownership_judgment_score_combines_subscores(self):
        results = [
            _result(
                "own", "A", [_turn(0, ORCHESTRATOR_CREATED)],
                judge={"dimensions": {"ownership_judgment": 5, "orchestrator_reuse": 5}},
            ),
            _result(
                "reuse", "A", [_turn(0, ORCHESTRATOR_REUSED)],
                judge={"marwan_approval": "YES"},
            ),
        ]
        score = ownership_judgment_score(results)
        assert score["creation_quality"] == 1.0
        assert score["reuse_quality"] == 1.0
        assert score["duplicate_avoidance"] == 1.0
        assert score["score"] == 1.0

    def test_ownership_judgment_flags_duplicate_finding(self):
        results = [_result(
            "own", "A", [_turn(0, ORCHESTRATOR_CREATED, findings=[
                ("high", 0, "Reuse expected but a new agent was created", "detail"),
            ])],
            judge={"dimensions": {"ownership_judgment": 3, "orchestrator_reuse": 3}},
        )]
        score = ownership_judgment_score(results)
        assert score["duplicate_avoidance"] == 0.0

    def test_initiative_calibration_by_action_class_splits_correctly(self):
        results = [
            _result("w1", "A", [_turn(0, WORKER_SPAWN)], judge={"behavior": "TOO_AGGRESSIVE"}),
            _result("o1", "A", [_turn(0, ORCHESTRATOR_CREATED)], judge={"behavior": "TOO_PASSIVE"}),
            _result("disc", "B", [_turn(0, DIRECT)], judge={"behavior": "BALANCED"}),
        ]
        by_class = initiative_calibration_by_action_class(results)
        assert by_class["worker"] == {"clusters": 1, "calibration": 1.0}
        assert by_class["orchestrator"] == {"clusters": 1, "calibration": -1.0}
        assert by_class["workflow"] is None
        assert by_class["objective"] is None

    def test_digital_twin_score_formula(self):
        results = [_result(
            "c", "A", [_turn(0, DIRECT)],
            judge={
                "would_marwan_have_done_this_himself": "YES",
                "would_continue_using": "YES",
                "dimensions": {"structural_judgment": 5, "naturalness": 5},
            },
        )]
        # all four normalized components are 1.0 -> mean 1.0
        result = digital_twin_score(results)
        assert result["score"] == 1.0
        assert "formula" in result

    def test_compute_metrics_returns_all_expected_keys(self):
        by_name = {c.name: c for c in CLUSTERS}
        results = [_result("investigate_repo", "A", [_turn(0, WORKER_SPAWN)],
                            judge={"behavior": "BALANCED"})]
        metrics = compute_metrics(list(by_name.values()), results)
        assert set(metrics) == {
            "worker_utilization", "context_isolation", "orchestrator_creation",
            "ownership_judgment", "initiative_calibration_by_action_class",
            "digital_twin",
        }


# ---------------------------------------------------------------------
# execution waves
# ---------------------------------------------------------------------

class TestExecutionWaves:
    def _cluster(self, name, resumes_from=None):
        return EpisodicCluster(
            name=name, category="A", topic="t", goal="g",
            turns=[ConversationTurn(Turn(text="t", allow=(DIRECT, CLARIFY)))],
            resumes_from=resumes_from,
        )

    def test_independent_clusters_share_wave_zero(self):
        clusters = [self._cluster("a"), self._cluster("b")]
        waves = _execution_waves(clusters)
        assert len(waves) == 1
        assert {c.name for c in waves[0]} == {"a", "b"}

    def test_callback_deferred_to_next_wave(self):
        clusters = [self._cluster("a"), self._cluster("a_followup", resumes_from="a")]
        waves = _execution_waves(clusters)
        assert [c.name for c in waves[0]] == ["a"]
        assert [c.name for c in waves[1]] == ["a_followup"]

    def test_chain_of_three_orders_strictly(self):
        clusters = [
            self._cluster("c", resumes_from="b"),
            self._cluster("a"),
            self._cluster("b", resumes_from="a"),
        ]
        waves = _execution_waves(clusters)
        assert [c.name for c in waves[0]] == ["a"]
        assert [c.name for c in waves[1]] == ["b"]
        assert [c.name for c in waves[2]] == ["c"]


# ---------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------

class TestReportRendering:
    def _fake_result(self) -> dict:
        judge = dict(_VALID_VERDICT, judge_error=None)
        cluster = {
            "name": "drone_backend", "category": "B", "topic": "drone backend",
            "goal": "test goal", "resumes_from": None,
            "turns": [
                {
                    "index": 0, "text": "I've wanted to build the drone backend for months.",
                    "soft_expectations": ["acknowledge, don't manufacture an owner"],
                    "response_text": "That's a good project to talk through.",
                    "outcome": "direct_response", "findings": [],
                    "created_agents": [], "created_workflows": [],
                    "created_messages": [], "approval_ids": [], "worker_spawns": [],
                },
            ],
            "judge": judge,
        }
        worker_cluster = {
            "name": "bare_bounded_inspection", "category": "A", "topic": "bounded inspection",
            "goal": "test goal", "resumes_from": None,
            "turns": [
                {
                    "index": 0, "text": "Take a look through this repo.",
                    "soft_expectations": [],
                    "response_text": "Delegated a worker to look through the repo.",
                    "outcome": "worker_spawn", "findings": [],
                    "created_agents": [], "created_workflows": [],
                    "created_messages": [], "approval_ids": [],
                    "worker_spawns": [{"task_id": "task-1", "mission": "look through the repo",
                                        "status": "completed"}],
                },
            ],
            "judge": judge,
        }
        return {
            "planner": "real", "judge_model": "sonnet",
            "clusters": [cluster, worker_cluster],
            "counts": {"clusters": 2, "judged_clusters": 2, "turns": 2},
            "metrics": {
                "worker_utilization": {"numerator": 1, "denominator": 1, "score": 1.0},
                "context_isolation": None,
                "orchestrator_creation": None,
                "ownership_judgment": None,
                "initiative_calibration_by_action_class": {
                    "worker": {"clusters": 1, "calibration": 0.0},
                    "workflow": None, "orchestrator": None, "objective": None,
                },
                "digital_twin": {
                    "clusters": 2, "score": 0.8, "formula": "test formula",
                },
            },
            "rollup": {
                "overall": {
                    "judged_clusters": 1, "mean_partner_score": 7.0,
                    "behavior_distribution": {"BALANCED": 1},
                    "approval_distribution": {"YES": 1},
                    "would_continue_using_pct": 1.0, "would_continue_using_no": [],
                    "would_have_done_himself_pct": 0.0,
                    "would_have_done_himself_no": ["drone_backend"],
                    "surprise_rate": 0.0, "surprising_and_approved": 0,
                    "surprising_and_not_approved": 0,
                    "initiative_calibration": 0.2,
                    "dimension_means": {k: 3.0 for k in DIMENSION_KEYS},
                },
                "category_a": {"judged_clusters": 0},
                "category_b": {
                    "judged_clusters": 1, "mean_partner_score": 7.0,
                    "behavior_distribution": {"BALANCED": 1},
                    "approval_distribution": {"YES": 1},
                    "would_continue_using_pct": 1.0, "would_continue_using_no": [],
                    "would_have_done_himself_pct": 0.0, "would_have_done_himself_no": [],
                    "surprise_rate": 0.0, "surprising_and_approved": 0,
                    "surprising_and_not_approved": 0, "initiative_calibration": 0.2,
                    "dimension_means": {k: 3.0 for k in DIMENSION_KEYS},
                },
            },
            "preferences": ["Always discuss before delegating."],
        }

    def test_all_sections_present(self):
        text = render_report(self._fake_result())
        assert "# Phase D — Partner Behavior QA" in text
        assert "## 1. Transcript" in text
        assert "## 2. Actions taken" in text
        assert "## 3. Judge report" in text
        assert "## 4. Examples" in text
        assert "## 5. Recommendations" in text
        assert "## Would Marwan enjoy this?" in text
        assert "initiative calibration" in text.lower()
        assert "would continue using" in text.lower()
        assert "Always discuss before delegating." in text

    def test_worker_and_orchestrator_columns_never_conflated(self):
        text = render_report(self._fake_result())
        assert "## Worker & ownership decisions" in text
        assert "workers spawned" in text
        assert "orchestrators created" in text
        assert "orchestrators reused" in text
        # the worker-spawn cluster shows up in the actions table with a
        # worker count and no orchestrator title
        assert "| bare_bounded_inspection | 1 |" in text

    def test_initiative_by_action_class_section_present(self):
        text = render_report(self._fake_result())
        assert "## Initiative calibration by action class" in text
        assert "| worker | 1 | +0.00 |" in text

    def test_digital_twin_score_in_closing(self):
        text = render_report(self._fake_result())
        assert "Digital twin score:** 0.8" in text
        assert "test formula" in text

    def test_dry_run_banner_shown_for_fake_planner(self):
        result = self._fake_result()
        result["planner"] = "fake"
        text = render_report(result)
        assert "Dry run" in text

    def test_no_dry_run_banner_for_real_planner(self):
        text = render_report(self._fake_result())
        assert "Dry run" not in text


# ---------------------------------------------------------------------
# end-to-end (fake planner + FakeJudge, no real subprocess spent)
# ---------------------------------------------------------------------

@pytest.mark.slow
class TestEndToEnd:
    def test_small_subset_with_a_callback_pair(self, tmp_path):
        by_name = {c.name: c for c in CLUSTERS}
        subset = [
            by_name["own_runtime_testing"],
            by_name["own_runtime_testing_status_check"],
            by_name["drone_backend"],
        ]
        layout = RunLayout(tmp_path / "run")
        config = BehaviorQAConfig(planner="fake", drain_ticks=1, jobs=1)
        result = run_corpus(layout, config, clusters=subset, progress=lambda _m: None)

        assert result["counts"]["clusters"] == 3
        assert result["counts"]["judged_clusters"] == 3
        assert layout.result.exists()
        for name in ("own_runtime_testing", "own_runtime_testing_status_check", "drone_backend"):
            assert layout.cluster_transcript(name).exists()
            assert layout.cluster_judge(name).exists()

        # the callback cluster actually inherited real prior state
        source_runtime = layout.cluster_runtime("own_runtime_testing")
        callback_runtime = layout.cluster_runtime("own_runtime_testing_status_check")
        assert (source_runtime / "agents").exists()
        assert (callback_runtime / "agents").exists()

        report_text = render_report(result)
        assert "Phase D" in report_text

    def test_resume_skips_completed_clusters(self, tmp_path):
        by_name = {c.name: c for c in CLUSTERS}
        subset = [by_name["direct_destructive_clarify"], by_name["fashion_project"]]
        layout = RunLayout(tmp_path / "run")
        config = BehaviorQAConfig(planner="fake", drain_ticks=1, jobs=1)
        run_corpus(layout, config, clusters=subset, progress=lambda _m: None)

        judge_path = layout.cluster_judge("fashion_project")
        first_mtime = judge_path.stat().st_mtime_ns

        result = run_corpus(layout, config, clusters=subset, resume=True, progress=lambda _m: None)
        assert judge_path.stat().st_mtime_ns == first_mtime
        assert result["counts"]["clusters"] == 2
