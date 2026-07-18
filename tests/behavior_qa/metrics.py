"""
Ontology-aware rollup metrics for Phase D.

`tests.behavior_qa.runner._rollup` already produces the original
partner-score / trust / surprise rollups unchanged. This module adds the
metrics the ontology-aware redesign requires: worker utilization, context
isolation, orchestrator ownership judgment, initiative calibration split by
action class, and the digital-twin composite. Every formula states its
numerator/denominator (or exact composition) in its own docstring — nothing
here is an undefined judge-generated guess.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from tests.behavior_qa.corpus import EpisodicCluster
from tests.soak.hierarchical.outcomes import (
    ERROR,
    OBJECTIVE_CREATED,
    OBJECTIVE_UPDATED,
    ORCHESTRATOR_CREATED,
    ORCHESTRATOR_REUSED,
    WORKER_SPAWN,
    WORKFLOW,
)

_YES_MAYBE_NO_SCORE = {"YES": 1.0, "MAYBE": 0.5, "NO": 0.0}

_ACTION_CLASS_FOR_OUTCOME = {
    WORKER_SPAWN: "worker",
    WORKFLOW: "workflow",
    ORCHESTRATOR_CREATED: "orchestrator",
    ORCHESTRATOR_REUSED: "orchestrator",
    OBJECTIVE_CREATED: "objective",
    OBJECTIVE_UPDATED: "objective",
}


def _cluster_by_name(clusters: List[EpisodicCluster]) -> Dict[str, EpisodicCluster]:
    return {c.name: c for c in clusters}


def _bounded_investigation_turn_indices(cluster: EpisodicCluster) -> List[int]:
    """Indices of turns whose hard contract allows WORKER_SPAWN — the
    bounded-investigation-shaped turns `worker_utilization` is defined over.

    Restricted to Category A: Category B turns carry the fully-permissive
    `allow=ALL_OUTCOMES` contract (judge-only evaluation, deliberately no
    single right answer), which trivially "allows" WORKER_SPAWN the same
    way it allows everything else — that is not evidence the turn is
    investigation-shaped.
    """
    if cluster.category != "A":
        return []
    return [i for i, ct in enumerate(cluster.turns) if WORKER_SPAWN in ct.turn.allow]


def worker_utilization(
    clusters: List[EpisodicCluster], results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Among bounded-investigation-shaped turns (the turn's hard contract
    allows WORKER_SPAWN), what fraction resulted in WORKER_SPAWN or WORKFLOW
    — a proportionate, delegated response — rather than an ungrounded
    direct answer, an irrelevant refusal, or excessive clarification?

    numerator   = such turns whose observed outcome is WORKER_SPAWN or WORKFLOW
    denominator = such turns, excluding ERROR outcomes (infra/exception noise)
    """
    by_name = _cluster_by_name(clusters)
    numerator = 0
    denominator = 0
    for r in results:
        cluster = by_name.get(r.get("name"))
        if cluster is None:
            continue
        indices = set(_bounded_investigation_turn_indices(cluster))
        if not indices:
            continue
        for turn in r.get("turns") or []:
            if turn.get("index") not in indices:
                continue
            outcome = turn.get("outcome")
            if outcome == ERROR:
                continue
            denominator += 1
            if outcome in (WORKER_SPAWN, WORKFLOW):
                numerator += 1
    if denominator == 0:
        return None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "score": round(numerator / denominator, 2),
    }


def context_isolation(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Among WORKER_SPAWN turns that have a following turn in the same
    cluster, what fraction of those follow-ups read as grounded recall
    (didn't themselves re-trigger a fresh worker_spawn) with a non-empty
    answer, rather than MR1 redoing the investigation inline in its own
    context? This is a cheap, deterministic structural proxy — a true
    semantic grounding check is the judge's `context_isolation` dimension.

    numerator   = follow-up turns with outcome != WORKER_SPAWN and non-empty response_text
    denominator = WORKER_SPAWN turns that have a following turn in the cluster
    """
    numerator = 0
    denominator = 0
    for r in results:
        turns = r.get("turns") or []
        for i, turn in enumerate(turns):
            if turn.get("outcome") != WORKER_SPAWN:
                continue
            if i + 1 >= len(turns):
                continue
            denominator += 1
            follow_up = turns[i + 1]
            if (
                follow_up.get("outcome") != WORKER_SPAWN
                and str(follow_up.get("response_text") or "").strip()
            ):
                numerator += 1
    if denominator == 0:
        return None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "score": round(numerator / denominator, 2),
    }


def orchestrator_creation_score(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """For clusters containing at least one ORCHESTRATOR_CREATED turn,
    average the judge's `ownership_judgment` + `orchestrator_reuse`
    dimensions (each normalized 1-5 -> 0-1), then average across clusters."""
    values = []
    for r in results:
        turns = r.get("turns") or []
        if not any(t.get("outcome") == ORCHESTRATOR_CREATED for t in turns):
            continue
        dims = (r.get("judge") or {}).get("dimensions") or {}
        sub = [
            (v - 1) / 4
            for v in (dims.get("ownership_judgment"), dims.get("orchestrator_reuse"))
            if isinstance(v, (int, float))
        ]
        if sub:
            values.append(sum(sub) / len(sub))
    if not values:
        return None
    return {"clusters": len(values), "score": round(sum(values) / len(values), 2)}


def ownership_judgment_score(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Combines four 0-1 sub-scores (unweighted mean of whichever have at
    least one applicable cluster):

      creation_quality    — `orchestrator_creation_score` above
      reuse_quality       — fraction of ORCHESTRATOR_REUSED-outcome clusters
                            judged `marwan_approval == YES`
      duplicate_avoidance — fraction of ownership-shaped clusters (any
                            ORCHESTRATOR_CREATED/ORCHESTRATOR_REUSED turn)
                            with no "Reuse expected but a new agent was
                            created" structural finding
      callback_fidelity   — for callback clusters (`resumes_from` set),
                            fraction judged
                            `would_marwan_have_done_this_himself == YES`
    """
    creation = orchestrator_creation_score(results)

    reuse_values: List[float] = []
    duplicate_ok: List[float] = []
    callback_values: List[float] = []
    for r in results:
        turns = r.get("turns") or []
        outcomes = [t.get("outcome") for t in turns]
        judge = r.get("judge") or {}
        if ORCHESTRATOR_REUSED in outcomes:
            reuse_values.append(1.0 if judge.get("marwan_approval") == "YES" else 0.0)
        if ORCHESTRATOR_CREATED in outcomes or ORCHESTRATOR_REUSED in outcomes:
            has_dup_finding = any(
                len(f) > 2 and f[2] == "Reuse expected but a new agent was created"
                for t in turns
                for f in (t.get("findings") or [])
            )
            duplicate_ok.append(0.0 if has_dup_finding else 1.0)
        if r.get("resumes_from"):
            callback_values.append(
                1.0 if judge.get("would_marwan_have_done_this_himself") == "YES" else 0.0
            )

    detail: Dict[str, Any] = {}
    parts: List[float] = []
    if creation is not None:
        detail["creation_quality"] = creation["score"]
        parts.append(creation["score"])
    if reuse_values:
        detail["reuse_quality"] = round(sum(reuse_values) / len(reuse_values), 2)
        parts.append(detail["reuse_quality"])
    if duplicate_ok:
        detail["duplicate_avoidance"] = round(sum(duplicate_ok) / len(duplicate_ok), 2)
        parts.append(detail["duplicate_avoidance"])
    if callback_values:
        detail["callback_fidelity"] = round(sum(callback_values) / len(callback_values), 2)
        parts.append(detail["callback_fidelity"])
    if not parts:
        return None
    detail["score"] = round(sum(parts) / len(parts), 2)
    return detail


def _dominant_action_class(turns: List[Dict[str, Any]]) -> Optional[str]:
    classes = [
        _ACTION_CLASS_FOR_OUTCOME[t["outcome"]]
        for t in turns
        if t.get("outcome") in _ACTION_CLASS_FOR_OUTCOME
    ]
    if not classes:
        return None
    return Counter(classes).most_common(1)[0][0]


def initiative_calibration_by_action_class(
    results: List[Dict[str, Any]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Same `(too_aggressive - too_passive) / n` formula as the overall
    `initiative_calibration`, computed separately per action class — the
    class is a cluster's *dominant* outcome (the most common of
    worker/workflow/orchestrator/objective across its turns). A cluster
    with none of those outcomes (pure discussion, clarification, refusal)
    contributes to no bucket, so aggressive orchestrator creation can never
    hide behind conservative worker behavior in the overall number."""
    buckets: Dict[str, List[str]] = {
        "worker": [], "workflow": [], "orchestrator": [], "objective": [],
    }
    for r in results:
        cls = _dominant_action_class(r.get("turns") or [])
        if cls is None:
            continue
        behavior = (r.get("judge") or {}).get("behavior")
        if behavior:
            buckets[cls].append(behavior)

    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for cls, behaviors in buckets.items():
        n = len(behaviors)
        if n == 0:
            out[cls] = None
            continue
        too_aggressive = behaviors.count("TOO_AGGRESSIVE")
        too_passive = behaviors.count("TOO_PASSIVE")
        out[cls] = {
            "clusters": n,
            "calibration": round((too_aggressive - too_passive) / n, 2),
        }
    return out


def digital_twin_score(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Documented composite (not an undefined judge-generated guess): for
    each judged cluster, average whichever of these four normalized signals
    are present —

      would_marwan_have_done_this_himself   YES=1, MAYBE=0.5, NO=0
      would_continue_using                  YES=1, MAYBE=0.5, NO=0
      structural_judgment dimension         (value-1)/4
      naturalness dimension                 (value-1)/4

    — then average that per-cluster composite across all judged clusters.
    """
    per_cluster: List[float] = []
    for r in results:
        judge = r.get("judge") or {}
        dims = judge.get("dimensions") or {}
        parts: List[float] = []
        for field in ("would_marwan_have_done_this_himself", "would_continue_using"):
            v = judge.get(field)
            if v in _YES_MAYBE_NO_SCORE:
                parts.append(_YES_MAYBE_NO_SCORE[v])
        for key in ("structural_judgment", "naturalness"):
            v = dims.get(key)
            if isinstance(v, (int, float)):
                parts.append((v - 1) / 4)
        if parts:
            per_cluster.append(sum(parts) / len(parts))
    if not per_cluster:
        return None
    return {
        "clusters": len(per_cluster),
        "score": round(sum(per_cluster) / len(per_cluster), 2),
        "formula": (
            "mean over judged clusters of: mean("
            "would_marwan_have_done_this_himself[YES=1,MAYBE=.5,NO=0], "
            "would_continue_using[YES=1,MAYBE=.5,NO=0], "
            "(structural_judgment-1)/4, (naturalness-1)/4)"
        ),
    }


def compute_metrics(
    clusters: List[EpisodicCluster], judged_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """All new P1 metrics, assembled once for the report and for tests."""
    return {
        "worker_utilization": worker_utilization(clusters, judged_results),
        "context_isolation": context_isolation(judged_results),
        "orchestrator_creation": orchestrator_creation_score(judged_results),
        "ownership_judgment": ownership_judgment_score(judged_results),
        "initiative_calibration_by_action_class": initiative_calibration_by_action_class(
            judged_results
        ),
        "digital_twin": digital_twin_score(judged_results),
    }
