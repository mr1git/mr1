"""
Orchestrates the Phase D corpus: drives every episodic cluster, judges it (in
real-planner mode), checkpoints as it goes, and assembles the aggregate
rollup the report reads.

Clusters run in dependency "waves" derived from `resumes_from` — a callback
cluster only becomes eligible once its source cluster has finished, since
the callback needs the source's *finished* runtime directory to copy
forward. Independent clusters within a wave can run in parallel
(`ProcessPoolExecutor`, spawn context — the same pattern
`tests/runtime_qa/runner.py` uses for isolated-runtime scenarios).
"""

from __future__ import annotations

import json
import multiprocessing
import re
import shutil
import time
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tests.behavior_qa.corpus import CLUSTERS, EpisodicCluster
from tests.behavior_qa.driver import RunLayout, build_cluster_session, run_cluster_turns
from tests.behavior_qa.judge import (
    DIMENSION_KEYS,
    FakeJudge,
    JudgeVerdict,
    judge_cluster,
    load_marwan_preferences,
)
from tests.behavior_qa.metrics import compute_metrics
from tests.soak.hierarchical.driver import _append_jsonl, _write_json, _read_jsonl
from tests.soak.hierarchical.fixture import build_fixture_repo


@dataclass
class BehaviorQAConfig:
    planner: str = "fake"       # "fake" | "real"
    judge_model: str = "sonnet"
    drain_ticks: int = 4
    jobs: int = 1
    judge_delay_s: float = 0.0  # optional throttle between judge calls


# Signature of an infrastructure/rate-limit failure bleeding into a turn's
# response text (MR1's own `claude` subprocess hit its usage limit mid-run),
# as opposed to a genuine partner-behavior response. `[MR1 ERROR]` is the
# prefix `MR1Process.send()` uses for any process-level failure; the rest
# narrows it to rate-limit/quota signatures specifically, per the Phase D
# incident where the corpus ran out of usage near the callback wave.
_INFRA_ERROR_PATTERN = re.compile(
    r"\[MR1 ERROR\].*?(rate.?limit|usage limit|hit your limit|quota|"
    r"\b429\b|status[_ ]?code[:\s]*429)",
    re.IGNORECASE | re.DOTALL,
)
_INFRA_ERROR_PREFIX = "infrastructure error"


def _turn_infra_error(record: Any) -> Optional[str]:
    text = str(getattr(record, "response_text", "") or "")
    return text.strip() if _INFRA_ERROR_PATTERN.search(text) else None


def _cluster_infra_error(records: List[Any]) -> Optional[str]:
    """None when every turn's response looks like real MR1 behavior. A
    string (used as the judge's `judge_error`) the moment any turn's
    response is itself an infrastructure/rate-limit failure — such a
    transcript must never be scored as a partner-behavior outcome."""
    for record in records:
        err = _turn_infra_error(record)
        if err is not None:
            return (
                f"{_INFRA_ERROR_PREFIX}: turn[{record.index}] MR1's own brain call "
                f"hit an infrastructure/rate-limit failure: {err[:200]!r}"
            )
    return None


def _is_infra_error(judge_error: Optional[str]) -> bool:
    return bool(judge_error) and judge_error.startswith(_INFRA_ERROR_PREFIX)


def _execution_waves(clusters: List[EpisodicCluster]) -> List[List[EpisodicCluster]]:
    """Small topological sort on `resumes_from` edges — the corpus only has
    a handful of dependency edges, not a general DAG scheduler."""
    by_name = {c.name: c for c in clusters}
    wave_of: Dict[str, int] = {}

    def wave(name: str, stack: tuple = ()) -> int:
        if name in wave_of:
            return wave_of[name]
        if name in stack:
            raise ValueError(f"resumes_from cycle involving {name!r}")
        cluster = by_name[name]
        if cluster.resumes_from is None:
            w = 0
        elif cluster.resumes_from not in by_name:
            # source not in this run's subset — treat as already satisfied
            w = 0
        else:
            w = wave(cluster.resumes_from, stack + (name,)) + 1
        wave_of[name] = w
        return w

    for c in clusters:
        wave(c.name)

    buckets: Dict[int, List[EpisodicCluster]] = {}
    for c in clusters:
        buckets.setdefault(wave_of[c.name], []).append(c)
    return [buckets[i] for i in sorted(buckets)]


def run_cluster(
    cluster: EpisodicCluster,
    layout: RunLayout,
    config: BehaviorQAConfig,
    preferences: List[str],
) -> Dict[str, Any]:
    """Run one isolated cluster end to end: build/copy-forward its runtime,
    drive its turns, judge it, persist, and return its result as a plain
    dict (so this crosses a process-pool boundary cleanly)."""
    source_runtime_root = (
        layout.cluster_runtime(cluster.resumes_from) if cluster.resumes_from else None
    )
    session = build_cluster_session(
        layout, cluster, planner=config.planner, source_runtime_root=source_runtime_root,
    )
    try:
        records = run_cluster_turns(session, cluster, drain_ticks=config.drain_ticks)
    finally:
        session.shutdown()

    transcript_path = layout.cluster_transcript(cluster.name)
    if transcript_path.exists():
        transcript_path.unlink()
    for record in records:
        _append_jsonl(transcript_path, record.to_dict())

    infra_error = _cluster_infra_error(records)
    if infra_error is not None:
        # MR1's own brain hit a rate limit mid-transcript — this transcript
        # cannot speak to partner behavior at all. Never spend a judge call
        # on it, and never let it read as a scored (bad) outcome.
        verdict = JudgeVerdict.degraded(infra_error)
    elif config.planner == "real":
        if config.judge_delay_s:
            time.sleep(config.judge_delay_s)
        verdict = judge_cluster(
            cluster, records, model=config.judge_model, preferences=preferences,
        )
    else:
        verdict = FakeJudge()(
            cluster, records, model=config.judge_model, preferences=preferences,
        )
    _write_json(layout.cluster_judge(cluster.name), verdict.to_dict())

    return {
        "name": cluster.name,
        "category": cluster.category,
        "topic": cluster.topic,
        "goal": cluster.goal,
        "resumes_from": cluster.resumes_from,
        "turns": [r.to_dict() for r in records],
        "judge": verdict.to_dict(),
    }


def _run_cluster_by_name(
    name: str, layout: RunLayout, config: BehaviorQAConfig, preferences: List[str],
) -> Dict[str, Any]:
    """Worker entry point for the process pool: re-selects the cluster from
    the canonical corpus by name (mirrors `runtime_qa.runner._run_one_named`)
    rather than pickling cluster objects captured by closure."""
    by_name = {c.name: c for c in CLUSTERS}
    return run_cluster(by_name[name], layout, config, preferences)


def _run_wave(
    pending: List[EpisodicCluster],
    layout: RunLayout,
    config: BehaviorQAConfig,
    preferences: List[str],
    *,
    progress: Callable[[str], None],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if config.jobs <= 1 or len(pending) <= 1:
        for cluster in pending:
            progress(f"cluster: {cluster.name} ({cluster.category}) ...")
            results[cluster.name] = run_cluster(cluster, layout, config, preferences)
            _report_cluster(results[cluster.name], progress)
        return results

    with ProcessPoolExecutor(
        max_workers=config.jobs, mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_run_cluster_by_name, c.name, layout, config, preferences): c
            for c in pending
        }
        for future in as_completed(futures):
            cluster = futures[future]
            results[cluster.name] = future.result()
            _report_cluster(results[cluster.name], progress)
    return results


def _report_cluster(result: Dict[str, Any], progress: Callable[[str], None]) -> None:
    judge = result.get("judge") or {}
    progress(
        f"  done: {result['name']:36s} behavior={judge.get('behavior')}  "
        f"score={judge.get('partner_score')}"
    )


def run_corpus(
    layout: RunLayout,
    config: BehaviorQAConfig,
    *,
    clusters: Optional[List[EpisodicCluster]] = None,
    resume: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    say = progress or (lambda _m: None)
    layout.root.mkdir(parents=True, exist_ok=True)

    all_clusters = list(clusters) if clusters is not None else list(CLUSTERS)

    if not resume:
        if layout.clusters_dir.exists():
            shutil.rmtree(layout.clusters_dir)
        for path in (layout.checkpoint, layout.result):
            if path.exists():
                path.unlink()

    build_fixture_repo(layout.fixture)
    preferences = load_marwan_preferences()

    checkpoint = _load_checkpoint(layout) if resume else {}
    done: set = set(checkpoint.get("done", []))
    by_name = {c.name: c for c in all_clusters}

    results: Dict[str, Dict[str, Any]] = {}
    for name in done:
        cluster = by_name.get(name)
        if cluster is not None:
            results[name] = _reload_cluster_result(layout, cluster)

    for wave in _execution_waves(all_clusters):
        pending = [c for c in wave if c.name not in done]
        if not pending:
            continue
        wave_results = _run_wave(pending, layout, config, preferences, progress=say)
        for name, result in wave_results.items():
            results[name] = result
            # A cluster whose judge call failed (rate limit, timeout, bad
            # JSON — anything caught by `judge_cluster`'s degrade path) did
            # not produce a real verdict. Leave it out of `done` so a
            # subsequent `--resume` retries it instead of treating a
            # transient failure as permanently finished.
            if not (result.get("judge") or {}).get("judge_error"):
                done.add(name)
            _save_checkpoint(layout, {"done": sorted(done)})

    result = _assemble_result(all_clusters, results, config)
    result["preferences"] = preferences
    _write_json(layout.result, result)
    return result


def _reload_cluster_result(layout: RunLayout, cluster: EpisodicCluster) -> Dict[str, Any]:
    turns = _read_jsonl(layout.cluster_transcript(cluster.name))
    judge_path = layout.cluster_judge(cluster.name)
    judge = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {}
    return {
        "name": cluster.name,
        "category": cluster.category,
        "topic": cluster.topic,
        "goal": cluster.goal,
        "resumes_from": cluster.resumes_from,
        "turns": turns,
        "judge": judge,
    }


def rejudge_cluster(
    layout: RunLayout,
    cluster: EpisodicCluster,
    *,
    judge_model: str = "sonnet",
    preferences: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Re-run only the judge against an already-recorded transcript, without
    re-driving MR1 — for retrying a judge-only failure (e.g. a rate limit
    on the judge call itself) without re-spending real conversation turns."""
    transcript_path = layout.cluster_transcript(cluster.name)
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"no transcript recorded for cluster {cluster.name!r}: {transcript_path}"
        )
    raw_records = _read_jsonl(transcript_path)
    records = [types.SimpleNamespace(**r) for r in raw_records]
    infra_error = _cluster_infra_error(records)
    if infra_error is not None:
        verdict = JudgeVerdict.degraded(infra_error)
    else:
        verdict = judge_cluster(
            cluster, records, model=judge_model, preferences=preferences or [],
        )
    _write_json(layout.cluster_judge(cluster.name), verdict.to_dict())
    return {
        "name": cluster.name,
        "category": cluster.category,
        "topic": cluster.topic,
        "goal": cluster.goal,
        "resumes_from": cluster.resumes_from,
        "turns": raw_records,
        "judge": verdict.to_dict(),
    }


def rejudge_corpus(
    layout: RunLayout,
    *,
    judge_model: str = "sonnet",
    clusters: Optional[List[EpisodicCluster]] = None,
    judge_delay_s: float = 0.0,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Re-judge every cluster that already has a recorded transcript in
    `layout`, without re-driving MR1. Restricting `clusters` re-judges only
    that subset while leaving every other cluster's previous verdict in
    `result.json` untouched."""
    say = progress or (lambda _m: None)
    if not layout.result.exists():
        raise FileNotFoundError(f"no result.json at {layout.result} — run the corpus first")
    previous = json.loads(layout.result.read_text(encoding="utf-8"))
    previous_by_name = {r["name"]: r for r in previous.get("clusters") or []}
    by_name = {c.name: c for c in CLUSTERS}

    target_clusters = (
        list(clusters) if clusters is not None
        else [by_name[name] for name in previous_by_name if name in by_name]
    )
    preferences = load_marwan_preferences()

    for cluster in target_clusters:
        if not layout.cluster_transcript(cluster.name).exists():
            say(f"skip (no transcript recorded): {cluster.name}")
            continue
        say(f"rejudge: {cluster.name}")
        if judge_delay_s:
            time.sleep(judge_delay_s)
        previous_by_name[cluster.name] = rejudge_cluster(
            layout, cluster, judge_model=judge_model, preferences=preferences,
        )
        _report_cluster(previous_by_name[cluster.name], say)

    ordered_clusters = [by_name[name] for name in previous_by_name if name in by_name]
    config = BehaviorQAConfig(planner=previous.get("planner", "real"), judge_model=judge_model)
    new_result = _assemble_result(ordered_clusters, previous_by_name, config)
    new_result["preferences"] = preferences
    _write_json(layout.result, new_result)
    return new_result


def _load_checkpoint(layout: RunLayout) -> Dict[str, Any]:
    if not layout.checkpoint.exists():
        return {}
    try:
        return json.loads(layout.checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_checkpoint(layout: RunLayout, data: Dict[str, Any]) -> None:
    _write_json(layout.checkpoint, data)


# ---------------------------------------------------------------------
# rollup
# ---------------------------------------------------------------------

def _rollup(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(subset)
    if n == 0:
        return {"judged_clusters": 0}

    judges = [r["judge"] for r in subset]
    scores = [j["partner_score"] for j in judges if j.get("partner_score") is not None]
    behaviors = [j.get("behavior") for j in judges]
    approvals = [j.get("marwan_approval") for j in judges]
    would_continue = [j.get("would_continue_using") for j in judges]
    would_himself = [j.get("would_marwan_have_done_this_himself") for j in judges]
    surprised = [bool(j.get("surprised_by_action")) for j in judges]

    too_aggressive = behaviors.count("TOO_AGGRESSIVE")
    too_passive = behaviors.count("TOO_PASSIVE")

    dims: Dict[str, float] = {}
    for key in DIMENSION_KEYS:
        vals = [
            j["dimensions"].get(key) for j in judges
            if isinstance(j.get("dimensions"), dict) and j["dimensions"].get(key) is not None
        ]
        if vals:
            dims[key] = round(sum(vals) / len(vals), 2)

    surprising_and_approved = sum(
        1 for s, a in zip(surprised, approvals) if s and a == "YES"
    )
    surprising_and_not_approved = sum(
        1 for s, a in zip(surprised, approvals) if s and a != "YES"
    )

    return {
        "judged_clusters": n,
        "mean_partner_score": round(sum(scores) / len(scores), 2) if scores else None,
        "behavior_distribution": dict(Counter(behaviors)),
        "approval_distribution": dict(Counter(approvals)),
        "would_continue_using_pct": round(would_continue.count("YES") / n, 2),
        "would_continue_using_no": [
            r["name"] for r, wc in zip(subset, would_continue) if wc == "NO"
        ],
        "would_have_done_himself_pct": round(would_himself.count("YES") / n, 2),
        "would_have_done_himself_no": [
            r["name"] for r, wh in zip(subset, would_himself) if wh == "NO"
        ],
        "surprise_rate": round(sum(surprised) / n, 2),
        "surprising_and_approved": surprising_and_approved,
        "surprising_and_not_approved": surprising_and_not_approved,
        "initiative_calibration": round((too_aggressive - too_passive) / n, 2),
        "dimension_means": dims,
    }


def _assemble_result(
    all_clusters: List[EpisodicCluster],
    results: Dict[str, Dict[str, Any]],
    config: BehaviorQAConfig,
) -> Dict[str, Any]:
    ordered = [results[c.name] for c in all_clusters if c.name in results]
    judged = [r for r in ordered if not (r.get("judge") or {}).get("judge_error")]
    errored = [r for r in ordered if (r.get("judge") or {}).get("judge_error")]
    infra_errored = [r for r in errored if _is_infra_error((r.get("judge") or {}).get("judge_error"))]

    return {
        "planner": config.planner,
        "judge_model": config.judge_model,
        "clusters": ordered,
        "counts": {
            "clusters": len(ordered),
            "judged_clusters": len(judged),
            "error_clusters": len(errored),
            "infra_error_clusters": len(infra_errored),
            "turns": sum(len(r["turns"]) for r in ordered),
        },
        "error_cluster_names": [r["name"] for r in errored],
        "infra_error_cluster_names": [r["name"] for r in infra_errored],
        "rollup": {
            "overall": _rollup(judged),
            "category_a": _rollup([r for r in judged if r["category"] == "A"]),
            "category_b": _rollup([r for r in judged if r["category"] == "B"]),
        },
        "metrics": compute_metrics(all_clusters, judged),
    }
