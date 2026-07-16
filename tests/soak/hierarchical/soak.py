"""
Soak orchestration: run the conversation, sample, checkpoint, analyse.

`run_soak` drives the full corpus (or resumes a partial run), draining the
scheduler and sampling between turns, restarts the process mid-conversation,
optionally idles to fill an overnight wall-clock budget, and finally runs the
hierarchical + health invariants over the live stores. Everything needed to
regenerate the human-readable review is persisted, so `analyze` can rebuild a
report from an interrupted run without executing anything.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tests.soak.hierarchical import invariants as inv
from tests.soak.hierarchical.arcs import ARCS, RESTART_AFTER_ARC, Arc
from tests.soak.hierarchical.driver import (
    HierarchicalSession,
    RunLayout,
    _append_jsonl,
    _make_paths,
    _read_jsonl,
    _write_json,
)
from tests.soak.hierarchical.fixture import build_fixture_repo
from tests.soak.hierarchical.outcomes import Turn, classify, evaluate_turn, route_of, advised_route_of
from mr1.runtime_test_cli import _patched_runtime_paths


# A flat, indexable view of the corpus: (global_index, arc, turn).
def _indexed_turns() -> List[Tuple[int, Arc, Turn]]:
    out: List[Tuple[int, Arc, Turn]] = []
    i = 0
    for arc in ARCS:
        for turn in arc.turns:
            out.append((i, arc, turn))
            i += 1
    return out


_STOP_REQUESTED = {"flag": False}


def _install_sigint() -> Callable[[], None]:
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):  # noqa: ANN001
        _STOP_REQUESTED["flag"] = True

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        # not on the main thread (e.g. under pytest) — SIGINT handling is a
        # nicety for the CLI, not a correctness requirement.
        return lambda: None

    def restore() -> None:
        try:
            signal.signal(signal.SIGINT, previous)
        except (ValueError, TypeError):
            pass

    return restore


@dataclass
class SoakConfig:
    planner: str = "fake"
    duration_s: float = 0.0          # 0 => run the corpus once and stop
    turn_interval_s: float = 0.0     # realistic pacing between turns
    drain_ticks: int = 4             # scheduler ticks after each turn
    sample_interval_s: float = 0.0   # min wall gap between idle-phase samples
    do_restart: bool = True          # restart mid-conversation
    seed: int = 7


def run_soak(
    layout: RunLayout,
    config: SoakConfig,
    *,
    resume: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """Run (or resume) the hierarchical conversation soak. Returns the result dict."""
    say = progress or (lambda _m: None)
    layout.root.mkdir(parents=True, exist_ok=True)
    build_fixture_repo(layout.fixture)

    checkpoint = _load_checkpoint(layout) if resume else {}
    start_index = int(checkpoint.get("next_global_index", 0)) if resume else 0
    if not resume:
        # fresh run — clear any prior feeds
        for path in (layout.transcript, layout.samples):
            if path.exists():
                path.unlink()

    paths = _make_paths(layout.runtime)
    restore_sigint = _install_sigint()
    _STOP_REQUESTED["flag"] = False
    started_wall = time.time()

    indexed = _indexed_turns()
    if max_turns is not None:
        indexed = indexed[:max_turns]

    result: Dict[str, Any] = {}
    with _patched_runtime_paths(paths):
        session = HierarchicalSession(paths, planner=config.planner, fixture_repo=layout.fixture)
        try:
            restarted = bool(checkpoint.get("restarted", False))
            for global_index, arc, turn in indexed:
                if _STOP_REQUESTED["flag"]:
                    say("SIGINT — stopping cleanly; run is resumable.")
                    break
                if global_index < start_index:
                    continue

                t0 = time.perf_counter()
                payload = session.handle_turn(turn.text, index=global_index, drain_ticks=config.drain_ticks)
                idle_brain = payload.get("idle_brain_calls", 0)
                turn_ms = (time.perf_counter() - t0) * 1000.0

                findings = evaluate_turn(
                    turn, global_index, payload,
                    enforce_consent=(config.planner == "real"),
                )
                entry = _transcript_entry(global_index, arc, turn, payload, findings)
                _append_jsonl(layout.transcript, entry)

                sample = session.sample(
                    turn_index=global_index,
                    arc=arc.name,
                    turn_ms=turn_ms,
                    idle_brain_calls=idle_brain,
                    wall_s=time.time() - started_wall,
                )
                _append_jsonl(layout.samples, sample.to_dict())

                _save_checkpoint(layout, {
                    "planner": config.planner,
                    "next_global_index": global_index + 1,
                    "restarted": restarted,
                    "corpus_turns": len(indexed),
                    "started_wall": started_wall,
                })
                say(f"[{global_index+1}/{len(indexed)}] {arc.name}: {entry['outcome']:16} "
                    f"{'OK' if not findings else str(len(findings))+' finding(s)'}  {turn.text[:60]!r}")

                # restart mid-conversation, once, after the designated arc
                if (config.do_restart and not restarted
                        and arc.name == RESTART_AFTER_ARC
                        and _is_last_turn_of_arc(indexed, global_index, arc)):
                    say("  ↻ restarting runtime mid-conversation …")
                    session.restart()
                    restarted = True
                    _save_checkpoint(layout, {
                        "planner": config.planner,
                        "next_global_index": global_index + 1,
                        "restarted": True,
                        "corpus_turns": len(indexed),
                        "started_wall": started_wall,
                    })

                if config.turn_interval_s > 0:
                    _interruptible_sleep(config.turn_interval_s)

            # ---- idle soak phase (fill remaining wall-clock budget) ----
            corpus_done = start_index >= len(indexed) or (indexed and global_index >= indexed[-1][0])
            if config.duration_s > 0 and not _STOP_REQUESTED["flag"]:
                _idle_phase(session, layout, config, started_wall, len(indexed), say)

            result = _analyze_live(session, layout, config, restarted, started_wall)
        finally:
            session.shutdown()
            restore_sigint()

    _write_json(layout.result, result)
    return result


def _is_last_turn_of_arc(indexed, global_index, arc) -> bool:
    arc_indices = [i for (i, a, _t) in indexed if a.name == arc.name]
    return bool(arc_indices) and global_index == arc_indices[-1]


def _idle_phase(session, layout, config: SoakConfig, started_wall, corpus_turns, say) -> None:
    """After the scripted conversation, keep the runtime alive under light
    read-only load until the wall-clock budget is met. This exercises long-horizon
    runtime health without growing the hierarchy."""
    idle_index = corpus_turns
    probes = ["what's running right now?", "show me the agents you have now."]
    while (time.time() - started_wall) < config.duration_s and not _STOP_REQUESTED["flag"]:
        gap = config.sample_interval_s or config.turn_interval_s or 5.0
        _interruptible_sleep(gap)
        if _STOP_REQUESTED["flag"]:
            break
        t0 = time.perf_counter()
        text = probes[idle_index % len(probes)]
        payload = session.handle_turn(text, index=idle_index, drain_ticks=config.drain_ticks)
        idle_brain = payload.get("idle_brain_calls", 0)
        turn_ms = (time.perf_counter() - t0) * 1000.0
        sample = session.sample(
            turn_index=idle_index, arc="idle", turn_ms=turn_ms,
            idle_brain_calls=idle_brain, wall_s=time.time() - started_wall,
        )
        _append_jsonl(layout.samples, sample.to_dict())
        idle_index += 1
        if idle_index % 20 == 0:
            say(f"  idle soak: {int(time.time()-started_wall)}s / {int(config.duration_s)}s, "
                f"{idle_index - corpus_turns} probes")


def _interruptible_sleep(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end and not _STOP_REQUESTED["flag"]:
        time.sleep(min(0.25, end - time.time()))


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def _analyze_live(session: HierarchicalSession, layout: RunLayout,
                  config: SoakConfig, restarted: bool, started_wall: float) -> Dict[str, Any]:
    limits = inv.HierarchyLimits()
    agents = session.list_agents()
    messages = session.list_messages()
    workflows = session.list_workflow_dicts()

    hierarchy_findings = inv.check_hierarchy(agents, session.root_agent_id, limits)
    message_findings = inv.check_messages(messages, agents)
    workflow_findings = inv.check_workflows_at_shutdown(workflows, limits)

    samples = _read_jsonl(layout.samples)
    health_findings = inv.check_health(samples)
    idle_brain = sum(int(s.get("brain_calls_idle", 0) or 0) for s in samples)
    idle_findings = inv.check_idle_brain_discipline(idle_brain, len(samples))

    transcript = _read_jsonl(layout.transcript)
    turn_findings: List[Tuple[str, int, str, str]] = []
    for entry in transcript:
        for f in entry.get("findings", []):
            if isinstance(f, list) and len(f) == 4:
                turn_findings.append(tuple(f))  # type: ignore[arg-type]

    return _assemble_result(
        config=config,
        restarted=restarted,
        wall_s=time.time() - started_wall,
        agents=[a.to_dict() for a in agents],
        root_agent_id=session.root_agent_id,
        messages=[m.to_dict() for m in messages],
        workflows=workflows,
        samples=samples,
        transcript=transcript,
        turn_findings=turn_findings,
        structural_findings={
            "hierarchy": hierarchy_findings,
            "messages": message_findings,
            "workflows": workflow_findings,
            "health": health_findings,
            "idle_brain": idle_findings,
        },
        limits=limits,
    )


def analyze(layout: RunLayout) -> Dict[str, Any]:
    """Report-only: rebuild a result from persisted feeds without executing.

    If a completed `result.json` exists it is returned as-is. Otherwise a
    partial result is reconstructed from the transcript + samples (outcome and
    health findings only; structural invariants require live stores and are
    marked unavailable)."""
    if layout.result.exists():
        import json
        return json.loads(layout.result.read_text(encoding="utf-8"))

    transcript = _read_jsonl(layout.transcript)
    samples = _read_jsonl(layout.samples)
    turn_findings: List[Tuple[str, int, str, str]] = []
    for entry in transcript:
        for f in entry.get("findings", []):
            if isinstance(f, list) and len(f) == 4:
                turn_findings.append(tuple(f))  # type: ignore[arg-type]
    health_findings = inv.check_health(samples) if samples else []
    idle_brain = sum(int(s.get("brain_calls_idle", 0) or 0) for s in samples)
    idle_findings = inv.check_idle_brain_discipline(idle_brain, len(samples))

    return _assemble_result(
        config=SoakConfig(),
        restarted=False,
        wall_s=samples[-1]["wall_s"] if samples else 0.0,
        agents=[],
        root_agent_id="",
        messages=[],
        workflows=[],
        samples=samples,
        transcript=transcript,
        turn_findings=turn_findings,
        structural_findings={
            "hierarchy": [("medium", "Structural invariants unavailable",
                           "no result.json; run was interrupted before final analysis")],
            "messages": [],
            "workflows": [],
            "health": health_findings,
            "idle_brain": idle_findings,
        },
        limits=inv.HierarchyLimits(),
        partial=True,
    )


def _assemble_result(*, config: SoakConfig, restarted: bool, wall_s: float,
                     agents, root_agent_id, messages, workflows, samples, transcript,
                     turn_findings, structural_findings, limits, partial: bool = False) -> Dict[str, Any]:
    all_structural = [f for group in structural_findings.values() for f in group]
    high_turn = [f for f in turn_findings if f[0] == "high"]
    high_structural = [f for f in all_structural if f[0] == "high"]
    passed = not high_turn and not high_structural and not partial

    # a small sample rollup
    rollup: Dict[str, Any] = {}
    if samples:
        first, last = samples[0], samples[-1]
        rollup = {
            "samples": len(samples),
            "rss_start_mb": round(first.get("rss_bytes", 0) / 1e6, 1),
            "rss_end_mb": round(last.get("rss_bytes", 0) / 1e6, 1),
            "fd_start": first.get("open_fds"),
            "fd_end": last.get("open_fds"),
            "event_bytes_end": last.get("event_bytes"),
            "brain_calls_total": last.get("brain_calls_total"),
            "idle_brain_calls": sum(int(s.get("brain_calls_idle", 0) or 0) for s in samples),
            "wall_s": round(wall_s, 1),
        }

    outcomes_count: Dict[str, int] = {}
    for entry in transcript:
        outcomes_count[entry.get("outcome", "?")] = outcomes_count.get(entry.get("outcome", "?"), 0) + 1

    return {
        "passed": passed,
        "partial": partial,
        "planner": config.planner,
        "restarted": restarted,
        "limits": {
            "max_depth": limits.max_depth,
            "max_children_per_agent": limits.max_children_per_agent,
            "max_total_agents": limits.max_total_agents,
            "max_total_workflows": limits.max_total_workflows,
            "max_workflows_per_agent": limits.max_workflows_per_agent,
            "max_agent_creations_per_hour": limits.max_agent_creations_per_hour,
        },
        "counts": {
            "turns": len(transcript),
            "agents": max(0, len(agents) - 1) if agents else 0,  # exclude root
            "messages": len(messages),
            "workflows": len(workflows),
            "outcomes": outcomes_count,
        },
        "findings": {
            "turn": [list(f) for f in turn_findings],
            "hierarchy": [list(f) for f in structural_findings["hierarchy"]],
            "messages": [list(f) for f in structural_findings["messages"]],
            "workflows": [list(f) for f in structural_findings["workflows"]],
            "health": [list(f) for f in structural_findings["health"]],
            "idle_brain": [list(f) for f in structural_findings["idle_brain"]],
        },
        "rollup": rollup,
        "agents": agents,
        "root_agent_id": root_agent_id,
        "messages": messages,
        "transcript_len": len(transcript),
    }


# ----------------------------------------------------------------------
# transcript / checkpoint
# ----------------------------------------------------------------------

def _transcript_entry(index: int, arc: Arc, turn: Turn, payload: Dict[str, Any], findings) -> Dict[str, Any]:
    created_agents = (payload.get("agents") or {}).get("created") or []
    created_workflows = (payload.get("workflows") or {}).get("created") or []
    created_messages = (payload.get("messages") or {}).get("created") or []
    return {
        "index": index,
        "arc": arc.name,
        "goal": arc.goal,
        "text": turn.text,
        "note": turn.note,
        "allow": list(turn.allow),
        "response_text": (payload.get("response_text") or "")[:600],
        "route": route_of(payload),
        "advised_route": advised_route_of(payload),
        "outcome": classify(payload),
        "created_agents": [
            {"agent_id": a.get("agent_id"), "title": a.get("title"),
             "parent": a.get("parent_agent_id"),
             "mission": (a.get("mission") or a.get("mission_preview") or "")[:300]}
            for a in created_agents
        ],
        "created_workflow_ids": [w.get("workflow_id") for w in created_workflows],
        "created_message_ids": [m.get("message_id") for m in created_messages],
        "approval_ids": [a.get("approval_request_id") for a in (payload.get("approvals_required") or [])],
        "ok": payload.get("ok", True),
        "findings": [list(f) for f in findings],
    }


def _save_checkpoint(layout: RunLayout, data: Dict[str, Any]) -> None:
    _write_json(layout.checkpoint, data)


def _load_checkpoint(layout: RunLayout) -> Dict[str, Any]:
    import json
    if not layout.checkpoint.exists():
        return {}
    try:
        return json.loads(layout.checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
