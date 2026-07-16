"""
Drives one episodic cluster's turns against a real (or fake, for harness
testing) MR1 runtime.

Reuses `tests.soak.hierarchical.driver.HierarchicalSession` wholesale rather
than re-deriving turn-handling, snapshot-diffing, or brain-call counting —
this module only adds what's specific to Phase D: per-cluster runtime
isolation, the callback copy-forward mechanism, and capturing the fuller
"outcome model" (route advice, confidence, override reason, soft
expectations) the behavioral judge needs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.behavior_qa.corpus import EpisodicCluster
from tests.soak.hierarchical.driver import HierarchicalSession, RuntimePaths, _make_paths
from tests.soak.hierarchical.outcomes import classify, evaluate_turn


@dataclass
class RunLayout:
    root: Path

    @property
    def fixture(self) -> Path:
        return self.root / "fixture"

    @property
    def clusters_dir(self) -> Path:
        return self.root / "clusters"

    def cluster_dir(self, name: str) -> Path:
        return self.clusters_dir / name

    def cluster_runtime(self, name: str) -> Path:
        return self.cluster_dir(name) / "runtime"

    def cluster_transcript(self, name: str) -> Path:
        return self.cluster_dir(name) / "transcript.jsonl"

    def cluster_judge(self, name: str) -> Path:
        return self.cluster_dir(name) / "judge.json"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"


@dataclass
class TurnRecord:
    """The full per-turn outcome model the judge and report both read."""

    index: int
    text: str
    note: str
    soft_expectations: List[str]
    response_text: str
    route: str
    route_advice: Dict[str, Any]
    override_reason: str
    final_action: Optional[str]
    created_agents: List[Dict[str, Any]]
    created_workflows: List[Dict[str, Any]]
    created_messages: List[Dict[str, Any]]
    approval_ids: List[str]
    ok: bool
    errors: List[Dict[str, Any]]
    outcome: str
    findings: List[tuple]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "note": self.note,
            "soft_expectations": self.soft_expectations,
            "response_text": self.response_text,
            "route": self.route,
            "route_advice": self.route_advice,
            "override_reason": self.override_reason,
            "final_action": self.final_action,
            "created_agents": self.created_agents,
            "created_workflows": self.created_workflows,
            "created_messages": self.created_messages,
            "approval_ids": self.approval_ids,
            "ok": self.ok,
            "errors": self.errors,
            "outcome": self.outcome,
            "findings": [list(f) for f in self.findings],
        }


def _last_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    arts = payload.get("turn_artifacts") or []
    if not arts:
        return {}
    last = arts[-1]
    return last if isinstance(last, dict) else {}


def _final_action(payload: Dict[str, Any]) -> Optional[str]:
    events = (payload.get("timeline") or {}).get("events") or []
    for event in events:
        if event.get("event_type") == "runtime_turn_decided":
            return (event.get("metadata") or {}).get("final_action")
    return None


def build_cluster_session(
    layout: RunLayout,
    cluster: EpisodicCluster,
    *,
    planner: str,
    source_runtime_root: Optional[Path] = None,
) -> HierarchicalSession:
    """Construct the session for one cluster.

    If `source_runtime_root` is given (the caller resolved
    `cluster.resumes_from` to a finished cluster's runtime directory), that
    directory is copied forward before the session is built, so the new
    cluster continues from real prior state — the same "real MR1 rebuilt over
    an existing runtime root" shape `HierarchicalSession.restart()` already
    relies on, just landed at a fresh path first so the source cluster's own
    artifacts are left untouched.
    """
    runtime_root = layout.cluster_runtime(cluster.name)
    if source_runtime_root is not None:
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_runtime_root, runtime_root)
    paths: RuntimePaths = _make_paths(runtime_root)
    return HierarchicalSession(paths, planner=planner, fixture_repo=layout.fixture)


def run_cluster_turns(
    session: HierarchicalSession,
    cluster: EpisodicCluster,
    *,
    drain_ticks: int,
) -> List[TurnRecord]:
    records: List[TurnRecord] = []
    for index, ct in enumerate(cluster.turns):
        payload = session.handle_turn(ct.text, index=index, drain_ticks=drain_ticks)
        artifact = _last_artifact(payload)
        route_advice = artifact.get("route_advice") or {}
        findings = evaluate_turn(ct.turn, index, payload, enforce_consent=False)
        records.append(TurnRecord(
            index=index,
            text=ct.text,
            note=ct.turn.note,
            soft_expectations=list(ct.soft_expectations),
            response_text=str(payload.get("response_text") or ""),
            route=str(artifact.get("route") or ""),
            route_advice=route_advice if isinstance(route_advice, dict) else {},
            override_reason=str(artifact.get("route_advice_override_reason") or ""),
            final_action=_final_action(payload),
            created_agents=list((payload.get("agents") or {}).get("created") or []),
            created_workflows=list((payload.get("workflows") or {}).get("created") or []),
            created_messages=list((payload.get("messages") or {}).get("created") or []),
            approval_ids=[
                a.get("approval_request_id") for a in (payload.get("approvals_required") or [])
            ],
            ok=bool(payload.get("ok", True)),
            errors=list(payload.get("errors") or []),
            outcome=classify(payload),
            findings=findings,
        ))
    return records
