"""
The hierarchical soak driver.

Wraps the real MR1 runtime the same way the runtime-QA runner does — a real
`MR1` over an isolated runtime root, driven turn-by-turn through the real
`MR1.step(...)` — and adds everything a *long, stateful, wall-clock* soak
needs on top:

* a fake or real brain + compiler (planner mode),
* a message diff in every turn payload (so orchestration decisions that send
  messages are observable),
* scheduler draining between turns so created workflows actually execute,
* resource sampling (RSS, FDs, latency, event bytes) each turn,
* an append-only transcript + samples feed and a checkpoint, so an interrupted
  soak is analysable and resumable,
* an in-place restart (crash + fresh process over the same disk) mid-run.

The driver never calls `create_child_agent`, `submit_workflow`, or
`send_message` itself. It only speaks to MR1 and observes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mr1.worker_runner import MockRunner, RunResult, RunStatus
from mr1.mr1 import MR1, StateManager
from mr1.workflow_store import WorkflowStore

from mr1.runtime_test_cli import (
    RuntimePaths,
    _diff_entities,
    _event_map,
    _load_turn_artifacts,
    _map_by_id,
    _new_items,
    _patched_runtime_paths,
    _error_payload,
)

from tests.soak.realtime import _open_fds, _rss_bytes

import mr1.mrn_loop as mrn_loop
import mr1.worker as worker_module

from tests.soak.hierarchical.fakes import (
    FakeBrainProcess,
    fake_mrn_reasoner,
    fake_worker_run,
    make_fake_compiler,
)
from tests.soak.hierarchical.fixture import build_fixture_repo


# ----------------------------------------------------------------------
# brain-call counting proxy (works for both fake and real MR1Process)
# ----------------------------------------------------------------------

class _CountingBrain:
    """Transparent proxy over the brain process that counts `send` calls."""

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "send_count", 0)

    def send(self, message: str, *, retriable: bool = False) -> str:
        object.__setattr__(self, "send_count", self.send_count + 1)
        return self._target.send(message, retriable=retriable)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


# ----------------------------------------------------------------------
# samples
# ----------------------------------------------------------------------

@dataclass
class Sample:
    turn_index: int
    arc: str
    turn_ms: float
    rss_bytes: int
    open_fds: int
    event_bytes: int
    total_events: int
    agent_count: int
    workflow_count: int
    message_count: int
    approval_count: int
    brain_calls_total: int
    brain_calls_idle: int
    wall_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _decision_map(decisions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """StateManager decisions have no natural id — key by a synthetic
    composite (unique in practice: `task_id` is generated fresh per
    delegation, and two decisions can't share a timestamp+action+task_id)."""
    return {
        f"{d.get('timestamp')}|{d.get('action')}|{d.get('task_id')}": d
        for d in decisions
    }


def _make_paths(runtime_root: Path) -> RuntimePaths:
    return RuntimePaths(
        isolated=True,
        runtime_root=runtime_root,
        workflow_root=runtime_root / "workflows",
        state_path=runtime_root / "active" / "mr1_state.json",
        context_path=runtime_root / "active" / "mr1_context.md",
        dumps_root=runtime_root / "dumps",
        rag_root=runtime_root / "rag",
    )


class HierarchicalSession:
    """One MR1 over one runtime root, driven by conversation, rebuildable in place."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        planner: str,
        fixture_repo: Path,
    ) -> None:
        self._paths = paths
        self._planner = planner
        self._fixture_repo = Path(fixture_repo)
        self._started = False
        self._brain: Optional[_CountingBrain] = None
        self._original_mrn_reasoner = mrn_loop.run_mrn_step_agent
        self._original_worker_run = worker_module.run
        self._mr1 = self._build_mr1()

    # -- construction --------------------------------------------------
    def _build_mr1(self) -> MR1:
        kwargs: Dict[str, Any] = dict(
            state_manager=StateManager(state_path=self._paths.state_path),
            workflow_store=WorkflowStore(root=self._paths.workflow_root),
            inbox_auto_triage=False,
            workflow_auto_tick=False,  # the driver ticks the scheduler explicitly
        )
        if self._planner == "fake":
            kwargs["workflow_runner"] = MockRunner(
                on_poll=lambda _h: RunResult(
                    status=RunStatus.SUCCEEDED, exit_code=0, summary="ok"
                )
            )
            kwargs["workflow_compiler"] = make_fake_compiler(self._fixture_repo)
        return MR1(**kwargs)

    @property
    def mr1(self) -> MR1:
        return self._mr1

    @property
    def brain_calls(self) -> int:
        return self._brain.send_count if self._brain else 0

    def ensure_started(self) -> None:
        if self._started:
            return
        if self._planner == "fake":
            fake = FakeBrainProcess()
            fake.start()
            # A restart builds a fresh in-memory FakeBrainProcess, but the title
            # index on disk survives it — seed from live state so the fake
            # brain never proposes a title that already exists post-restart.
            fake._used_titles.update(a.title for a in self._mr1._scoped_agents.list_agents())
            self._mr1._process = fake
            # A newly-created orchestrator runs its own first step through
            # MRnStepRunner, whose default reasoner spawns a *real* `claude`
            # subprocess independent of MR1's own `_process`. Patch it too, or
            # "fake mode" still makes real LLM calls the moment an agent is born.
            mrn_loop.run_mrn_step_agent = fake_mrn_reasoner
            # A bounded-investigation turn (root.py's _route_bounded_investigation)
            # calls the real mr1.worker.run(), which spawns a real `claude`
            # subprocess — patch it too, or "fake mode" still makes a real LLM
            # call the moment MR1 delegates to a worker.
            worker_module.run = fake_worker_run
        else:
            self._mr1.start()
        # wrap whatever process now exists, for brain-call counting
        self._brain = _CountingBrain(self._mr1._process)
        self._mr1._process = self._brain
        self._started = True

    def restart(self) -> None:
        """Simulate a crash + a fresh process over the same runtime root."""
        try:
            self._mr1.shutdown("hierarchical-soak-restart")
        except Exception:
            pass
        self._started = False
        self._brain = None
        self._mr1 = self._build_mr1()
        self.ensure_started()

    def shutdown(self) -> None:
        try:
            self._mr1.shutdown("hierarchical-soak")
        except Exception:
            pass
        mrn_loop.run_mrn_step_agent = self._original_mrn_reasoner
        worker_module.run = self._original_worker_run

    # -- snapshotting --------------------------------------------------
    def _snapshot(self) -> Dict[str, Dict[str, Any]]:
        caller = self._mr1._root_agent_id
        ra = self._mr1._runtime_access
        events = self._mr1._event_log.list_events()
        return {
            "agents": _map_by_id(ra.list_agents(caller_agent_id=caller), "agent_id"),
            "workflows": _map_by_id(ra.list_workflows(caller_agent_id=caller), "workflow_id"),
            "approvals": _map_by_id(
                ra.list_pending_approvals(caller_agent_id=caller), "approval_request_id"
            ),
            "messages": {
                m.message_id: m.to_dict()
                for m in self._mr1._message_store.list_messages()
            },
            "events": _event_map(events),
            "turn_artifacts": _load_turn_artifacts(self._paths.state_path),
            "decisions": _decision_map(self._mr1._state.decisions),
            # StateManager stores tasks keyed by task_id but doesn't embed
            # the id into the value itself — add it so downstream diff
            # consumers (behavior_qa's worker_spawns) can match by task_id.
            "tasks": {
                task_id: {**task, "task_id": task_id}
                for task_id, task in self._mr1._state.tasks.items()
            },
            "_max_event_index": max((e.event_index for e in events), default=None),
        }

    def handle_turn(self, text: str, *, index: int, drain_ticks: int = 0) -> Dict[str, Any]:
        before = self._snapshot()
        response_text = ""
        errors: List[Dict[str, Any]] = []
        ok = True
        is_builtin = text.strip().startswith("/")
        try:
            self.ensure_started()
            if is_builtin:
                result = self._mr1._handle_builtin(text)
                response_text = result if result is not None else self._mr1.step(text, announce=True)
            else:
                response_text = self._mr1.step(text, announce=True)
        except Exception as exc:  # noqa: BLE001 — a soak records, never crashes
            ok = False
            errors.append(_error_payload(exc))

        # A capability block (and the approval it raises) happens when the
        # scheduler actually launches the task, not inside step() itself — so
        # the "after" snapshot must be taken post-drain, or a workflow
        # submitted this turn that gets blocked a tick later reads as if no
        # approval was ever raised.
        idle_brain_before = self.brain_calls
        for _ in range(max(0, drain_ticks)):
            try:
                self._mr1._scheduler.tick()
            except Exception:
                pass
        idle_brain_calls = self.brain_calls - idle_brain_before

        after = self._snapshot()
        created_agents, updated_agents = _diff_entities(before["agents"], after["agents"])
        created_workflows, updated_workflows = _diff_entities(before["workflows"], after["workflows"])
        created_messages, updated_messages = _diff_entities(before["messages"], after["messages"])
        new_events = _new_items(before["events"], after["events"], sort_key="event_index")
        new_turn_artifacts = _new_items(
            before["turn_artifacts"], after["turn_artifacts"], sort_key="timestamp"
        )
        approvals_required = _new_items(before["approvals"], after["approvals"], sort_key="created_at")
        new_decisions = _new_items(before["decisions"], after["decisions"], sort_key="timestamp")
        created_tasks, updated_tasks = _diff_entities(before["tasks"], after["tasks"])

        return {
            "ok": ok,
            "request_index": index,
            "input": text,
            "response_text": response_text,
            "agents": {"created": created_agents, "updated": updated_agents},
            "workflows": {"created": created_workflows, "updated": updated_workflows},
            "messages": {"created": created_messages, "updated": updated_messages},
            "tasks": {"created": created_tasks, "updated": updated_tasks},
            "approvals_required": approvals_required,
            "timeline": {"events": new_events, "max_event_index": after["_max_event_index"]},
            "turn_artifacts": new_turn_artifacts,
            "decisions": new_decisions,
            "errors": errors,
            "idle_brain_calls": idle_brain_calls,
        }

    # -- metrics -------------------------------------------------------
    def sample(self, *, turn_index: int, arc: str, turn_ms: float,
               idle_brain_calls: int, wall_s: float) -> Sample:
        snap = self._snapshot()
        event_path = self._paths.runtime_root / "events" / "events.jsonl"
        event_bytes = event_path.stat().st_size if event_path.exists() else 0
        return Sample(
            turn_index=turn_index,
            arc=arc,
            turn_ms=round(turn_ms, 3),
            rss_bytes=_rss_bytes(),
            open_fds=_open_fds(),
            event_bytes=event_bytes,
            total_events=len(snap["events"]),
            agent_count=len(snap["agents"]),
            workflow_count=len(snap["workflows"]),
            message_count=len(snap["messages"]),
            approval_count=len(snap["approvals"]),
            brain_calls_total=self.brain_calls,
            brain_calls_idle=idle_brain_calls,
            wall_s=round(wall_s, 3),
        )

    def list_agents(self) -> List[Any]:
        return self._mr1._scoped_agents.list_agents()

    def list_messages(self) -> List[Any]:
        return self._mr1._message_store.list_messages()

    def list_workflow_dicts(self) -> List[Dict[str, Any]]:
        caller = self._mr1._root_agent_id
        return self._mr1._runtime_access.list_workflows(caller_agent_id=caller)

    @property
    def root_agent_id(self) -> str:
        return self._mr1._root_agent_id


# ----------------------------------------------------------------------
# checkpoint / feed IO
# ----------------------------------------------------------------------

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@dataclass
class RunLayout:
    root: Path

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def fixture(self) -> Path:
        return self.root / "fixture"

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.jsonl"

    @property
    def samples(self) -> Path:
        return self.root / "samples.jsonl"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"
