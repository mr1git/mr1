"""
Real-time soak harness (B5).

The 10 000-tick virtual soak proves the *logic* terminates. It cannot prove that
MR1 survives eight hours of wall-clock operation, because it never spends any:
no real subprocesses, no real file descriptors, no real memory pressure, no real
clock skew, no real disk. Those are exactly the things that kill a long-running
process, and none of them can be simulated away.

So this is a separate tool, and it is honest about what it does. It runs a
**real** supervisor — real `SystemClock`, real scheduler, real `WorkerAsyncRunner`
spawning real subprocesses, real consent grants, real capability audit, real
disk — for however long you ask, sampling the whole time. The only thing it will
stub is the brain, and only if you tell it to:

    --planner fake   deterministic planner, no LLM. Validates the harness, and
                     makes an 8h soak free. Everything else is still real.
    --planner real   the actual `CompilerPlanner`, which spawns `claude`.

Nothing here fakes the passage of time. An 8-hour soak takes eight hours. If the
report says it ran for eight hours, it ran for eight hours.

Usage
-----
    python -m tests.soak.realtime --duration 30s --planner fake      # smoke
    python -m tests.soak.realtime --duration 8h  --planner real
    python -m tests.soak.realtime --duration 24h --planner real --objective-interval 30m
    python -m tests.soak.realtime --resume <soak-dir>                # after a crash
    python -m tests.soak.realtime --report <soak-dir>                # re-analyse

Ctrl-C is a clean stop: the samples are already on disk, the report is written,
and `--resume` picks the same run back up.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from mr1.autonomy.consent import ConsentGrantStore
from mr1.autonomy.notify import FileSink, Notifier
from mr1.autonomy.objectives import KIND_RECURRING, ObjectiveStore
from mr1.autonomy.ownership import ROLE_SERVICE, ExecutionOwnership
from mr1.autonomy.retention import RetentionPolicy
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.autonomy.status import collect_status
from mr1.clock import SystemClock
from mr1.event_log import EventLog
from mr1.scoped_agents import AgentStore


SAMPLES_NAME = "samples.jsonl"
REPORT_NAME = "report.json"
CONFIG_NAME = "soak_config.json"
FEED_NAME = "notifications.jsonl"

# The authority the soak objective holds: read-only git inspection, in one
# directory, for the length of the run. Nothing that can write, fetch, or reach
# the network — so an unattended MR1 holds exactly what it needs and no more.
#
# It is a *set* rather than one literal command because the real planner chooses
# its own. Pinned to `^git status`, the first 8h real-planner run parked 43
# seconds in: the compiler answered "report what changed" with
# `git log --oneline -10`, the authority preflight correctly refused it, and the
# objective escalated. MR1 was right; the grant had been over-fitted to the fake
# planner's one hard-coded command.
SOAK_ARG_PREDICATE = {
    "argv": {
        "regex": (
            r"^git\s+"
            r"(status|log|diff|show|branch|tag|ls-files|"
            r"rev-parse|describe|shortlog|blame|count-objects)"
            r"(\s|$)"
        )
    }
}


class SoakDirtyError(RuntimeError):
    """The soak directory already holds a run. Refuse rather than inherit it."""


# ---------------------------------------------------------------------------
# The test objective: low-risk, bounded scope, genuinely exercised
# ---------------------------------------------------------------------------


def genesis_style_spec(workspace: Path) -> dict[str, Any]:
    """
    The workflow the soak objective runs, over and over, for hours.

    `git status --short` against the repo: real subprocess, real capability
    gate, real risk-1.0 `shell_command` — so it must pass through a real consent
    grant to run at all. Read-only and idempotent, so a soak that crashes and
    retries cannot damage anything.

    This is the shape of the Genesis cycle (look at the repo, decide something)
    with the deciding removed. What it proves is that MR1 can hold standing
    authority and use it, unattended, for hours, without leaking anything.
    """
    return {
        "title": "soak: inspect the repository",
        "tasks": [
            {
                "label": "inspect",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {
                    "argv": ["git", "status", "--short"],
                    "cwd": str(workspace),
                },
            }
        ],
    }


class FakeSoakPlanner:
    """
    Deterministic planner. No LLM, no network, no cost.

    Every other part of the run is real. This exists so the harness itself can
    be validated in thirty seconds rather than eight hours, and so a 24h soak
    can be run without spending 24h of tokens on a spec that never changes.
    """

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self.calls = 0

    def plan(self, objective, context) -> dict[str, Any]:
        self.calls += 1
        return genesis_style_spec(self._workspace)


class CountingPlanner:
    """
    Wraps the real planner so its calls are actually counted.

    Without this, `--planner real` reported `brain_calls: 0` forever: the sampler
    reads `getattr(planner, "calls", 0)` and `CompilerPlanner` has no such
    attribute. That silently zeroed the single most important number in the run —
    LLM calls against idle ticks — and would have reported a *passing* soak whose
    cost invariant was never checked at all. A metric that cannot be wrong because
    it is always zero is worse than no metric.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.calls = 0

    def plan(self, objective, context) -> dict[str, Any]:
        self.calls += 1
        return self._inner.plan(objective, context)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    at: str
    elapsed_s: float
    # process
    rss_bytes: int
    cpu_s: float
    threads: int
    open_fds: int
    # disk
    runtime_bytes: int
    events_live_bytes: int
    events_archive_bytes: int
    archive_segments: int
    # runtime
    total_events: int
    workflows_total: int
    workflows_active: int
    objective_runs: int
    objective_status: str
    brain_calls: int
    grant_uses: int
    # loops
    supervisor_ticks: int
    supervisor_tick_errors: int
    scheduler_ticks: int
    scheduler_tick_errors: int
    tick_latency_ms: float
    heartbeat_age_s: Optional[float]
    health: str
    backpressure: list[str] = field(default_factory=list)
    # Which run wrote this sample. A resumed soak appends to the same file, so
    # the analyser has to know where one segment ends and the next begins —
    # and inferring that from `elapsed_s` going backwards does not work, because
    # every run starts at 0.0 and `0.0 < 0.0` is false. Defaulted so samples
    # written before this field existed still load.
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SoakHarness:
    """
    Runs a real supervisor for a wall-clock duration and watches it.

    The supervisor runs in *this* process: that is what makes the memory and FD
    numbers meaningful (they are the numbers of the thing under test) and what
    lets the planner be swapped without a CLI hook. It still takes the real
    `ServiceLock` and the real `ExecutionOwnership`, so it is the same
    deployment shape `mr1 serve` uses.
    """

    def __init__(
        self,
        soak_dir: Path,
        *,
        workspace: Path,
        duration_s: float,
        sample_interval_s: float = 60.0,
        tick_interval_s: float = 10.0,
        objective_interval_s: float = 300.0,
        planner: str = "fake",
        retention_interval_s: float = 600.0,
        events_max_live_bytes: int = 1 * 1024 * 1024,
        events_keep_recent: int = 200,
    ):
        self.soak_dir = Path(soak_dir)
        self.soak_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_root = self.soak_dir / "runtime"
        self.workspace = Path(workspace)
        self.duration_s = float(duration_s)
        self.sample_interval_s = float(sample_interval_s)
        self.tick_interval_s = float(tick_interval_s)
        self.objective_interval_s = float(objective_interval_s)
        self.planner_kind = planner
        self.retention_interval_s = retention_interval_s
        self.events_max_live_bytes = events_max_live_bytes
        self.events_keep_recent = events_keep_recent

        self.clock = SystemClock()
        self._stop = threading.Event()
        self._started_monotonic = 0.0
        self._samples: list[Sample] = []
        self._planner: Any = None
        self._supervisor: Optional[Supervisor] = None
        self._objective_id: Optional[str] = None
        self._grant_id: Optional[str] = None
        self._interrupted = False
        self._parked_reason: Optional[str] = None
        self._run_id = f"run-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"

    # -- setup ---------------------------------------------------------

    def has_previous_run(self) -> bool:
        """Does this directory already hold soak state from an earlier run?"""
        return any(
            (self.runtime_root / name).exists()
            for name in ("objectives", "workflows", "consent_grants", "health.json")
        )

    def reset(self) -> None:
        """Wipe the previous run. Explicit, and loud."""
        for path in (self.runtime_root,):
            if path.exists():
                shutil.rmtree(path)
        for name in (SAMPLES_NAME, REPORT_NAME):
            try:
                (self.soak_dir / name).unlink()
            except OSError:
                pass
        print(f"[soak] --reset: wiped previous run state in {self.soak_dir}")

    def prepare(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        agents = AgentStore(root=self.runtime_root / "agents")
        objectives = ObjectiveStore(self.runtime_root, clock=self.clock)
        consent = ConsentGrantStore(
            self.runtime_root,
            clock=self.clock,
            scoped_agent_store=agents,
        )

        existing = [
            item for item in objectives.list_objectives()
            if item.title == "soak"
        ]
        if existing:
            objective = existing[0]
            # A parked objective is terminal for an unattended run: the supervisor
            # will not replan it, so resuming into one would tick for eight hours
            # and do nothing. Say so now rather than at the end.
            if objective.status in {"waiting_human", "quarantined"}:
                raise SoakDirtyError(
                    f"the objective in {self.soak_dir} is parked as "
                    f"{objective.status} from a previous run and will never plan "
                    f"again on its own:\n    {objective.status_reason[:160]}\n"
                    "Start a clean run with --reset (or a fresh --dir)."
                )
        else:
            objective = objectives.create(
                title="soak",
                # Concrete enough that the real planner stays inside the grant.
                # "inspect the repository and report what changed" is the kind of
                # statement a human writes and a compiler answers with whatever
                # git subcommand it likes — which is how the first 8h real run
                # ended up asking for `git log` under a `git status` grant.
                statement=(
                    "run a single read-only git inspection of the repository "
                    "working tree using one shell_command task — for example "
                    "`git status --short` — and report the result. "
                    "Do not modify, stage, commit, fetch, or push anything."
                ),
                kind=KIND_RECURRING,
                trigger={
                    "type": "interval",
                    "interval_s": self.objective_interval_s,
                    # A soak that is paused, or whose machine sleeps, must not
                    # come back and fire a hundred workflows at once. This is
                    # the B2 bound, exercised for real.
                    "missed_run_policy": "catch_up_once",
                },
                owner_agent_id=agents.root_agent_id,
                idempotent=True,
            )
        self._objective_id = objective.objective_id

        # A grant left behind by an earlier run carries that run's predicate. If
        # the harness's policy has since changed, reusing it silently means the
        # fix you just made never takes effect — which is exactly how the second
        # 8h attempt inherited `^git status` and parked before it planned once.
        # The authority MR1 holds must be the authority this run intends.
        active = consent.list_active(objective_id=objective.objective_id)
        stale = [
            grant for grant in active
            if grant.arg_predicate != SOAK_ARG_PREDICATE
            or grant.capability_name != "shell_command"
        ]
        for grant in stale:
            consent.revoke(
                grant.grant_id,
                revoked_by=agents.root_agent_id,
                reason="soak: grant predicate no longer matches the harness policy",
            )
            print(
                f"[soak] revoked a stale grant from a previous run "
                f"({grant.arg_predicate.get('argv', {}).get('regex', '?')})"
            )

        current = [grant for grant in active if grant not in stale]
        if current:
            self._grant_id = current[0].grant_id
        else:
            grant = consent.create(
                objective_id=objective.objective_id,
                capability_name="shell_command",
                scope_roots=[self.workspace],
                max_risk=1.0,
                granted_by=agents.root_agent_id,
                ttl_s=max(self.duration_s * 3, 7 * 86_400),
                arg_predicate=dict(SOAK_ARG_PREDICATE),
                reason="realtime soak: read-only git inspection",
            )
            self._grant_id = grant.grant_id

        self._planner = (
            FakeSoakPlanner(self.workspace)
            if self.planner_kind == "fake" else
            self._real_planner(agents)
        )

        self._supervisor = Supervisor(
            self.runtime_root,
            config=SupervisorConfig(
                tick_interval_s=self.tick_interval_s,
                scheduler_tick_interval_s=1.0,
                retention_interval_s=self.retention_interval_s,
                doctor_interval_s=300.0,
                max_concurrent_workflows=2,
                max_plans_per_hour=120,
                max_workflows_per_objective_per_day=10_000,
            ),
            clock=self.clock,
            workspace_root=self.workspace,
            planner=self._planner,
            execution_ownership=ExecutionOwnership(self.runtime_root, role=ROLE_SERVICE),
            retention_policy=RetentionPolicy(
                events_max_live_bytes=self.events_max_live_bytes,
                events_keep_recent=self.events_keep_recent,
                workflow_archive_after_days=0.0,   # archive terminal work aggressively
                workflow_keep_recent=20,
            ),
            notifier=Notifier(
                self.runtime_root,
                sinks=[FileSink(path=self.soak_dir / FEED_NAME)],
                clock=self.clock,
            ),
        )

        self._write_config()

    def _real_planner(self, agents):
        from mr1.autonomy.planner import CompilerPlanner
        from mr1.workflow_compiler import WorkflowCompilerClient

        return CountingPlanner(
            CompilerPlanner(WorkflowCompilerClient(scoped_agent_store=agents))
        )

    def _write_config(self) -> None:
        payload = {
            "duration_s": self.duration_s,
            "sample_interval_s": self.sample_interval_s,
            "tick_interval_s": self.tick_interval_s,
            "objective_interval_s": self.objective_interval_s,
            "planner": self.planner_kind,
            "workspace": str(self.workspace),
            "objective_id": self._objective_id,
            "grant_id": self._grant_id,
            "started_at": self.clock.now_iso(),
            "pid": os.getpid(),
        }
        (self.soak_dir / CONFIG_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # -- the run -------------------------------------------------------

    def run(self) -> dict[str, Any]:
        if self._supervisor is None:
            self.prepare()

        self._install_signal_handlers()
        self._started_monotonic = time.monotonic()

        supervisor = self._supervisor
        assert supervisor is not None
        supervisor._claim_execution_authority()

        sampler = threading.Thread(target=self._sample_loop, name="soak-sampler", daemon=True)
        sampler.start()

        print(
            f"[soak] running for {self.duration_s:.0f}s "
            f"(planner={self.planner_kind}, tick={self.tick_interval_s}s, "
            f"objective every {self.objective_interval_s:.0f}s)"
        )
        print(f"[soak] dir: {self.soak_dir}")

        try:
            while not self._stop.is_set():
                if (time.monotonic() - self._started_monotonic) >= self.duration_s:
                    break
                supervisor.tick()
                if self._parked():
                    break
                self._stop.wait(self.tick_interval_s)
        finally:
            self._stop.set()
            sampler.join(timeout=self.sample_interval_s + 5)
            self._sample()  # one final reading
            supervisor.shutdown()

        report = self.report()
        self._write_report(report)
        return report

    def _parked(self) -> bool:
        """
        Stop the moment the objective needs a human. Do not idle for eight hours.

        `waiting_human` and `quarantined` are terminal for an unattended run: the
        supervisor will not replan a parked objective, so every remaining tick is
        guaranteed to do nothing. The first 8h real-planner soak parked 43 seconds
        in and then sat there for hours collecting samples of an idle process —
        which is not a soak, it is a very slow way to read one escalation.

        This is a *finding*, not a crash: the run ends, the report says the
        objective parked and why, and the escalation is already in the inbox.
        """
        if not self._objective_id:
            return False
        try:
            objective = ObjectiveStore(self.runtime_root, clock=self.clock).load(
                self._objective_id
            )
        except Exception:  # noqa: BLE001 - never let the guard kill the soak
            return False
        if objective is None or objective.status not in {"waiting_human", "quarantined"}:
            return False

        self._parked_reason = objective.status_reason or objective.status
        print(
            f"\n[soak] the objective parked as {objective.status} after "
            f"{time.monotonic() - self._started_monotonic:.0f}s — stopping early.\n"
            f"[soak] MR1 needs a human and will not plan again on its own:\n"
            f"[soak]   {self._parked_reason[:200]}\n"
            f"[soak] the escalation is in the inbox; see notifications.jsonl",
            file=sys.stderr,
        )
        self._stop.set()
        return True

    def _install_signal_handlers(self) -> None:
        def _handle(signum, _frame):
            self._interrupted = True
            print(f"\n[soak] {signal.Signals(signum).name} — stopping cleanly, samples are safe")
            self._stop.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, _handle)
            except ValueError:  # not the main thread
                pass

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as exc:  # noqa: BLE001 - sampling must never kill the soak
                print(f"[soak] sample failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            self._stop.wait(self.sample_interval_s)

    def _sample(self) -> Sample:
        supervisor = self._supervisor
        assert supervisor is not None

        gauges = supervisor.gauges()
        scheduler_metrics = supervisor.scheduler.metrics()
        events = EventLog(self.runtime_root / "events")
        history = events.history_stats()
        objectives = ObjectiveStore(self.runtime_root, clock=self.clock)
        objective = objectives.load(self._objective_id) if self._objective_id else None
        consent = ConsentGrantStore(self.runtime_root, clock=self.clock)
        grant = consent.load(self._grant_id) if self._grant_id else None
        status = collect_status(self.runtime_root, clock=self.clock)

        sample = Sample(
            at=self.clock.now_iso(),
            elapsed_s=round(time.monotonic() - self._started_monotonic, 1),
            rss_bytes=_rss_bytes(),
            cpu_s=round(time.process_time(), 3),
            threads=threading.active_count(),
            open_fds=_open_fds(),
            runtime_bytes=_dir_bytes(self.runtime_root),
            events_live_bytes=int(history["live_bytes"]),
            events_archive_bytes=int(history["archive_bytes"]),
            archive_segments=int(history["archive_segments"]),
            total_events=int(history["total_events"]),
            workflows_total=(
                # Live *and* archived. Retention moves terminal workflows out of
                # `workflows/` while the soak is running, so counting only the
                # live directory undercounts the work MR1 actually did — and
                # made a 24h soak report a duplicate launch that never happened.
                len(list((self.runtime_root / "workflows").glob("wf-*")))
                + len(list((self.runtime_root / "archive" / "workflows").glob("wf-*")))
            ),
            workflows_active=int(gauges.get("active_workflows", 0)),
            objective_runs=int(objective.success_count) if objective else 0,
            objective_status=objective.status if objective else "missing",
            brain_calls=int(getattr(self._planner, "calls", 0)),
            grant_uses=int(grant.use_count) if grant else 0,
            supervisor_ticks=int(gauges.get("supervisor_tick_count", 0)),
            supervisor_tick_errors=int(gauges.get("supervisor_tick_errors", 0)),
            scheduler_ticks=int(scheduler_metrics["tick_count"]),
            scheduler_tick_errors=int(scheduler_metrics["tick_errors"]),
            tick_latency_ms=float(gauges.get("supervisor_tick_duration_ms", 0.0)),
            heartbeat_age_s=status.service.get("heartbeat_age_s"),
            health=status.health,
            backpressure=list(gauges.get("backpressure", [])),
            run_id=self._run_id,
        )
        self._samples.append(sample)
        self._append_sample(sample)
        return sample

    def _append_sample(self, sample: Sample) -> None:
        path = self.soak_dir / SAMPLES_NAME
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # -- analysis ------------------------------------------------------

    def report(self) -> dict[str, Any]:
        samples = self._samples or load_samples(self.soak_dir)
        report = analyse(
            samples,
            runtime_root=self.runtime_root,
            duration_s=self.duration_s,
            planner=self.planner_kind,
            interrupted=self._interrupted,
            objective_id=self._objective_id,
        )
        if self._parked_reason:
            report["parked_reason"] = self._parked_reason
        return report

    def _write_report(self, report: dict[str, Any]) -> None:
        (self.soak_dir / REPORT_NAME).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def analyse(
    samples: list[Sample],
    *,
    runtime_root: Path,
    duration_s: float,
    planner: str,
    interrupted: bool = False,
    objective_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Turn the samples into a verdict.

    Each check is a claim the soak is meant to falsify. A soak that "passed"
    without checking anything is a soak that wasted eight hours.
    """
    runtime_root = Path(runtime_root)
    failures: list[str] = []
    warnings: list[str] = []

    if not samples:
        return {
            "passed": False,
            "failures": ["no samples were collected"],
            "warnings": [],
            "samples": 0,
        }

    first, last = samples[0], samples[-1]

    # 1. No stuck objective.
    if last.objective_status in {"quarantined", "waiting_human"}:
        failures.append(
            f"objective ended {last.objective_status} — MR1 gave up or is blocked"
        )

    # 2. No unbounded retry / no runaway planning.
    #    Every brain call must correspond to a run. More plans than runs means
    #    MR1 replanned something repeatedly, which is the runaway signature.
    if last.brain_calls > max(1, last.objective_runs) * 3:
        failures.append(
            f"{last.brain_calls} brain calls for {last.objective_runs} completed runs "
            "— planning is running away"
        )

    # 3. No unexpected LLM calls on idle ticks. A tick that finds nothing due
    #    must cost nothing, so brain calls can never exceed the number of times
    #    the objective actually fired.
    if last.supervisor_ticks > 0 and last.brain_calls > last.supervisor_ticks:
        failures.append(
            f"{last.brain_calls} brain calls across {last.supervisor_ticks} ticks "
            "— the brain is being called on idle ticks"
        )

    # The timeline is the authority for anything we count, not the filesystem.
    # Retention archives terminal workflows *while the soak runs*, so a directory
    # count is a count of what has not been archived yet — which is not the same
    # question. `list_events()` is complete history (B1), so it is immune to that.
    try:
        events = EventLog(runtime_root / "events").list_events()
    except Exception as exc:  # noqa: BLE001
        events = []
        warnings.append(f"could not read the timeline: {exc}")

    workflows_created = sum(1 for e in events if e.event_type == "workflow_created")
    tasks_started = sum(1 for e in events if e.event_type == "workflow_task_started")

    # 4. No duplicate workflow launch. One run, one workflow, one task start, one
    #    consent-authorized execution. Any of these outrunning the others means a
    #    task was launched twice — the thing two processes ticking one runtime
    #    root would cause, and the thing B8 exists to make impossible.
    if workflows_created and tasks_started > workflows_created:
        failures.append(
            f"{tasks_started} task starts for {workflows_created} workflows "
            "— a task was launched more than once"
        )
    if workflows_created and last.grant_uses > workflows_created:
        failures.append(
            f"the consent grant was used {last.grant_uses} times for "
            f"{workflows_created} workflows — a task ran more than once"
        )

    # 5. No authority use outside consent.

    executed = [e for e in events if e.event_type == "capability_executed"]
    # The authority question is asked at the gate, not at the exec: it is
    # `capability_allowed` that records *why* a capability was permitted. For a
    # risk-1.0 capability the only two legitimate answers are a standing consent
    # grant or a human approval. Anything else means MR1 authorized itself.
    allowed = [e for e in events if e.event_type == "capability_allowed"]
    ungranted = [
        e for e in allowed
        if e.target_id == "shell_command"
        and not (e.metadata or {}).get("consent_grant_id")
        and not e.approval_request_id
    ]
    if ungranted:
        failures.append(
            f"{len(ungranted)} risk-1.0 execution(s) were allowed with neither a "
            "consent grant nor an approval — authority was used outside consent"
        )

    blocked = [e for e in events if e.event_type == "capability_blocked"]
    if blocked:
        warnings.append(f"{len(blocked)} capability request(s) were blocked")

    # 6. No stale heartbeat — *while the supervisor was up*.
    #
    #    A resumed soak spans a gap the process was legitimately not running
    #    across, and the first sample after each (re)start reads the previous
    #    segment's `health.json`. Counting that as "the loop went quiet" makes
    #    every `--resume` report a false stale heartbeat, which would have made
    #    the resume feature useless.
    #
    #    Segments come from `run_id`, not from `elapsed_s` going backwards: every
    #    run starts at 0.0, and `0.0 < 0.0` is false, so a boundary between two
    #    runs that both began at zero was invisible. That is precisely how the
    #    dead 8h attempt's stale heartbeat leaked into a later run's verdict.
    ages: list[float] = []
    previous_run: Optional[str] = None
    for index, sample in enumerate(samples):
        run_id = getattr(sample, "run_id", "") or ""
        starts_segment = index == 0 or run_id != previous_run
        previous_run = run_id
        if starts_segment:
            # The first heartbeat a run reads is whatever the *previous* run left
            # in health.json. It says nothing about this one.
            continue
        if sample.heartbeat_age_s is not None:
            ages.append(sample.heartbeat_age_s)

    max_age = max(ages) if ages else None
    if max_age is not None and max_age > 180:
        failures.append(f"heartbeat went stale: {max_age:.0f}s without a tick")

    # 7. No tick errors.
    if last.supervisor_tick_errors:
        failures.append(f"{last.supervisor_tick_errors} supervisor tick error(s)")
    if last.scheduler_tick_errors:
        failures.append(f"{last.scheduler_tick_errors} scheduler tick error(s)")

    # 8. Bounded growth, consistent with retention. Memory is the one that kills
    #    a long-running process, and disk is the one that kills it slowly.
    rss_growth = last.rss_bytes - first.rss_bytes
    rss_growth_ratio = (last.rss_bytes / first.rss_bytes) if first.rss_bytes else 1.0
    if rss_growth_ratio > 2.0 and rss_growth > 256 * 1024**2:
        failures.append(
            f"RSS grew {rss_growth / 1024**2:.0f} MiB "
            f"({rss_growth_ratio:.1f}x) — that is a leak, not a working set"
        )

    fd_growth = last.open_fds - first.open_fds
    if fd_growth > 100:
        failures.append(f"open file descriptors grew by {fd_growth} — a handle leak")

    if last.events_live_bytes > 4 * 1024**2 and last.archive_segments == 0:
        warnings.append(
            "the live event log grew past 4 MiB without rotating — check retention"
        )

    latencies = [s.tick_latency_ms for s in samples if s.tick_latency_ms > 0]
    latencies_sorted = sorted(latencies)

    def _pct(p: float) -> float:
        if not latencies_sorted:
            return 0.0
        index = min(len(latencies_sorted) - 1, int(len(latencies_sorted) * p))
        return round(latencies_sorted[index], 3)

    # Tick cost must not creep: an O(n) tick is a soak that gets slower forever.
    half = len(latencies) // 2
    early = latencies[:half] or [0.0]
    late = latencies[half:] or [0.0]
    early_median = sorted(early)[len(early) // 2]
    late_median = sorted(late)[len(late) // 2]
    if early_median > 0 and late_median > early_median * 5 and late_median > 50:
        failures.append(
            f"tick latency crept from {early_median:.1f}ms to {late_median:.1f}ms "
            "— the tick is getting more expensive as history grows"
        )

    return {
        "passed": not failures,
        "interrupted": interrupted,
        "planner": planner,
        "objective_id": objective_id,
        "duration_requested_s": duration_s,
        "duration_actual_s": last.elapsed_s,
        "samples": len(samples),
        "failures": failures,
        "warnings": warnings,
        "invariants_checked": [
            "no stuck objective",
            "no unbounded retry / runaway planning",
            "no unexpected LLM calls on idle ticks",
            "no duplicate workflow launch",
            "no authority use outside consent",
            "no stale heartbeat",
            "no tick errors",
            "bounded memory, fd, and disk growth consistent with retention",
            "flat tick latency",
        ],
        "totals": {
            "objective_runs": last.objective_runs,
            "objective_status": last.objective_status,
            "brain_calls": last.brain_calls,
            "grant_uses": last.grant_uses,
            "workflows": workflows_created or last.workflows_total,
            "task_starts": tasks_started,
            "supervisor_ticks": last.supervisor_ticks,
            "scheduler_ticks": last.scheduler_ticks,
            "total_events": last.total_events,
            "capability_executions": len(executed),
        },
        "memory": {
            "rss_start_bytes": first.rss_bytes,
            "rss_end_bytes": last.rss_bytes,
            "rss_growth_bytes": rss_growth,
            "rss_growth_ratio": round(rss_growth_ratio, 3),
            "fd_start": first.open_fds,
            "fd_end": last.open_fds,
            "threads_end": last.threads,
        },
        "disk": {
            "runtime_bytes_start": first.runtime_bytes,
            "runtime_bytes_end": last.runtime_bytes,
            "events_live_bytes_end": last.events_live_bytes,
            "events_archive_bytes_end": last.events_archive_bytes,
            "archive_segments": last.archive_segments,
        },
        "tick_latency_ms": {
            "p50": _pct(0.5),
            "p95": _pct(0.95),
            "max": round(max(latencies), 3) if latencies else 0.0,
            "early_median": round(early_median, 3),
            "late_median": round(late_median, 3),
        },
        "cpu_s": last.cpu_s,
        "health_end": last.health,
        "backpressure_end": last.backpressure,
    }


def load_samples(soak_dir: Path) -> list[Sample]:
    path = Path(soak_dir) / SAMPLES_NAME
    if not path.exists():
        return []
    samples: list[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            samples.append(Sample(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    return samples


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "",
        "=" * 68,
        f"  REAL-TIME SOAK — {'PASSED' if report['passed'] else 'FAILED'}",
        "=" * 68,
        f"  planner            {report['planner']}",
        f"  duration           {report['duration_actual_s']:.0f}s "
        f"of {report['duration_requested_s']:.0f}s requested"
        + ("  (interrupted)" if report.get("interrupted") else ""),
        f"  samples            {report['samples']}",
        "",
    ]
    totals = report["totals"]
    lines += [
        "  WORK",
        f"    objective runs   {totals['objective_runs']}  ({totals['objective_status']})",
        f"    workflows        {totals['workflows']}",
        f"    brain calls      {totals['brain_calls']}",
        f"    grant uses       {totals['grant_uses']}",
        f"    capability execs {totals['capability_executions']}",
        f"    supervisor ticks {totals['supervisor_ticks']}",
        f"    scheduler ticks  {totals['scheduler_ticks']}",
        f"    events           {totals['total_events']}",
        "",
    ]
    memory = report["memory"]
    lines += [
        "  RESOURCES",
        f"    RSS              {memory['rss_start_bytes'] / 1024**2:.1f} MiB "
        f"-> {memory['rss_end_bytes'] / 1024**2:.1f} MiB "
        f"({memory['rss_growth_ratio']:.2f}x)",
        f"    file descriptors {memory['fd_start']} -> {memory['fd_end']}",
        f"    threads          {memory['threads_end']}",
        f"    cpu              {report['cpu_s']:.1f}s",
        "",
    ]
    disk = report["disk"]
    lines += [
        "  DISK",
        f"    runtime root     {disk['runtime_bytes_start'] / 1024**2:.1f} MiB "
        f"-> {disk['runtime_bytes_end'] / 1024**2:.1f} MiB",
        f"    events live      {disk['events_live_bytes_end'] / 1024:.0f} KiB",
        f"    events archived  {disk['events_archive_bytes_end'] / 1024:.0f} KiB "
        f"in {disk['archive_segments']} segment(s)",
        "",
    ]
    latency = report["tick_latency_ms"]
    lines += [
        "  TICK LATENCY (ms)",
        f"    p50 {latency['p50']}   p95 {latency['p95']}   max {latency['max']}",
        f"    early median {latency['early_median']} -> late median {latency['late_median']}",
        "",
        f"  HEALTH             {report['health_end']}",
        "",
    ]
    if report.get("parked_reason"):
        lines += [
            "  STOPPED EARLY — THE OBJECTIVE NEEDS A HUMAN",
            f"    {report['parked_reason'][:180]}",
            "    MR1 will not plan again on its own. The escalation is in the inbox.",
            "",
        ]
    if report["failures"]:
        lines.append("  FAILURES")
        lines += [f"    ✗ {item}" for item in report["failures"]]
        lines.append("")
    if report["warnings"]:
        lines.append("  WARNINGS")
        lines += [f"    ! {item}" for item in report["warnings"]]
        lines.append("")
    if not report["failures"]:
        lines.append("  INVARIANTS HELD")
        lines += [f"    ✓ {item}" for item in report["invariants_checked"]]
        lines.append("")
    lines.append("=" * 68)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Process metrics
# ---------------------------------------------------------------------------


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024


def _open_fds() -> int:
    """
    Count this process's open descriptors, without spawning anything.

    `/proc/<pid>/fd` on Linux, `/dev/fd` on macOS. Both are a directory listing:
    microseconds, no fork.

    This used to shell out to `lsof` with a 20-second timeout on every sample.
    That is a subprocess spawned to measure a file-descriptor count — 1440 of
    them over a 24h soak — which perturbs the number it is trying to read and
    can stall the sampler for 20 seconds when the system is busy. `lsof` remains
    only as a last resort, on a short leash.
    """
    for path in (f"/proc/{os.getpid()}/fd", "/dev/fd"):
        try:
            return len(os.listdir(path))
        except OSError:
            continue
    try:
        out = subprocess.run(
            ["lsof", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return max(0, len(out.stdout.strip().splitlines()) - 1)
    except Exception:  # noqa: BLE001
        return -1


def _dir_bytes(path: Path) -> int:
    total = 0
    for item in Path(path).rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_duration(raw: str) -> float:
    text = str(raw).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86_400}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tests.soak.realtime",
        description="Run and analyse a real wall-clock soak of the MR1 autonomy runtime.",
    )
    parser.add_argument(
        "--duration",
        default="30s",
        help="Wall-clock duration: 30s, 45m, 8h, 24h. This is real time; it is not simulated.",
    )
    parser.add_argument(
        "--planner",
        choices=["fake", "real"],
        default="fake",
        help="fake: deterministic, no LLM (everything else is still real). real: spawns claude.",
    )
    parser.add_argument("--dir", type=Path, default=None, help="Soak directory (default: a new one).")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Repo the objective inspects.")
    parser.add_argument("--sample-interval", default="60s")
    parser.add_argument("--tick-interval", default="10s")
    parser.add_argument("--objective-interval", default="5m", help="How often the objective fires.")
    parser.add_argument("--retention-interval", default="10m")
    parser.add_argument(
        "--events-max-live-bytes",
        type=int,
        default=1 * 1024 * 1024,
        help="Rotate events.jsonl past this size during the soak (proves retention runs).",
    )
    parser.add_argument(
        "--events-keep-recent",
        type=int,
        default=200,
        help="Events left in the live log on rotation. Must be < the events a soak produces.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume a soak directory.")
    parser.add_argument("--report", type=Path, default=None, help="Re-analyse a finished soak directory.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Wipe a previous run in --dir and start clean. Required to reuse a "
            "directory that already holds one: a new run must never silently "
            "inherit an old objective's parked state or an old grant's authority."
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.report:
        samples = load_samples(args.report)
        report = analyse(
            samples,
            runtime_root=Path(args.report) / "runtime",
            duration_s=0.0,
            planner="unknown",
        )
        print(json.dumps(report, indent=2, default=str) if args.json else render_report(report))
        return 0 if report["passed"] else 1

    soak_dir = args.resume or args.dir or (
        Path("soak-runs") / time.strftime("soak-%Y%m%dT%H%M%S")
    )

    harness = SoakHarness(
        soak_dir,
        workspace=args.workspace,
        duration_s=parse_duration(args.duration),
        sample_interval_s=parse_duration(args.sample_interval),
        tick_interval_s=parse_duration(args.tick_interval),
        objective_interval_s=parse_duration(args.objective_interval),
        planner=args.planner,
        retention_interval_s=parse_duration(args.retention_interval),
        events_max_live_bytes=args.events_max_live_bytes,
        events_keep_recent=args.events_keep_recent,
    )

    if args.reset:
        harness.reset()
    elif args.resume is None and harness.has_previous_run():
        # The trap this exists to close: the second 8h attempt reused the first
        # one's directory, inherited its parked objective and its `^git status`
        # grant, and stopped at tick 1 — with the fixes in the code and no way
        # for them to take effect. Silent state inheritance is not a convenience.
        print(
            f"error: {soak_dir} already holds a previous run.\n"
            "  A new run must not inherit its objective's state or its consent grant.\n"
            f"  Start clean:   --dir {soak_dir} --reset\n"
            f"  Or continue:   --resume {soak_dir}\n"
            "  Or use a fresh --dir.",
            file=sys.stderr,
        )
        return 2

    try:
        report = harness.run()
    except SoakDirtyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, default=str) if args.json else render_report(report))
    print(f"[soak] report: {harness.soak_dir / REPORT_NAME}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
