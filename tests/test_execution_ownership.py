"""
B8 — cross-process execution ownership.

The claim under test is not "a lock file exists". It is: two *real processes*
pointed at one runtime root cannot both launch the same task, and the one that
loses knows it lost rather than failing silently.

So the central test spawns two interpreters. Everything else here pins the
edges around it: promotion when the owner dies, a stale owner file that must
never read as ownership, and the unenforced default that keeps every embedded
use of `Scheduler` behaving exactly as it did before.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mr1.autonomy.ownership import (
    ROLE_REPL,
    ROLE_SERVICE,
    ExecutionOwnership,
    execution_owner,
    read_owner_file,
)
from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _child_env() -> dict[str, str]:
    """A subprocess gets the repo on its path, not the throwaway script's dir."""
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    return env

SPEC = {
    "title": "Ownership probe",
    "tasks": [
        {
            "label": "only",
            "title": "The one task",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "do the thing exactly once",
        }
    ],
}


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def test_acquire_is_exclusive_and_idempotent(tmp_path):
    first = ExecutionOwnership(tmp_path, role=ROLE_SERVICE)
    assert first.acquire() is True
    assert first.acquire() is True, "acquire must be idempotent for the holder"
    assert first.owned is True

    # A separate open() conflicts even inside one process — flock binds to the
    # open file description, not to the PID. This is the property the whole
    # checkpoint rests on.
    second = ExecutionOwnership(tmp_path, role=ROLE_REPL)
    assert second.acquire() is False
    assert second.owned is False
    assert second.is_held_elsewhere() is True

    first.release()
    assert second.acquire() is True
    second.release()


def test_release_frees_the_root_for_the_next_holder(tmp_path):
    with ExecutionOwnership(tmp_path, role=ROLE_SERVICE):
        assert ExecutionOwnership(tmp_path).is_held() is True
    assert ExecutionOwnership(tmp_path).is_held() is False
    assert execution_owner(tmp_path) is None


def test_current_owner_reports_the_holder(tmp_path):
    owner_ship = ExecutionOwnership(tmp_path, role=ROLE_SERVICE)
    owner_ship.acquire()
    try:
        seen = execution_owner(tmp_path)
        assert seen is not None
        assert seen.role == ROLE_SERVICE
        assert seen.pid > 0
        assert seen.acquired_at
    finally:
        owner_ship.release()


def test_a_stale_owner_file_is_not_ownership(tmp_path):
    """
    The failure mode a PID file has and flock does not.

    A killed owner leaves `execution_owner.json` behind. If that file were the
    authority, the runtime root would be permanently unusable by anyone.
    """
    (tmp_path / "execution_owner.json").write_text(
        json.dumps({"pid": 999_999, "host": "gone", "role": "service", "acquired_at": "x"}),
        encoding="utf-8",
    )
    (tmp_path / "execution.lock").write_text("", encoding="utf-8")

    assert read_owner_file(tmp_path) is not None, "the label is there"
    assert execution_owner(tmp_path) is None, "but nobody holds the lock"

    fresh = ExecutionOwnership(tmp_path, role=ROLE_SERVICE)
    assert fresh.acquire() is True
    fresh.release()


def test_describe_is_the_status_surface(tmp_path):
    ownership = ExecutionOwnership(tmp_path, role=ROLE_SERVICE)
    assert ownership.describe()["held"] is False
    assert ownership.describe()["owned_by_me"] is False
    ownership.acquire()
    try:
        described = ownership.describe()
        assert described["owned_by_me"] is True
        assert described["held"] is True
        assert described["owner"]["role"] == ROLE_SERVICE
    finally:
        ownership.release()


# ---------------------------------------------------------------------------
# The scheduler seam
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


CREATED_BY = Provenance(type="user", id="test")


def test_a_scheduler_without_ownership_still_executes(store, tmp_path):
    """The default must not change: embedded and test schedulers own nothing."""
    runner = MockRunner()
    scheduler = Scheduler(store, runner, auto_tick=False)
    scheduler.submit_workflow(SPEC, CREATED_BY)
    scheduler.tick()

    assert scheduler.has_execution_authority() is True
    assert len(runner.started_task_ids) == 1
    assert scheduler.metrics()["delegated_ticks"] == 0
    assert scheduler.metrics()["execution_owned"] is True


def test_a_follower_scheduler_launches_nothing(store, tmp_path):
    """The core in-process proof of the same property the subprocess test proves."""
    incumbent = ExecutionOwnership(tmp_path, role=ROLE_SERVICE)
    assert incumbent.acquire() is True

    runner = MockRunner()
    follower = Scheduler(
        store,
        runner,
        auto_tick=False,
        execution_ownership=ExecutionOwnership(tmp_path, role=ROLE_REPL),
    )
    workflow_id = follower.submit_workflow(SPEC, CREATED_BY)

    for _ in range(5):
        follower.tick()

    assert runner.started_task_ids == [], "a follower must never launch a task"
    assert follower.metrics()["delegated_ticks"] == 5, "and must know it was refused"
    assert follower.metrics()["execution_owned"] is False

    # Submission and inspection stay available to a non-owner — it is execution
    # that is exclusive, not access.
    workflow = follower.get_workflow(workflow_id)
    assert workflow is not None
    assert [task.status for task in workflow.tasks.values()] == [TaskStatus.CREATED]

    incumbent.release()


def test_a_follower_promotes_itself_when_the_owner_leaves(store, tmp_path):
    incumbent = ExecutionOwnership(tmp_path, role=ROLE_SERVICE)
    incumbent.acquire()

    runner = MockRunner()
    scheduler = Scheduler(
        store,
        runner,
        auto_tick=False,
        execution_ownership=ExecutionOwnership(tmp_path, role=ROLE_REPL),
    )
    scheduler.submit_workflow(SPEC, CREATED_BY)
    scheduler.tick()
    assert runner.started_task_ids == []

    incumbent.release()  # the service exits

    scheduler.tick()
    assert len(runner.started_task_ids) == 1, "authority must never stay orphaned"
    assert scheduler.metrics()["execution_owned"] is True


def test_scheduler_shutdown_hands_the_root_back(store, tmp_path):
    ownership = ExecutionOwnership(tmp_path, role=ROLE_SERVICE)
    scheduler = Scheduler(store, MockRunner(), auto_tick=False, execution_ownership=ownership)
    scheduler.tick()
    assert ownership.owned is True

    scheduler.shutdown()
    assert ownership.owned is False
    assert execution_owner(tmp_path) is None


# ---------------------------------------------------------------------------
# The supervisor seam
# ---------------------------------------------------------------------------


def test_a_supervisor_without_authority_still_observes_but_never_executes(tmp_path):
    """
    A supervisor that starts while a REPL holds the root is not broken.

    It still observes, sweeps, and reconciles — it just does not launch. That
    keeps `mr1 serve` useful next to an open REPL instead of making the two
    mutually exclusive, and it means the service takes over silently and
    correctly the moment the REPL exits.
    """
    from mr1.autonomy.service import Supervisor

    runtime_root = tmp_path / "runtime"
    incumbent = ExecutionOwnership(runtime_root, role=ROLE_REPL)
    assert incumbent.acquire() is True

    runner = MockRunner()
    supervisor = Supervisor(
        runtime_root,
        runner=runner,
        auto_scheduler_tick=False,
        execution_ownership=ExecutionOwnership(runtime_root, role=ROLE_SERVICE),
    )
    supervisor.scheduler.submit_workflow(SPEC, CREATED_BY)

    supervisor.tick()
    supervisor.scheduler.tick()
    assert runner.started_task_ids == []
    assert supervisor.gauges()["execution_owned"] is False

    incumbent.release()

    supervisor.scheduler.tick()
    assert len(runner.started_task_ids) == 1
    assert supervisor.gauges()["execution_owned"] is True
    supervisor.shutdown()


# ---------------------------------------------------------------------------
# The proof: two real processes
# ---------------------------------------------------------------------------


CHILD = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path

    from mr1.autonomy.ownership import ExecutionOwnership
    from mr1.kazi_runner import MockRunner
    from mr1.scheduler import Scheduler
    from mr1.workflow_store import WorkflowStore

    runtime_root = Path(sys.argv[1])
    role = sys.argv[2]
    launch_log = Path(sys.argv[3])
    deadline = float(sys.argv[4])

    def record(task):
        # One short append per launch: atomic on POSIX, so the log is a true
        # count of how many processes started this task.
        with open(launch_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "role": role,
                                     "task_id": task.task_id}) + "\\n")

    store = WorkflowStore(root=runtime_root / "workflows")
    ownership = ExecutionOwnership(runtime_root, role=role)
    scheduler = Scheduler(
        store,
        MockRunner(on_start=record),
        auto_tick=False,
        agent_id=role,
        execution_ownership=ownership,
    )

    while time.time() < deadline:
        scheduler.tick()
        time.sleep(0.02)

    metrics = scheduler.metrics()
    print(json.dumps({
        "role": role,
        "pid": os.getpid(),
        "owned": ownership.owned,
        "delegated_ticks": metrics["delegated_ticks"],
        "tick_count": metrics["tick_count"],
    }))
    """
)


def test_two_processes_cannot_execute_the_same_task(tmp_path):
    """
    Two independent interpreters, one runtime root, one ready task.

    Both loop on `tick()` for the same window. Exactly one may start the task;
    the other must record delegated ticks — proof it was live and trying, not
    merely late.
    """
    runtime_root = tmp_path / "runtime"
    store = WorkflowStore(root=runtime_root / "workflows")
    Scheduler(store, MockRunner(), auto_tick=False).submit_workflow(SPEC, CREATED_BY)

    child_script = tmp_path / "child.py"
    child_script.write_text(CHILD, encoding="utf-8")
    launch_log = tmp_path / "launches.jsonl"

    import time

    deadline = time.time() + 3.0
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(child_script),
                str(runtime_root),
                role,
                str(launch_log),
                str(deadline),
            ],
            cwd=str(REPO_ROOT),
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for role in (ROLE_SERVICE, ROLE_REPL)
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=45)
        assert proc.returncode == 0, f"child failed:\n{err}"
        results.append(json.loads(out.strip().splitlines()[-1]))

    launches = [
        json.loads(line)
        for line in launch_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if launch_log.exists() else []

    assert len(launches) == 1, f"the task must be launched exactly once, got {launches}"

    owners = [item for item in results if item["owned"]]
    followers = [item for item in results if not item["owned"]]
    assert len(owners) == 1, f"exactly one process may own the root: {results}"
    assert len(followers) == 1
    assert launches[0]["pid"] == owners[0]["pid"], "the owner is the one that executed"

    follower = followers[0]
    assert follower["delegated_ticks"] > 0, (
        "the loser must have actually attempted ticks and been refused — "
        "a silently idle process would prove nothing"
    )
    assert follower["tick_count"] == 0, "a delegated tick does no scheduler work"


def test_the_owner_releases_the_root_on_process_exit(tmp_path):
    """Ownership dies with the process. No stale lock, no manual cleanup."""
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True)

    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from mr1.autonomy.ownership import ExecutionOwnership

        ownership = ExecutionOwnership(Path(sys.argv[1]), role="service")
        assert ownership.acquire()
        # Exit *without* releasing. The kernel must do it for us.
        """
    )
    child = tmp_path / "grabber.py"
    child.write_text(script, encoding="utf-8")

    done = subprocess.run(
        [sys.executable, str(child), str(runtime_root)],
        cwd=str(REPO_ROOT),
        env=_child_env(),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr

    assert execution_owner(runtime_root) is None
    reclaimed = ExecutionOwnership(runtime_root, role=ROLE_REPL)
    assert reclaimed.acquire() is True
    reclaimed.release()
