"""
B7 — approval resume / WorkflowStore mutation safety.

The Opus plan listed `_resume_blocked_workflow_task` as an open race: it built
a fresh `WorkflowStore` and was believed to load-modify-save outside any lock.
Audited against the committed tree, it does not. `WorkflowStore.locked()` is a
real `fcntl.flock`, and the resume takes it *before* loading the workflow.

So this file does not fix a race. It pins the invariant that makes the race
impossible, in the three ways it could regress:

  1. The workflow is loaded *inside* the lock, never before it. A resume built
     on a view taken outside the lock would be a lost update waiting for a
     concurrent scheduler tick.
  2. The lock is genuinely cross-process, not merely cross-thread.
  3. A scheduler ticking concurrently with an approval decision cannot clobber
     the resume — the task stays reopened.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from mr1.capability_policy import (
    CapabilityApprovalDecision,
    CapabilityApprovalStore,
    CapabilityRequest,
    PolicyEngine,
    ScopeContext,
    build_approval_request,
    maybe_route_approval_request,
    metadata_for_capability,
)
from mr1.messages import MessageStore
from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


REPO_ROOT = Path(__file__).resolve().parents[1]
CREATED_BY = Provenance(type="user", id="test")

SPEC = {
    "title": "Approval resume",
    "tasks": [
        {
            "label": "blocked",
            "title": "Needs approval",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "the task an approval unblocks",
        },
        {
            "label": "other",
            "title": "Unrelated work",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "keeps the scheduler writing to this workflow",
        },
    ],
}


@pytest.fixture
def runtime(tmp_path):
    store = WorkflowStore(root=tmp_path / "workflows")
    agents = PersistentAgentStore(root=tmp_path / "agents")
    return tmp_path, store, agents


def _block_task(store: WorkflowStore, workflow_id: str, label: str) -> str:
    """Put a task into the exact state an approval block leaves behind."""
    with store.locked():
        workflow = store.load_workflow(workflow_id)
        task = workflow.task_by_label(label)
        task.status = TaskStatus.BLOCKED
        task.last_error_type = "approval_required"
        task.last_error = "capability requires approval"
        store.save_workflow(workflow)
        return task.task_id


def _pending_approval(
    root: Path,
    approvals: CapabilityApprovalStore,
    agents: PersistentAgentStore,
    workflow_id: str,
    task_id: str,
) -> str:
    """Route a real risk-1.0 approval through the real policy engine."""
    messages = MessageStore(root=root / "messages", scoped_agent_store=agents)
    request = CapabilityRequest(
        actor_id=agents.root_agent_id,
        actor_type="mr1",
        actor_clearance=0.99,
        invocation_mode="workflow",
        capability_name="shell_command",
        args={"argv": ["echo", "hi"], "cwd": str(root)},
        scope=ScopeContext(allowed_roots=[root], workspace_root=root),
        workflow_id=workflow_id,
        task_id=task_id,
    )
    metadata = metadata_for_capability("shell_command", "tool")
    decision = PolicyEngine().evaluate(request, metadata)
    assert decision.status == "requires_approval"

    approval_id, _created = maybe_route_approval_request(
        build_approval_request(request, metadata, decision),
        approval_store=approvals,
        message_store=messages,
        scoped_agent_store=agents,
    )
    return approval_id


def _approve(approvals, agents, approval_id: str) -> None:
    approvals.apply_decision(
        approval_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval_id,
            decision="approved",
            decided_by=agents.root_agent_id,
            reason="ok",
            timestamp=approvals.clock.monotonic(),
            approval_scope="single_use",
        ),
        scoped_agent_store=agents,
    )


# ---------------------------------------------------------------------------
# 1. The workflow is loaded inside the lock
# ---------------------------------------------------------------------------


def test_the_resume_loads_the_workflow_after_taking_the_lock(runtime):
    """
    The invariant, asserted on ordering rather than on outcome.

    A resume that loaded the workflow before acquiring the lock would still
    *pass* an outcome test on an idle system, and lose the update the moment a
    scheduler ticked concurrently. So watch the call order directly.
    """
    root, store, agents = runtime
    scheduler = Scheduler(store, MockRunner(), auto_tick=False, scoped_agent_store=agents)
    workflow_id = scheduler.submit_workflow(SPEC, CREATED_BY)
    task_id = _block_task(store, workflow_id, "blocked")

    trace: list[str] = []
    real_locked = store.locked
    real_load = store.load_workflow
    real_save = store.save_workflow

    import contextlib

    @contextlib.contextmanager
    def traced_locked():
        trace.append("lock")
        with real_locked():
            yield
        trace.append("unlock")

    def traced_load(wid):
        trace.append("load")
        return real_load(wid)

    def traced_save(wf):
        trace.append("save")
        return real_save(wf)

    store.locked = traced_locked
    store.load_workflow = traced_load
    store.save_workflow = traced_save

    approvals = CapabilityApprovalStore(
        root / "capability_approvals",
        workflow_store=store,
    )
    approval_id = _pending_approval(root, approvals, agents, workflow_id, task_id)
    _approve(approvals, agents, approval_id)

    assert "lock" in trace, "the resume must take the store lock"
    first_lock = trace.index("lock")
    assert trace.index("load") > first_lock, "the workflow must be loaded INSIDE the lock"
    assert trace.index("save") > first_lock
    assert trace.index("save") < trace.index("unlock"), "and saved before releasing it"

    store.locked = real_locked
    store.load_workflow = real_load
    store.save_workflow = real_save

    workflow = store.load_workflow(workflow_id)
    assert workflow.tasks[task_id].status is not TaskStatus.BLOCKED


# ---------------------------------------------------------------------------
# 2. The lock is cross-process
# ---------------------------------------------------------------------------


HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from mr1.workflow_store import WorkflowStore

    store = WorkflowStore(root=Path(sys.argv[1]) / "workflows")
    hold_s = float(sys.argv[2])
    ready = Path(sys.argv[3])
    with store.locked():
        ready.write_text("held", encoding="utf-8")
        time.sleep(hold_s)
    """
)


def test_the_resume_window_excludes_another_process(runtime):
    """
    An in-process `threading.Lock` would pass every other test in this file.

    Only a real flock makes an approval decision in *this* process wait for a
    mutation held by *another* one — which is exactly the configuration B8
    makes routine: a service executing while a CLI approves.
    """
    root, store, agents = runtime
    scheduler = Scheduler(store, MockRunner(), auto_tick=False, scoped_agent_store=agents)
    workflow_id = scheduler.submit_workflow(SPEC, CREATED_BY)
    task_id = _block_task(store, workflow_id, "blocked")

    approvals = CapabilityApprovalStore(root / "capability_approvals")
    approval_id = _pending_approval(root, approvals, agents, workflow_id, task_id)

    script = root / "holder.py"
    script.write_text(HOLDER, encoding="utf-8")
    ready = root / "held.flag"
    hold_s = 1.5

    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    holder = subprocess.Popen(
        [sys.executable, str(script), str(root), str(hold_s), str(ready)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 20
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "the holder process never took the store lock"

        started = time.monotonic()
        _approve(approvals, agents, approval_id)
        waited = time.monotonic() - started
    finally:
        out, err = holder.communicate(timeout=30)
        assert holder.returncode == 0, err

    assert waited > 0.5, (
        f"the approval resume returned in {waited:.3f}s while another process held "
        "the store lock — the mutation window is not cross-process protected"
    )
    workflow = store.load_workflow(workflow_id)
    assert workflow.tasks[task_id].status is not TaskStatus.BLOCKED


# ---------------------------------------------------------------------------
# 3. A concurrent scheduler cannot clobber the resume
# ---------------------------------------------------------------------------


def test_a_concurrent_scheduler_tick_never_loses_the_resume(runtime):
    """
    The lost update the plan was worried about, driven for real.

    A scheduler ticking the same workflow is writing task state constantly. If
    either side loaded outside the lock, some iteration would save a workflow
    whose `blocked` task was still BLOCKED — silently un-approving it, with the
    human's decision recorded as applied. Twenty racing rounds; zero tolerated.
    """
    root, store, agents = runtime
    approvals = CapabilityApprovalStore(
        root / "capability_approvals",
        workflow_store=store,
    )

    for _ in range(20):
        scheduler = Scheduler(store, MockRunner(), auto_tick=False, scoped_agent_store=agents)
        workflow_id = scheduler.submit_workflow(SPEC, CREATED_BY)
        task_id = _block_task(store, workflow_id, "blocked")
        approval_id = _pending_approval(root, approvals, agents, workflow_id, task_id)

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def tick_hard():
            try:
                barrier.wait(timeout=10)
                for _ in range(12):
                    scheduler.tick()
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        def approve():
            try:
                barrier.wait(timeout=10)
                _approve(approvals, agents, approval_id)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=tick_hard), threading.Thread(target=approve)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors, f"concurrent access raised: {errors[0]!r}"

        workflow = store.load_workflow(workflow_id)
        assert workflow is not None, "the workflow record must survive the race intact"
        task = workflow.tasks[task_id]
        assert task.status is not TaskStatus.BLOCKED, (
            "the scheduler wrote back a stale workflow and reverted an approved task "
            "to BLOCKED — the approval was recorded but silently did nothing"
        )
        assert task.last_error_type != "approval_required"
        scheduler.shutdown()
