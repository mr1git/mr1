"""
Tests for the scheduler's DAG progression on the happy path.

Uses the MockRunner so we never actually spawn subprocesses. The
scheduler is built with `auto_tick=False` so tests can drive `tick()`
deterministically.
"""

from pathlib import Path

import pytest

from mr1 import workflow_events as ev
from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus, WorkflowStatus
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "Three-task DAG",
    "tasks": [
        {
            "label": "a",
            "title": "Task A",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "do a",
        },
        {
            "label": "b",
            "title": "Task B",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "do b",
            "depends_on": ["a"],
        },
        {
            "label": "c",
            "title": "Task C",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "do c",
            "depends_on": ["a"],
        },
    ],
}


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def runner():
    return MockRunner()


@pytest.fixture
def scheduler(store, runner):
    s = Scheduler(store, runner, auto_tick=False, concurrency=4)
    yield s
    s.shutdown()


def _task_by_label(wf, label):
    return wf.task_by_label(label)


class TestDagProgression:
    def test_initial_statuses_after_submit(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))

        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.PENDING
        for task in wf.tasks.values():
            assert task.status is TaskStatus.CREATED

    def test_first_tick_promotes_created_and_launches_a(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        a = _task_by_label(wf, "a")
        b = _task_by_label(wf, "b")
        c = _task_by_label(wf, "c")

        assert a.status is TaskStatus.RUNNING
        assert b.status is TaskStatus.WAITING
        assert c.status is TaskStatus.WAITING
        assert wf.status is WorkflowStatus.RUNNING
        assert runner.started_task_ids == [a.task_id]

    def test_a_succeeds_promotes_b_and_c_and_runs_them(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        a_id = store.load_workflow(wf_id).label_to_task_id["a"]
        runner.complete(a_id, RunStatus.SUCCEEDED, summary="A done")

        scheduler.tick()
        wf = store.load_workflow(wf_id)

        a = _task_by_label(wf, "a")
        b = _task_by_label(wf, "b")
        c = _task_by_label(wf, "c")

        assert a.status is TaskStatus.SUCCEEDED
        assert a.result_summary == "A done"
        assert {b.status, c.status} == {TaskStatus.RUNNING}
        assert wf.status is WorkflowStatus.RUNNING

    def test_full_workflow_succeeds(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        runner.complete(wf.label_to_task_id["a"], RunStatus.SUCCEEDED)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        runner.complete(wf.label_to_task_id["b"], RunStatus.SUCCEEDED)
        runner.complete(wf.label_to_task_id["c"], RunStatus.SUCCEEDED)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.SUCCEEDED
        assert wf.finished_at is not None
        for t in wf.tasks.values():
            assert t.status is TaskStatus.SUCCEEDED
            assert t.result_path is not None

    def test_events_recorded_in_order(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        runner.complete(wf.label_to_task_id["a"], RunStatus.SUCCEEDED)
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        runner.complete(wf.label_to_task_id["b"], RunStatus.SUCCEEDED)
        runner.complete(wf.label_to_task_id["c"], RunStatus.SUCCEEDED)
        scheduler.tick()

        events = store.load_events(wf_id)
        types = [e.event_type for e in events]
        assert types[0] == ev.WORKFLOW_SUBMITTED
        assert ev.TASK_STARTED in types
        assert ev.TASK_SUCCEEDED in types
        assert ev.WORKFLOW_SUCCEEDED == types[-1]

    def test_result_json_written_for_succeeded_task(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        runner.complete(a_id, RunStatus.SUCCEEDED, summary="hello")
        scheduler.tick()

        payload = store.read_result(wf_id, a_id)
        assert payload is not None
        assert payload.get("status") == "succeeded"
        assert payload.get("summary") == "hello"

    def test_output_json_written_for_succeeded_task(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        runner.complete(a_id, RunStatus.SUCCEEDED, summary="hello")
        scheduler.tick()

        output = store.load_task_output(wf_id, a_id)
        assert output is not None
        assert output.status == "succeeded"
        assert output.text == "hello"
        assert output.summary == "hello"

    def test_concurrency_cap_is_respected(self, store):
        runner = MockRunner()
        sched = Scheduler(store, runner, auto_tick=False, concurrency=1)
        try:
            # Three independent tasks (no deps); concurrency=1 means only
            # one should be RUNNING after the first tick.
            spec = {
                "title": "parallel",
                "tasks": [
                    {"label": "a", "title": "a", "task_kind": "agent",
                     "agent_type": "kazi", "prompt": "x"},
                    {"label": "b", "title": "b", "task_kind": "agent",
                     "agent_type": "kazi", "prompt": "x"},
                    {"label": "c", "title": "c", "task_kind": "agent",
                     "agent_type": "kazi", "prompt": "x"},
                ],
            }
            wf_id = sched.submit_workflow(spec, Provenance(type="agent", id="MR1"))
            sched.tick()
            wf = store.load_workflow(wf_id)
            running = [t for t in wf.tasks.values() if t.status is TaskStatus.RUNNING]
            ready = [t for t in wf.tasks.values() if t.status is TaskStatus.READY]
            assert len(running) == 1
            assert len(ready) == 2
        finally:
            sched.shutdown()


class TestDataflowDag:
    def test_downstream_receives_materialized_prompt_and_raw_prompt_is_preserved(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        started_prompts: dict[str, str] = {}

        def on_start(task):
            started_prompts[task.label] = task.prompt

        runner = MockRunner(on_start=on_start)
        sched = Scheduler(store, runner, auto_tick=False, concurrency=4)
        spec = {
            "title": "Dataflow",
            "tasks": [
                {
                    "label": "a",
                    "title": "Produce",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Produce text",
                },
                {
                    "label": "b",
                    "title": "Consume",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Summarize the upstream text.",
                    "depends_on": ["a"],
                    "inputs": [{"name": "producer_text", "from": "a.result.text"}],
                },
            ],
        }
        try:
            wf_id = sched.submit_workflow(spec, Provenance(type="agent", id="MR1"))
            sched.tick()
            wf = store.load_workflow(wf_id)
            runner.complete(wf.label_to_task_id["a"], RunStatus.SUCCEEDED, summary="hello world")
            sched.tick()

            wf = store.load_workflow(wf_id)
            b = wf.task_by_label("b")
            assert b.status is TaskStatus.RUNNING
            assert b.prompt == "Summarize the upstream text."
            assert b.inputs_path is not None
            assert b.materialized_prompt_path is not None
            assert "hello world" in Path(b.materialized_prompt_path).read_text(encoding="utf-8")
            assert started_prompts["b"] == Path(b.materialized_prompt_path).read_text(encoding="utf-8")

            resolved_inputs = store.load_task_inputs(wf_id, b.task_id)
            assert resolved_inputs is not None
            assert resolved_inputs[0].value == "hello world"
        finally:
            sched.shutdown()

    def test_missing_input_fails_before_launch_and_emits_event(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = MockRunner()
        sched = Scheduler(store, runner, auto_tick=False, concurrency=4)
        spec = {
            "title": "Missing input",
            "tasks": [
                {
                    "label": "a",
                    "title": "Produce",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Produce text",
                },
                {
                    "label": "b",
                    "title": "Consume",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Consume metrics",
                    "depends_on": ["a"],
                    "inputs": [{"name": "accuracy", "from": "a.result.metrics.accuracy"}],
                },
            ],
        }
        try:
            wf_id = sched.submit_workflow(spec, Provenance(type="agent", id="MR1"))
            sched.tick()
            wf = store.load_workflow(wf_id)
            runner.complete(wf.label_to_task_id["a"], RunStatus.SUCCEEDED, summary="hello world")
            sched.tick()

            wf = store.load_workflow(wf_id)
            b = wf.task_by_label("b")
            assert b.status is TaskStatus.FAILED
            assert b.dataflow_error is not None
            assert runner.started_task_ids == [wf.label_to_task_id["a"]]
            event_types = [event.event_type for event in store.load_events(wf_id)]
            assert ev.INPUT_RESOLUTION_FAILED in event_types
        finally:
            sched.shutdown()

    def test_duplicate_artifact_names_fail_task(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        runner = MockRunner()
        sched = Scheduler(store, runner, auto_tick=False, concurrency=4)
        try:
            wf_id = sched.submit_workflow(
                {
                    "title": "Artifacts",
                    "tasks": [
                        {
                            "label": "a",
                            "title": "Produce",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "prompt": "Produce artifact",
                        }
                    ],
                },
                Provenance(type="agent", id="MR1"),
            )
            sched.tick()
            wf = store.load_workflow(wf_id)
            a_id = wf.label_to_task_id["a"]
            runner.complete(
                a_id,
                RunStatus.SUCCEEDED,
                summary="done",
                result_payload={
                    "status": "succeeded",
                    "summary": "done",
                    "artifacts": [
                        {"name": "report", "kind": "json", "path": "/tmp/r1.json"},
                        {"name": "report", "kind": "json", "path": "/tmp/r2.json"},
                    ],
                },
            )
            sched.tick()
            wf = store.load_workflow(wf_id)
            assert wf.task_by_label("a").status is TaskStatus.FAILED
            assert "duplicate artifact name" in (wf.task_by_label("a").dataflow_error or "")
        finally:
            sched.shutdown()
