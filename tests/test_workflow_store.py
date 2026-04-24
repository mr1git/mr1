"""Tests for mr1.workflow_store"""

import json
import threading

import pytest

from mr1.dataflow import ResolvedTaskInput, TaskOutput
from mr1.workflow_models import (
    Provenance,
    Task,
    TaskStatus,
    Workflow,
    WorkflowEvent,
    WorkflowStatus,
)
from mr1.workflow_store import WorkflowStore


def _make_workflow(wf_id: str = "wf-test") -> Workflow:
    t = Task(
        task_id="tk-1",
        workflow_id=wf_id,
        label="a",
        title="Task A",
        task_kind="agent",
        agent_type="kazi",
        prompt="Say hi",
        status=TaskStatus.READY,
    )
    return Workflow(
        workflow_id=wf_id,
        title="Test workflow",
        status=WorkflowStatus.PENDING,
        created_by=Provenance(type="user", id="cli"),
        tasks={t.task_id: t},
        label_to_task_id={"a": t.task_id},
    )


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


class TestSaveAndLoad:
    def test_save_then_load_roundtrip(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)
        loaded = store.load_workflow(wf.workflow_id)
        assert loaded is not None
        assert loaded.to_dict() == wf.to_dict()

    def test_load_missing_returns_none(self, store):
        assert store.load_workflow("does-not-exist") is None

    def test_save_creates_directory(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)
        assert store.workflow_dir(wf.workflow_id).is_dir()
        assert store.workflow_json_path(wf.workflow_id).exists()

    def test_reload_across_store_instances(self, store, tmp_path):
        wf = _make_workflow()
        store.save_workflow(wf)
        # Fresh store instance over the same root.
        store2 = WorkflowStore(root=tmp_path / "workflows")
        loaded = store2.load_workflow(wf.workflow_id)
        assert loaded is not None
        assert loaded.title == wf.title
        assert list(loaded.tasks) == list(wf.tasks)

    def test_atomic_write_no_temp_artifact(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)
        tmp_path = store.workflow_json_path(wf.workflow_id).with_suffix(".json.tmp")
        assert not tmp_path.exists()


class TestListWorkflows:
    def test_empty(self, store):
        assert store.list_workflows() == []

    def test_returns_saved_workflows(self, store):
        wf_a = _make_workflow("wf-a")
        wf_b = _make_workflow("wf-b")
        store.save_workflow(wf_a)
        store.save_workflow(wf_b)
        ids = {wf.workflow_id for wf in store.list_workflows()}
        assert ids == {"wf-a", "wf-b"}

    def test_skips_dirs_without_workflow_json(self, store, tmp_path):
        wf = _make_workflow("wf-real")
        store.save_workflow(wf)
        (store.root / "orphan-dir").mkdir()
        (store.root / "orphan-dir" / "stray.txt").write_text("noise")
        ids = {wf.workflow_id for wf in store.list_workflows()}
        assert ids == {"wf-real"}

    def test_skips_non_directory_entries(self, store):
        (store.root / "not-a-dir.json").write_text("{}")
        assert store.list_workflows() == []


class TestEvents:
    def test_append_then_load(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)
        ev = WorkflowEvent.new(
            event_type="task_started",
            workflow_id=wf.workflow_id,
            task_id="tk-1",
            agent_id="scheduler",
        )
        store.append_event(ev)
        events = store.load_events(wf.workflow_id)
        assert len(events) == 1
        assert events[0].event_type == "task_started"

    def test_events_jsonl_append_only(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)
        for i in range(3):
            store.append_event(WorkflowEvent.new(
                event_type=f"evt_{i}",
                workflow_id=wf.workflow_id,
                task_id="tk-1",
            ))
        events = store.load_events(wf.workflow_id)
        assert [e.event_type for e in events] == ["evt_0", "evt_1", "evt_2"]

    def test_events_survive_store_reopen(self, store, tmp_path):
        wf = _make_workflow()
        store.save_workflow(wf)
        store.append_event(WorkflowEvent.new(
            event_type="workflow_submitted",
            workflow_id=wf.workflow_id,
        ))
        store2 = WorkflowStore(root=tmp_path / "workflows")
        events = store2.load_events(wf.workflow_id)
        assert len(events) == 1
        assert events[0].event_type == "workflow_submitted"

    def test_filter_by_task_id(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)
        store.append_event(WorkflowEvent.new(
            event_type="task_started",
            workflow_id=wf.workflow_id, task_id="tk-1",
        ))
        store.append_event(WorkflowEvent.new(
            event_type="task_started",
            workflow_id=wf.workflow_id, task_id="tk-2",
        ))
        filtered = store.load_events(wf.workflow_id, task_id="tk-2")
        assert [e.task_id for e in filtered] == ["tk-2"]

    def test_filter_by_time_range(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)
        # Manually craft events with fixed timestamps.
        evs = [
            WorkflowEvent(
                timestamp="2026-01-01T00:00:00+00:00",
                event_type="early",
                workflow_id=wf.workflow_id,
            ),
            WorkflowEvent(
                timestamp="2026-06-01T00:00:00+00:00",
                event_type="middle",
                workflow_id=wf.workflow_id,
            ),
            WorkflowEvent(
                timestamp="2026-12-01T00:00:00+00:00",
                event_type="late",
                workflow_id=wf.workflow_id,
            ),
        ]
        for e in evs:
            store.append_event(e)
        mid = store.load_events(
            wf.workflow_id,
            since="2026-03-01T00:00:00+00:00",
            until="2026-09-01T00:00:00+00:00",
        )
        assert [e.event_type for e in mid] == ["middle"]


class TestResults:
    def test_write_then_read(self, store):
        path = store.write_result("wf-1", "tk-1", {"status": "succeeded", "output": "hi"})
        assert path.exists()
        assert store.read_result("wf-1", "tk-1") == {"status": "succeeded", "output": "hi"}

    def test_read_missing(self, store):
        assert store.read_result("wf-1", "tk-1") is None

    def test_write_and_load_task_output(self, store):
        output = TaskOutput(
            task_id="tk-1",
            workflow_id="wf-1",
            status="succeeded",
            summary="done",
            text="hello",
        )
        path = store.write_task_output("wf-1", "tk-1", output)
        assert path.exists()
        loaded = store.load_task_output("wf-1", "tk-1")
        assert loaded is not None
        assert loaded.to_dict() == output.to_dict()

    def test_write_and_load_task_inputs(self, store):
        inputs = [
            ResolvedTaskInput(
                name="producer_text",
                source="produce.result.text",
                resolved_task_id="tk-1",
                resolved_type="text",
                value="hello",
            )
        ]
        path = store.write_task_inputs("wf-1", "tk-2", inputs)
        assert path.exists()
        loaded = store.load_task_inputs("wf-1", "tk-2")
        assert loaded is not None
        assert [item.to_dict() for item in loaded] == [item.to_dict() for item in inputs]

    def test_write_materialized_prompt(self, store):
        path = store.write_materialized_prompt("wf-1", "tk-2", "prompt body")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "prompt body"

    def test_task_artifacts_dir_created(self, store):
        path = store.task_artifacts_dir("wf-1", "tk-1")
        assert path.exists()
        assert path.is_dir()


class TestLockingAtomicity:
    def test_concurrent_saves_do_not_corrupt(self, store):
        wf = _make_workflow()
        store.save_workflow(wf)

        errors: list[BaseException] = []

        def writer(i: int):
            try:
                copy = _make_workflow()
                copy.title = f"title-{i}"
                for _ in range(20):
                    store.save_workflow(copy)
                    store.append_event(WorkflowEvent.new(
                        event_type=f"evt-{i}",
                        workflow_id=copy.workflow_id,
                    ))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

        # File must still be valid JSON after all the concurrent writes.
        with open(store.workflow_json_path(wf.workflow_id), "r") as f:
            assert json.load(f)["workflow_id"] == wf.workflow_id

    def test_locked_context_allows_compound_mutation(self, store):
        wf = _make_workflow()
        with store.locked():
            store.save_workflow(wf)
            store.append_event(WorkflowEvent.new(
                event_type="workflow_submitted",
                workflow_id=wf.workflow_id,
            ))
        assert store.load_workflow(wf.workflow_id) is not None
        assert len(store.load_events(wf.workflow_id)) == 1
