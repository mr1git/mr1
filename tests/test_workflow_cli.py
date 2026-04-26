"""
Tests for the deterministic workflow CLI.

These confirm that:
  * `submit` writes a workflow directory without starting a scheduler,
  * read commands load from the store and produce deterministic output,
  * invalid specs exit non-zero and leave the store empty.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.dataflow import Artifact, ResolvedTaskInput, TaskOutput
from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "CLI-submitted workflow",
    "tasks": [
        {"label": "a", "title": "A", "task_kind": "agent",
         "agent_type": "kazi", "prompt": "x"},
        {"label": "b", "title": "B", "task_kind": "agent",
         "agent_type": "kazi", "prompt": "x", "depends_on": ["a"]},
    ],
}


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


class TestSubmit:
    def test_writes_workflow_to_disk_and_prints_id(
        self, tmp_path, store, capsys
    ):
        path = _write_spec(tmp_path, SPEC)
        rc = workflow_cli.main(["submit", str(path)], store=store)
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert out.startswith("wf-")

        workflows = store.list_workflows()
        assert len(workflows) == 1
        wf = workflows[0]
        assert wf.workflow_id == out
        for task in wf.tasks.values():
            assert task.status is TaskStatus.CREATED

    def test_invalid_spec_exits_non_zero(self, tmp_path, store, capsys):
        bad = {"tasks": [
            {"label": "a", "task_kind": "agent", "agent_type": "kazi",
             "depends_on": ["b"]},
            {"label": "b", "task_kind": "agent", "agent_type": "kazi",
             "depends_on": ["a"]},
        ]}
        path = _write_spec(tmp_path, bad)
        rc = workflow_cli.main(["submit", str(path)], store=store)
        assert rc == 2
        assert store.list_workflows() == []
        err = capsys.readouterr().err
        assert "cycle" in err

    def test_missing_file_exits_non_zero(self, tmp_path, store, capsys):
        rc = workflow_cli.main(
            ["submit", str(tmp_path / "nope.json")], store=store
        )
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_malformed_json_exits_non_zero(self, tmp_path, store, capsys):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        rc = workflow_cli.main(["submit", str(path)], store=store)
        assert rc == 2
        assert "invalid JSON" in capsys.readouterr().err


class TestReadCommands:
    def test_workflows_empty(self, store, capsys):
        rc = workflow_cli.main(["workflows"], store=store)
        assert rc == 0
        assert "No workflows." in capsys.readouterr().out

    def test_workflows_lists_after_submit(self, tmp_path, store, capsys):
        path = _write_spec(tmp_path, SPEC)
        workflow_cli.main(["submit", str(path)], store=store)
        capsys.readouterr()  # reset buffer

        rc = workflow_cli.main(["workflows"], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "WORKFLOW_ID" in out
        assert "pending" in out
        assert "CLI-submitted" in out

    def test_workflow_detail(self, tmp_path, store, capsys):
        path = _write_spec(tmp_path, SPEC)
        workflow_cli.main(["submit", str(path)], store=store)
        wf_id = capsys.readouterr().out.strip()

        rc = workflow_cli.main(["workflow", wf_id], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert wf_id in out
        assert "a" in out and "b" in out
        assert "created" in out  # task status column

    def test_workflow_not_found(self, store, capsys):
        rc = workflow_cli.main(["workflow", "wf-missing"], store=store)
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_task_detail(self, tmp_path, store, capsys):
        path = _write_spec(tmp_path, SPEC)
        workflow_cli.main(["submit", str(path)], store=store)
        wf_id = capsys.readouterr().out.strip()

        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]

        rc = workflow_cli.main(["task", a_id], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert a_id in out
        assert "label:" in out
        assert "status:" in out
        assert "dependency_policy:" in out
        assert "run_if:" in out
        assert "condition_result:" in out
        assert "skip_reason:" in out

    def test_task_not_found(self, store, capsys):
        rc = workflow_cli.main(["task", "tk-nope"], store=store)
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_jobs_empty(self, store, capsys):
        rc = workflow_cli.main(["jobs"], store=store)
        assert rc == 0
        assert "No live tasks." in capsys.readouterr().out

    def test_events_shows_submitted(self, tmp_path, store, capsys):
        path = _write_spec(tmp_path, SPEC)
        workflow_cli.main(["submit", str(path)], store=store)
        wf_id = capsys.readouterr().out.strip()

        rc = workflow_cli.main(["events", wf_id], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "workflow_submitted" in out

    def test_events_workflow_not_found(self, store, capsys):
        rc = workflow_cli.main(["events", "wf-missing"], store=store)
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_result_inputs_and_artifacts_commands(self, tmp_path, store, capsys):
        path = _write_spec(tmp_path, SPEC)
        workflow_cli.main(["submit", str(path)], store=store)
        wf_id = capsys.readouterr().out.strip()
        wf = store.load_workflow(wf_id)
        a = wf.task_by_label("a")
        b = wf.task_by_label("b")
        store.write_task_output(
            wf_id,
            a.task_id,
            TaskOutput(
                task_id=a.task_id,
                workflow_id=wf_id,
                status="succeeded",
                summary="done",
                text="hello world",
                artifacts=[
                    Artifact(
                        artifact_id="art-1",
                        workflow_id=wf_id,
                        task_id=a.task_id,
                        name="report",
                        kind="json",
                        path="/tmp/report.json",
                    )
                ],
            ),
        )
        store.write_task_inputs(
            wf_id,
            b.task_id,
            [
                ResolvedTaskInput(
                    name="producer_text",
                    source="a.result.text",
                    resolved_task_id=a.task_id,
                    resolved_type="text",
                    value="hello world",
                )
            ],
        )
        a.artifacts = [
            Artifact(
                artifact_id="art-1",
                workflow_id=wf_id,
                task_id=a.task_id,
                name="report",
                kind="json",
                path="/tmp/report.json",
            )
        ]
        store.save_workflow(wf)

        rc = workflow_cli.main(["result", a.task_id], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "summary:" in out
        assert "hello world" in out

        rc = workflow_cli.main(["inputs", b.task_id], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "producer_text" in out
        assert "a.result.text" in out

        rc = workflow_cli.main(["artifacts", wf_id], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "report" in out

    def test_schema_json_output(self, store, capsys):
        rc = workflow_cli.main(["schema", "--json"], store=store)

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"workflow", "task", "inputs", "refs", "conditions", "task-kinds"}

    def test_schema_inputs_json_output(self, store, capsys):
        rc = workflow_cli.main(["schema", "inputs", "--json"], store=store)

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["item_shape"] == {
            "name": "string",
            "from": "<label>.<reference>",
        }

    def test_schema_refs_text_output(self, store, capsys):
        rc = workflow_cli.main(["schema", "refs"], store=store)

        assert rc == 0
        out = capsys.readouterr().out
        assert '"supported_patterns"' in out
        assert "<label>.result.text" in out


class TestReplaceWorkflow:
    def test_replace_workflow_does_not_autorun_without_r_flag(self, tmp_path, store, capsys):
        scheduler = Scheduler(store, MockRunner(), auto_tick=False)
        try:
            wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            a_id = wf.label_to_task_id["a"]
            scheduler._runner.complete(a_id, RunStatus.FAILED, error="boom")
            scheduler.tick()

            fragment = _write_spec(tmp_path, {
                "tasks": [{
                    "label": "a",
                    "title": "A2",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "replacement",
                }]
            })
            rc = workflow_cli.main(
                ["replace-workflow", wf_id, "a", str(fragment)],
                store=store,
            )
            assert rc == 0
            wf = store.load_workflow(wf_id)
            assert wf.task_by_label("a").status is TaskStatus.READY
        finally:
            scheduler.shutdown()

    def test_replace_workflow_r_flag_autoruns(self, tmp_path, store, capsys):
        scheduler = Scheduler(store, MockRunner(), auto_tick=False)
        try:
            wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            a_id = wf.label_to_task_id["a"]
            scheduler._runner.complete(a_id, RunStatus.FAILED, error="boom")
            scheduler.tick()

            fragment = _write_spec(tmp_path, {
                "tasks": [{
                    "label": "a",
                    "title": "A2",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "replacement",
                }]
            })
            rc = workflow_cli.main(
                ["replace-workflow", "-r", wf_id, "a", str(fragment)],
                store=store,
            )
            assert rc == 0
            wf = store.load_workflow(wf_id)
            assert wf.task_by_label("a").status is TaskStatus.RUNNING
        finally:
            scheduler.shutdown()

    def test_invalid_schema_section_is_deterministic(self, store, capsys):
        rc = workflow_cli.main(["schema", "nope"], store=store)

        assert rc == 2
        assert capsys.readouterr().err.strip() == "error: schema section not found: nope"


class TestSubmitDoesNotStartScheduler:
    """
    The CLI must not instantiate a scheduler. A scheduler would spawn a
    daemon thread; we can detect that by counting live threads before and
    after the call.
    """

    def test_no_new_threads(self, tmp_path, store):
        import threading
        path = _write_spec(tmp_path, SPEC)
        before = {t.ident for t in threading.enumerate()}
        workflow_cli.main(["submit", str(path)], store=store)
        after = {t.ident for t in threading.enumerate()}
        # Allow the same set or a subset (threads can exit), but no new
        # threads should have been created by submit.
        new = after - before
        assert new == set(), f"submit started unexpected threads: {new}"
