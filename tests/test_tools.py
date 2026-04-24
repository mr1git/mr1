"""Tests for deterministic workflow tool tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.kazi_runner import MockRunner
from mr1.mr1 import MR1, StateManager
from mr1.scheduler import Scheduler, WorkflowSpecError, validate_spec
from mr1.tools import ToolConfigError, default_tool_registry
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="agent", id="MR1")


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def scheduler(store):
    sched = Scheduler(store, MockRunner(), auto_tick=False, agent_id="scheduler")
    yield sched
    sched.shutdown()


def _task_by_label(wf, label):
    return wf.task_by_label(label)


def _tool_workflow(task: dict, downstream: dict | None = None) -> dict:
    tasks = [task]
    if downstream is not None:
        tasks.append(downstream)
    return {"title": "Tool workflow", "tasks": tasks}


class TestToolValidation:
    def test_unknown_tool_rejected(self):
        with pytest.raises(WorkflowSpecError, match="unknown tool_type"):
            validate_spec({
                "tasks": [{
                    "label": "x",
                    "title": "X",
                    "task_kind": "tool",
                    "tool_type": "unknown_tool",
                    "tool_config": {},
                }],
            })

    def test_missing_tool_type_rejected(self):
        with pytest.raises(WorkflowSpecError, match="tool_type"):
            validate_spec({
                "tasks": [{
                    "label": "x",
                    "title": "X",
                    "task_kind": "tool",
                    "tool_config": {},
                }],
            })

    def test_shell_command_validation_rejects_bad_shapes(self, tmp_path):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="argv"):
            registry.validate_spec("shell_command", {"argv": "python --version"})
        with pytest.raises(ToolConfigError, match="argv"):
            registry.validate_spec("shell_command", {"argv": []})
        with pytest.raises(ToolConfigError, match="argv"):
            registry.validate_spec("shell_command", {"argv": ["python", 1]})
        with pytest.raises(ToolConfigError, match="cwd"):
            registry.validate_spec("shell_command", {"argv": ["python"], "cwd": str(tmp_path / "missing")})
        with pytest.raises(ToolConfigError, match="timeout_s"):
            registry.validate_spec("shell_command", {"argv": ["python"], "timeout_s": 301})

    def test_read_file_missing_path_is_runtime_failure(self, scheduler, store, tmp_path):
        missing = tmp_path / "missing.txt"
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "read_notes",
                "title": "Read notes",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": str(missing)},
            }),
            PROV,
        )
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "read_notes")
        assert task.status is TaskStatus.FAILED
        output = store.load_task_output(wf_id, task.task_id)
        assert output is not None
        assert output.status == "failed"
        assert "does not exist" in output.metadata["error"]


class TestReadWriteTools:
    def test_read_file_writes_output_and_downstream_consumes_text(self, store, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("alpha\nbeta\n", encoding="utf-8")
        started_prompts: dict[str, str] = {}
        runner = MockRunner(on_start=lambda task: started_prompts.setdefault(task.label, task.prompt))
        scheduler = Scheduler(store, runner, auto_tick=False, agent_id="scheduler")
        try:
            wf_id = scheduler.submit_workflow(
                _tool_workflow(
                    {
                        "label": "read_notes",
                        "title": "Read notes",
                        "task_kind": "tool",
                        "tool_type": "read_file",
                        "tool_config": {"path": str(notes)},
                    },
                    {
                        "label": "summarize",
                        "title": "Summarize",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "depends_on": ["read_notes"],
                        "inputs": [{"name": "notes", "from": "read_notes.result.text"}],
                        "prompt": "Summarize these notes.",
                    },
                ),
                PROV,
            )
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            read_task = _task_by_label(wf, "read_notes")
            assert read_task.status is TaskStatus.SUCCEEDED
            assert store.load_task_output(wf_id, read_task.task_id) is not None

            scheduler.tick()
            wf = store.load_workflow(wf_id)
            summarize = _task_by_label(wf, "summarize")
            assert summarize.status is TaskStatus.RUNNING
            assert "alpha" in started_prompts["summarize"]
            output = store.load_task_output(wf_id, read_task.task_id)
            assert output.text.startswith("alpha")
            assert output.artifacts[0].name == "file"
        finally:
            scheduler.shutdown()

    def test_write_file_registers_artifact_and_respects_overwrite_false(self, scheduler, store, tmp_path):
        target = tmp_path / "outputs" / "summary.txt"
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "write_summary",
                "title": "Write summary",
                "task_kind": "tool",
                "tool_type": "write_file",
                "tool_config": {
                    "path": str(target),
                    "content": "hello world",
                    "create_dirs": True,
                    "overwrite": False,
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "write_summary")
        assert task.status is TaskStatus.SUCCEEDED
        assert target.read_text(encoding="utf-8") == "hello world"
        assert task.artifacts[0].name == "written_file"

        wf_id_2 = scheduler.submit_workflow(
            _tool_workflow({
                "label": "write_summary",
                "title": "Write summary",
                "task_kind": "tool",
                "tool_type": "write_file",
                "tool_config": {
                    "path": str(target),
                    "content": "new content",
                    "create_dirs": True,
                    "overwrite": False,
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf2 = store.load_workflow(wf_id_2)
        task2 = _task_by_label(wf2, "write_summary")
        assert task2.status is TaskStatus.FAILED
        assert "overwrite is false" in (task2.tool_error or "")

    def test_downstream_can_consume_write_file_artifact(self, store, tmp_path):
        target = tmp_path / "out.txt"
        started_prompts: dict[str, str] = {}
        runner = MockRunner(on_start=lambda task: started_prompts.setdefault(task.label, task.prompt))
        scheduler = Scheduler(store, runner, auto_tick=False, agent_id="scheduler")
        try:
            wf_id = scheduler.submit_workflow(
                _tool_workflow(
                    {
                        "label": "writer",
                        "title": "Writer",
                        "task_kind": "tool",
                        "tool_type": "write_file",
                        "tool_config": {
                            "path": str(target),
                            "content": "hello",
                            "overwrite": True,
                        },
                    },
                    {
                        "label": "consumer",
                        "title": "Consumer",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "depends_on": ["writer"],
                        "inputs": [{"name": "path", "from": "writer.artifact.written_file"}],
                        "prompt": "Use the written file path.",
                    },
                ),
                PROV,
            )
            scheduler.tick()
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            consumer = _task_by_label(wf, "consumer")
            assert consumer.status is TaskStatus.RUNNING
            assert str(target) in started_prompts["consumer"]
        finally:
            scheduler.shutdown()


class TestShellCommandTool:
    def test_shell_command_success(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "python_version",
                "title": "Python version",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {"argv": [sys.executable, "--version"]},
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "python_version")
        output = store.load_task_output(wf_id, task.task_id)
        assert task.status is TaskStatus.SUCCEEDED
        assert output is not None
        assert output.status == "succeeded"
        assert output.data["exit_code"] == 0
        assert "Python" in output.data["stdout"] or "Python" in output.data["stderr"]
        names = {artifact.name for artifact in task.artifacts}
        assert "stdout" in names or "stderr" in names

    def test_shell_command_nonzero_preserves_output_json(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "bad_cmd",
                "title": "Bad command",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import sys; print('bad'); sys.stderr.write('err\\n'); sys.exit(5)",
                    ],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "bad_cmd")
        output = store.load_task_output(wf_id, task.task_id)
        assert task.status is TaskStatus.FAILED
        assert output is not None
        assert output.status == "failed"
        assert output.data["exit_code"] == 5
        assert "bad" in output.data["stdout"]
        assert "err" in output.data["stderr"]

    def test_shell_command_timeout(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "slow_cmd",
                "title": "Slow command",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import sys,time; print('slow', flush=True); sys.stderr.write('wait\\n'); sys.stderr.flush(); time.sleep(2)",
                    ],
                    "timeout_s": 1,
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "slow_cmd")
        output = store.load_task_output(wf_id, task.task_id)
        assert task.status is TaskStatus.TIMED_OUT
        assert output is not None
        assert output.status == "timed_out"
        assert "slow" in output.data["stdout"]
        assert "wait" in output.data["stderr"]

    def test_shell_command_truncation_and_artifacts(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "noisy",
                "title": "Noisy",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import sys; print('A'*200); sys.stderr.write('B'*200)",
                    ],
                    "capture_max_bytes": 32,
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "noisy")
        output = store.load_task_output(wf_id, task.task_id)
        assert output is not None
        assert output.data["stdout_truncated"] is True
        assert output.data["stderr_truncated"] is True
        assert len(output.data["stdout"]) <= 32
        assert len(output.data["stderr"]) <= 32
        assert {artifact.name for artifact in task.artifacts} == {"stdout", "stderr"}


class TestToolSchedulerIntegration:
    def test_tool_does_not_consume_agent_concurrency_slots(self, store, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("hello", encoding="utf-8")
        runner = MockRunner()
        scheduler = Scheduler(store, runner, auto_tick=False, concurrency=1, agent_id="scheduler")
        try:
            wf_id = scheduler.submit_workflow(
                {
                    "title": "Mixed",
                    "tasks": [
                        {
                            "label": "agent",
                            "title": "Agent",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "prompt": "work",
                        },
                        {
                            "label": "reader",
                            "title": "Reader",
                            "task_kind": "tool",
                            "tool_type": "read_file",
                            "tool_config": {"path": str(notes)},
                        },
                    ],
                },
                PROV,
            )
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            assert _task_by_label(wf, "agent").status is TaskStatus.RUNNING
            assert _task_by_label(wf, "reader").status is TaskStatus.SUCCEEDED
            assert runner.started_task_ids == [wf.label_to_task_id["agent"]]
        finally:
            scheduler.shutdown()

    def test_tool_failure_blocks_dependents(self, scheduler, store, tmp_path):
        wf_id = scheduler.submit_workflow(
            _tool_workflow(
                {
                    "label": "reader",
                    "title": "Reader",
                    "task_kind": "tool",
                    "tool_type": "read_file",
                    "tool_config": {"path": str(tmp_path / "missing.txt")},
                },
                {
                    "label": "after",
                    "title": "After",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "depends_on": ["reader"],
                    "prompt": "after",
                },
            ),
            PROV,
        )
        scheduler.tick()
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        assert _task_by_label(wf, "reader").status is TaskStatus.FAILED
        assert _task_by_label(wf, "after").status is TaskStatus.BLOCKED

    def test_successful_tool_writes_output_immediately(self, scheduler, store, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("hello", encoding="utf-8")
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "reader",
                "title": "Reader",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": str(notes)},
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "reader")
        assert task.output_path is not None
        assert Path(task.output_path).exists()


class TestToolDataflowIntegration:
    def test_downstream_can_reference_shell_result_data_stdout(self, store):
        started_prompts: dict[str, str] = {}
        runner = MockRunner(on_start=lambda task: started_prompts.setdefault(task.label, task.prompt))
        scheduler = Scheduler(store, runner, auto_tick=False, agent_id="scheduler")
        try:
            wf_id = scheduler.submit_workflow(
                _tool_workflow(
                    {
                        "label": "shell",
                        "title": "Shell",
                        "task_kind": "tool",
                        "tool_type": "shell_command",
                        "tool_config": {"argv": [sys.executable, "--version"]},
                    },
                    {
                        "label": "consume",
                        "title": "Consume",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "depends_on": ["shell"],
                        "inputs": [{"name": "stdout", "from": "shell.result.data.stdout"}],
                        "prompt": "Use stdout.",
                    },
                ),
                PROV,
            )
            scheduler.tick()
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            assert _task_by_label(wf, "consume").status is TaskStatus.RUNNING
            assert "Python" in started_prompts["consume"]
        finally:
            scheduler.shutdown()

    def test_downstream_can_reference_shell_artifact_stdout(self, store):
        started_prompts: dict[str, str] = {}
        runner = MockRunner(on_start=lambda task: started_prompts.setdefault(task.label, task.prompt))
        scheduler = Scheduler(store, runner, auto_tick=False, agent_id="scheduler")
        try:
            wf_id = scheduler.submit_workflow(
                _tool_workflow(
                    {
                        "label": "shell",
                        "title": "Shell",
                        "task_kind": "tool",
                        "tool_type": "shell_command",
                        "tool_config": {"argv": [sys.executable, "--version"]},
                    },
                    {
                        "label": "consume",
                        "title": "Consume",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "depends_on": ["shell"],
                        "inputs": [{"name": "stdout_path", "from": "shell.artifact.stdout"}],
                        "prompt": "Use stdout artifact.",
                    },
                ),
                PROV,
            )
            scheduler.tick()
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            assert _task_by_label(wf, "consume").status is TaskStatus.RUNNING
            assert "stdout.txt" in started_prompts["consume"]
        finally:
            scheduler.shutdown()

    def test_missing_tool_output_reference_fails_before_launch(self, store):
        runner = MockRunner()
        scheduler = Scheduler(store, runner, auto_tick=False, agent_id="scheduler")
        try:
            wf_id = scheduler.submit_workflow(
                _tool_workflow(
                    {
                        "label": "shell",
                        "title": "Shell",
                        "task_kind": "tool",
                        "tool_type": "shell_command",
                        "tool_config": {"argv": [sys.executable, "--version"]},
                    },
                    {
                        "label": "consume",
                        "title": "Consume",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "depends_on": ["shell"],
                        "inputs": [{"name": "missing", "from": "shell.artifact.nope"}],
                        "prompt": "Use missing artifact.",
                    },
                ),
                PROV,
            )
            scheduler.tick()
            scheduler.tick()
            wf = store.load_workflow(wf_id)
            consume = _task_by_label(wf, "consume")
            assert consume.status is TaskStatus.FAILED
            assert "failed to resolve workflow input(s)" in (consume.dataflow_error or "")
            assert runner.started_task_ids == []
        finally:
            scheduler.shutdown()


class TestToolCliAndMr1:
    def test_workflow_cli_tools_and_task_detail(self, tmp_path, store, capsys):
        rc = workflow_cli.main(["tools"], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "read_file" in out
        assert "shell_command" in out

        spec_path = tmp_path / "tool_workflow.json"
        spec_path.write_text(json.dumps(_tool_workflow({
            "label": "reader",
            "title": "Reader",
            "task_kind": "tool",
            "tool_type": "read_file",
            "tool_config": {"path": str(tmp_path / "notes.txt")},
        })), encoding="utf-8")
        workflow_cli.main(["submit", str(spec_path)], store=store)
        wf_id = capsys.readouterr().out.strip()
        wf = store.load_workflow(wf_id)
        task_id = wf.label_to_task_id["reader"]

        rc = workflow_cli.main(["task", task_id], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "tool:" in out
        assert "tool_config:" in out

    def test_mr1_tools_command(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        mr1 = MR1(
            workflow_store=store,
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
        )
        mr1._state = StateManager(state_path=tmp_path / "mr1_state.json")
        output = mr1._handle_builtin("/tools")
        assert "read_file" in output
        assert "write_file" in output
