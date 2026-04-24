"""Tests for deterministic watcher tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.kazi_runner import MockRunner
from mr1.mr1 import MR1, StateManager
from mr1.scheduler import Scheduler, WorkflowSpecError, validate_spec
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


def _watcher_spec(task: dict, downstream_prompt: str = "after wait") -> dict:
    return {
        "title": "Watcher workflow",
        "tasks": [
            task,
            {
                "label": "after",
                "title": "After",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": [task["label"]],
                "prompt": downstream_prompt,
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
    sched = Scheduler(store, runner, auto_tick=False, agent_id="scheduler")
    yield sched
    sched.shutdown()


def _task_by_label(wf, label):
    return wf.task_by_label(label)


def _make_py_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


class TestWatcherScheduler:
    def test_file_exists_watcher_unlocks_downstream(self, scheduler, store, runner, tmp_path):
        watched = tmp_path / "watched.txt"
        spec = _watcher_spec({
            "label": "wait_file",
            "title": "Wait for file",
            "task_kind": "watcher",
            "watcher_type": "file_exists",
            "watch_config": {"path": str(watched), "poll_interval_s": 0},
        })

        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        wait_task = _task_by_label(wf, "wait_file")
        after_task = _task_by_label(wf, "after")
        assert wait_task.status is TaskStatus.RUNNING
        assert wait_task.last_checked_at is not None
        assert wait_task.last_check_result["state"] == "not_satisfied"
        assert after_task.status is TaskStatus.WAITING

        watched.write_text("ready", encoding="utf-8")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        wait_task = _task_by_label(wf, "wait_file")
        after_task = _task_by_label(wf, "after")
        assert wait_task.status is TaskStatus.SUCCEEDED
        assert wait_task.watch_satisfied_at is not None
        assert wait_task.last_check_result["state"] == "satisfied"
        assert after_task.status is TaskStatus.RUNNING
        assert runner.started_task_ids == [after_task.task_id]

        events = [event.event_type for event in store.load_events(wf_id)]
        assert "watcher_started" in events
        assert "watcher_checked" in events
        assert "watcher_satisfied" in events
        assert "task_started" in events
        assert "task_succeeded" in events

    def test_time_reached_watcher_future_and_past(self, scheduler, store):
        future_spec = _watcher_spec({
            "label": "future_wait",
            "title": "Future",
            "task_kind": "watcher",
            "watcher_type": "time_reached",
            "watch_config": {"at": "2999-01-01T00:00:00", "poll_interval_s": 0},
        })
        future_wf_id = scheduler.submit_workflow(future_spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        scheduler.tick()
        future_wf = store.load_workflow(future_wf_id)
        future_task = _task_by_label(future_wf, "future_wait")
        assert future_task.status is TaskStatus.RUNNING
        assert future_task.last_check_result["state"] == "not_satisfied"

        past_spec = _watcher_spec({
            "label": "past_wait",
            "title": "Past",
            "task_kind": "watcher",
            "watcher_type": "time_reached",
            "watch_config": {"at": "2000-01-01T00:00:00", "poll_interval_s": 0},
        })
        past_wf_id = scheduler.submit_workflow(past_spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        scheduler.tick()
        past_wf = store.load_workflow(past_wf_id)
        past_task = _task_by_label(past_wf, "past_wait")
        assert past_task.status is TaskStatus.SUCCEEDED
        assert past_task.last_check_result["state"] == "satisfied"

    def test_manual_event_watcher_trigger_unlocks_downstream(self, scheduler, store):
        spec = _watcher_spec({
            "label": "approve",
            "title": "Approve",
            "task_kind": "watcher",
            "watcher_type": "manual_event",
            "watch_config": {"event": "approved", "poll_interval_s": 0},
        })
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "approve")
        assert task.status is TaskStatus.RUNNING
        assert task.last_check_result["state"] == "not_satisfied"

        scheduler.trigger_watcher(
            wf_id,
            "approve",
            event_name="approved",
            metadata={"by": "user"},
        )
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "approve")
        assert task.status is TaskStatus.SUCCEEDED
        assert task.condition["metadata"] == {"by": "user"}
        assert task.last_check_result["state"] == "satisfied"

        scheduler.tick()
        wf = store.load_workflow(wf_id)
        assert _task_by_label(wf, "after").status is TaskStatus.RUNNING

        events = [event.event_type for event in store.load_events(wf_id)]
        assert "watcher_satisfied" in events
        assert "task_succeeded" in events

    def test_condition_script_watcher_exit_codes_and_timeout(self, scheduler, store, tmp_path):
        ok_script = _make_py_script(
            tmp_path / "ok.py",
            "import sys\nprint('ok')\nsys.exit(0)\n",
        )
        no_script = _make_py_script(
            tmp_path / "no.py",
            "import sys\nprint('no')\nsys.exit(1)\n",
        )
        bad_script = _make_py_script(
            tmp_path / "bad.py",
            "import sys\nprint('bad')\nsys.exit(2)\n",
        )
        slow_script = _make_py_script(
            tmp_path / "slow.py",
            "import time\ntime.sleep(2)\n",
        )

        specs = {
            "ok": _watcher_spec({
                "label": "ok_wait",
                "title": "OK",
                "task_kind": "watcher",
                "watcher_type": "condition_script",
                "watch_config": {"path": str(ok_script), "poll_interval_s": 0},
            }),
            "no": _watcher_spec({
                "label": "no_wait",
                "title": "NO",
                "task_kind": "watcher",
                "watcher_type": "condition_script",
                "watch_config": {"path": str(no_script), "poll_interval_s": 0},
            }),
            "bad": _watcher_spec({
                "label": "bad_wait",
                "title": "BAD",
                "task_kind": "watcher",
                "watcher_type": "condition_script",
                "watch_config": {"path": str(bad_script), "poll_interval_s": 0},
            }),
            "slow": _watcher_spec({
                "label": "slow_wait",
                "title": "SLOW",
                "task_kind": "watcher",
                "watcher_type": "condition_script",
                "watch_config": {
                    "path": str(slow_script),
                    "poll_interval_s": 0,
                    "timeout_s": 1,
                },
            }),
        }

        wf_ids = {
            name: scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
            for name, spec in specs.items()
        }
        scheduler.tick()
        scheduler.tick()

        ok_task = _task_by_label(store.load_workflow(wf_ids["ok"]), "ok_wait")
        no_task = _task_by_label(store.load_workflow(wf_ids["no"]), "no_wait")
        bad_task = _task_by_label(store.load_workflow(wf_ids["bad"]), "bad_wait")
        slow_task = _task_by_label(store.load_workflow(wf_ids["slow"]), "slow_wait")

        assert ok_task.status is TaskStatus.SUCCEEDED
        assert ok_task.last_check_result["state"] == "satisfied"
        assert no_task.status is TaskStatus.RUNNING
        assert no_task.last_check_result["state"] == "not_satisfied"
        assert bad_task.status is TaskStatus.FAILED
        assert bad_task.last_check_result["state"] == "failed"
        assert slow_task.status is TaskStatus.TIMED_OUT
        assert slow_task.last_check_result["state"] == "timed_out"

    def test_poll_interval_skip_keeps_running_and_preserves_last_check(self, scheduler, store, tmp_path):
        watched = tmp_path / "never.txt"
        spec = _watcher_spec({
            "label": "wait_file",
            "title": "Wait",
            "task_kind": "watcher",
            "watcher_type": "file_exists",
            "watch_config": {"path": str(watched), "poll_interval_s": 60},
        })
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "wait_file")
        first_checked = task.last_checked_at
        first_result = dict(task.last_check_result)

        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "wait_file")
        assert task.status is TaskStatus.RUNNING
        assert task.last_checked_at == first_checked
        assert task.last_check_result == first_result


class TestWatcherValidation:
    def test_rejects_missing_watcher_type(self):
        with pytest.raises(WorkflowSpecError, match="watcher_type"):
            validate_spec({
                "tasks": [{
                    "label": "wait",
                    "title": "Wait",
                    "task_kind": "watcher",
                    "watch_config": {"path": "/tmp/x"},
                }],
            })

    def test_rejects_unknown_watcher_type(self):
        with pytest.raises(WorkflowSpecError, match="unknown watcher_type"):
            validate_spec({
                "tasks": [{
                    "label": "wait",
                    "title": "Wait",
                    "task_kind": "watcher",
                    "watcher_type": "unknown",
                    "watch_config": {},
                }],
            })

    def test_condition_script_rejects_missing_nonexistent_and_nonfile_paths(self, tmp_path):
        missing = tmp_path / "missing.py"
        with pytest.raises(WorkflowSpecError, match="does not exist"):
            validate_spec({
                "tasks": [{
                    "label": "wait",
                    "title": "Wait",
                    "task_kind": "watcher",
                    "watcher_type": "condition_script",
                    "watch_config": {"path": str(missing)},
                }],
            })

        with pytest.raises(WorkflowSpecError, match="not a file"):
            validate_spec({
                "tasks": [{
                    "label": "wait",
                    "title": "Wait",
                    "task_kind": "watcher",
                    "watcher_type": "condition_script",
                    "watch_config": {"path": str(tmp_path)},
                }],
            })

    def test_agent_task_behavior_still_valid(self):
        validate_spec({
            "tasks": [{
                "label": "a",
                "title": "A",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "x",
            }],
        })


class TestWatcherCliAndMr1:
    def test_workflow_cli_watchers_and_trigger(self, scheduler, store, tmp_path, capsys):
        watched = tmp_path / "cli-watch.txt"
        watch_spec = _watcher_spec({
            "label": "wait_file",
            "title": "Wait file",
            "task_kind": "watcher",
            "watcher_type": "file_exists",
            "watch_config": {"path": str(watched), "poll_interval_s": 0},
        })
        scheduler.submit_workflow(watch_spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        scheduler.tick()

        rc = workflow_cli.main(["watchers"], store=store)
        assert rc == 0
        out = capsys.readouterr().out
        assert "WATCHER" in out
        assert "file_exists" in out

        trigger_spec = _watcher_spec({
            "label": "approve",
            "title": "Approve",
            "task_kind": "watcher",
            "watcher_type": "manual_event",
            "watch_config": {"event": "approved", "poll_interval_s": 0},
        })
        wf_id = scheduler.submit_workflow(trigger_spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        scheduler.tick()
        capsys.readouterr()

        rc = workflow_cli.main(["trigger", wf_id, "approve", "approved"], store=store)
        assert rc == 0
        task_id = capsys.readouterr().out.strip()
        assert task_id.startswith("tk-")

        scheduler.tick()
        wf = store.load_workflow(wf_id)
        assert _task_by_label(wf, "after").status is TaskStatus.RUNNING

    def test_mr1_watchers_and_workflow_trigger_commands(self, tmp_path):
        store = WorkflowStore(root=tmp_path / "workflows")
        mr1 = MR1(
            workflow_store=store,
            workflow_runner=MockRunner(),
            workflow_auto_tick=False,
        )
        mr1._state = StateManager(state_path=tmp_path / "mr1_state.json")

        spec_path = tmp_path / "manual_workflow.json"
        spec_path.write_text(json.dumps(_watcher_spec({
            "label": "approve",
            "title": "Approve",
            "task_kind": "watcher",
            "watcher_type": "manual_event",
            "watch_config": {"event": "approved", "poll_interval_s": 0},
        })), encoding="utf-8")

        submit_result = mr1._handle_builtin(f"/workflow submit {spec_path}")
        wf_id = submit_result.split(": ", 1)[1]
        mr1._handle_builtin("/scheduler tick")
        mr1._handle_builtin("/scheduler tick")

        watchers_output = mr1._handle_builtin("/watchers")
        assert "manual_event" in watchers_output

        trigger_output = mr1._handle_builtin(f"/workflow trigger {wf_id} approve approved")
        assert trigger_output.startswith("triggered watcher: tk-")

        wf = store.load_workflow(wf_id)
        assert _task_by_label(wf, "approve").status is TaskStatus.SUCCEEDED
        assert _task_by_label(wf, "after").status is TaskStatus.RUNNING
