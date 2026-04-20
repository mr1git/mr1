"""Tests for mr1.viz snapshot generation."""

import json
from pathlib import Path

from mr1.viz import build_snapshot, canonical_task_id


def _append_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_canonical_task_id_strips_round_suffix():
    assert canonical_task_id("task-123-r0") == "task-123"
    assert canonical_task_id("task-123-sub0-r1") == "task-123-sub0"


def test_build_snapshot_tracks_live_tree_and_statuses(tmp_path):
    state_path = tmp_path / "memory" / "active" / "mr1_state.json"
    tasks_dir = tmp_path / "tasks"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "session_id": "sess-1",
                "started_at": "2026-04-20T01:00:00+00:00",
                "tasks": {
                    "task-top": {
                        "agent_type": "mr2",
                        "status": "running",
                        "pid": 111,
                        "description": "top level job",
                        "started_at": "2026-04-20T01:01:00+00:00",
                    }
                },
                "decisions": [
                    {
                        "timestamp": "2026-04-20T01:01:00+00:00",
                        "input_summary": "do the thing",
                        "action": "spawn_mr2",
                        "task_id": "task-top",
                    }
                ],
                "agent_pids": [111],
            }
        )
    )

    _append_jsonl(
        tasks_dir / "task-top" / "logs" / "mr1.jsonl",
        [
            {
                "timestamp": "2026-04-20T01:01:00+00:00",
                "task_id": "task-top",
                "agent_type": "mr1",
                "action": "delegate",
                "result": "ok",
                "metadata": {
                    "to": "mr2",
                    "description": "top level job",
                    "parent_task_id": "mr1",
                },
            }
        ],
    )
    _append_jsonl(
        tasks_dir / "task-top-r0" / "logs" / "mr2.jsonl",
        [
            {
                "timestamp": "2026-04-20T01:01:01+00:00",
                "task_id": "task-top-r0",
                "agent_type": "mr2",
                "action": "spawn",
                "result": "ok",
                "metadata": {"pid": 111},
            }
        ],
    )
    _append_jsonl(
        tasks_dir / "task-top-sub0" / "logs" / "mr2.jsonl",
        [
            {
                "timestamp": "2026-04-20T01:01:05+00:00",
                "task_id": "task-top-sub0",
                "agent_type": "mr2",
                "action": "delegate",
                "result": "ok",
                "metadata": {
                    "to": "kazi",
                    "description": "child worker job",
                    "parent_task_id": "task-top",
                },
            }
        ],
    )
    _append_jsonl(
        tasks_dir / "task-top-sub0" / "logs" / "kazi.jsonl",
        [
            {
                "timestamp": "2026-04-20T01:01:06+00:00",
                "task_id": "task-top-sub0",
                "agent_type": "kazi",
                "action": "spawn",
                "result": "ok",
                "metadata": {"pid": 222},
            },
            {
                "timestamp": "2026-04-20T01:01:10+00:00",
                "task_id": "task-top-sub0",
                "agent_type": "kazi",
                "action": "complete",
                "result": "ok",
                "metadata": {"pid": 222, "duration_s": 4.0},
            },
        ],
    )

    snapshot = build_snapshot(state_path=state_path, tasks_dir=tasks_dir)

    assert snapshot["session"]["session_id"] == "sess-1"
    assert snapshot["summary"]["task_count"] == 2
    assert snapshot["summary"]["running_count"] == 1

    tasks = {task["task_id"]: task for task in snapshot["tasks"]}
    assert tasks["task-top"]["parent_task_id"] == "mr1"
    assert tasks["task-top"]["status"] == "running"
    assert tasks["task-top"]["description"] == "top level job"

    assert tasks["task-top-sub0"]["parent_task_id"] == "task-top"
    assert tasks["task-top-sub0"]["status"] == "completed"
    assert tasks["task-top-sub0"]["description"] == "child worker job"

    assert snapshot["events"][0]["task_id"] in {"task-top", "task-top-sub0"}
