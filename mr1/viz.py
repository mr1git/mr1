"""
MR1 visualization snapshot utilities.

Builds a deterministic JSON snapshot from MR1 state and task logs so a
separate CLI UI can render the agent tree without touching runtime internals.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


_PKG_ROOT = Path(__file__).resolve().parent
_DEFAULT_STATE_PATH = _PKG_ROOT / "memory" / "active" / "mr1_state.json"
_DEFAULT_TASKS_DIR = _PKG_ROOT / "tasks"

_ROUND_SUFFIX_RE = re.compile(r"-r\d+$")
_SUBTASK_RE = re.compile(r"(.*)-sub\d+$")
_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "timeout",
    "context_exceeded",
    "denied",
    "killed",
}
_SYSTEM_AGENT_PREFIXES = ("mem_", "ctx_", "com_", "mini")
_MAX_CONVERSATION = 80


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_task_id(task_id: str) -> str:
    """Collapse MRn round ids like task-123-r0 into their base task id."""
    return _ROUND_SUFFIX_RE.sub("", task_id)


def _infer_parent_task_id(task_id: str) -> Optional[str]:
    match = _SUBTASK_RE.fullmatch(task_id)
    if match:
        return match.group(1)
    return "mr1"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_log_entries(tasks_dir: Path) -> Iterable[dict[str, Any]]:
    if not tasks_dir.exists():
        return []

    entries: list[dict[str, Any]] = []
    for log_file in sorted(tasks_dir.glob("*/logs/*.jsonl")):
        if log_file.name.startswith("."):
            continue
        try:
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    entries.sort(key=lambda entry: entry.get("timestamp", ""))
    return entries


def _status_from_entry(entry: dict[str, Any]) -> Optional[str]:
    action = entry.get("action")
    result = entry.get("result")

    if action == "complete":
        return "completed" if result == "ok" else "failed"
    if action == "distill":
        return "completed" if result == "ok" else "failed"
    if action == "timeout":
        return "timeout"
    if action == "context_exceeded":
        return "context_exceeded"
    if action == "kill":
        return "killed"
    if action == "permission_check" and result == "denied":
        return "denied"
    return None


def _describe_event(entry: dict[str, Any]) -> str:
    agent = str(entry.get("agent_type", "?")).upper()
    action = str(entry.get("action", "?"))
    metadata = entry.get("metadata", {}) or {}

    if action == "delegate":
        target = metadata.get("to", "?")
        description = str(metadata.get("description", "")).strip()
        if description:
            return f"{agent} delegated to {target}: {description[:80]}"
        return f"{agent} delegated to {target}"

    if action == "spawn":
        pid = metadata.get("pid")
        return f"{agent} spawned pid={pid}"

    if action == "complete":
        duration = metadata.get("duration_s")
        if duration is not None:
            return f"{agent} completed in {duration}s"
        return f"{agent} completed"

    if action == "timeout":
        timeout_s = metadata.get("timeout_s")
        return f"{agent} timed out after {timeout_s}s"

    if action == "context_exceeded":
        return f"{agent} hit the context window"

    if action == "kill":
        return f"{agent} was killed"

    if action == "permission_check":
        detail = str(metadata.get("detail", "")).strip()
        return f"{agent} permission denied: {detail[:80]}"

    return f"{agent} {action}"


def _infer_lane(agent_type: Optional[str], lane_hint: Optional[str] = None) -> str:
    if lane_hint in ("conversation", "system"):
        return lane_hint
    if not agent_type:
        return "conversation"
    if agent_type.startswith(_SYSTEM_AGENT_PREFIXES) or agent_type == "mem_dltr":
        return "system"
    return "conversation"


def _normalize_event_type(entry: dict[str, Any]) -> str:
    action = entry.get("action")
    result = entry.get("result")
    if action == "delegate":
        return "task_attached"
    if action == "spawn":
        return "task_spawned"
    if action == "complete" and result == "ok":
        return "task_completed"
    if action in ("complete", "timeout", "context_exceeded", "permission_check", "kill"):
        return "task_failed" if action != "kill" else "task_detached"
    if action == "distill":
        return "system_event"
    return "system_event"


def build_snapshot(
    state_path: Path = _DEFAULT_STATE_PATH,
    tasks_dir: Path = _DEFAULT_TASKS_DIR,
) -> dict[str, Any]:
    """Build a UI-friendly snapshot from persisted state and task logs."""
    state = _load_json(state_path)
    raw_entries = list(_iter_log_entries(tasks_dir))

    decisions = state.get("decisions", [])
    state_tasks = state.get("tasks", {})
    task_records: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    def ensure_task(task_id: str) -> dict[str, Any]:
        record = task_records.get(task_id)
        if record is None:
            record = {
                "task_id": task_id,
                "parent_task_id": _infer_parent_task_id(task_id),
                "agent_type": None,
                "status": "pending",
                "description": "",
                "started_at": None,
                "updated_at": None,
                "finished_at": None,
                "pid": None,
                "event_count": 0,
            }
            task_records[task_id] = record
        return record

    for entry in raw_entries:
        original_task_id = entry.get("task_id")
        if not original_task_id or str(original_task_id).startswith("mr1-brain-"):
            continue

        task_id = canonical_task_id(str(original_task_id))
        record = ensure_task(task_id)
        record["event_count"] += 1
        record["updated_at"] = entry.get("timestamp") or record["updated_at"]

        metadata = entry.get("metadata", {}) or {}
        action = entry.get("action")
        agent_type = entry.get("agent_type")
        lane = _infer_lane(agent_type, metadata.get("lane"))

        if agent_type and record["agent_type"] is None and agent_type != "mr1":
            record["agent_type"] = agent_type
        if not record.get("lane"):
            record["lane"] = lane

        if action == "spawn":
            record["started_at"] = record["started_at"] or entry.get("timestamp")
            record["pid"] = metadata.get("pid") or record["pid"]
            if record["status"] not in _TERMINAL_STATUSES:
                record["status"] = "running"

        if action == "delegate":
            record["started_at"] = record["started_at"] or entry.get("timestamp")
            record["description"] = (
                str(metadata.get("description", "")).strip() or record["description"]
            )
            record["parent_task_id"] = metadata.get("parent_task_id") or record["parent_task_id"]
            record["lane"] = _infer_lane(record.get("agent_type"), metadata.get("lane"))

        terminal_status = _status_from_entry(entry)
        if terminal_status is not None:
            record["status"] = terminal_status
            record["finished_at"] = entry.get("timestamp")
            record["updated_at"] = entry.get("timestamp")
            if terminal_status in _TERMINAL_STATUSES:
                record["detached_at"] = entry.get("timestamp")

        events.append(
            {
                "timestamp": entry.get("timestamp"),
                "task_id": task_id,
                "agent_type": agent_type,
                "action": action,
                "result": entry.get("result"),
                "event_type": _normalize_event_type(entry),
                "lane": lane,
                "summary": _describe_event(entry),
            }
        )

    for task_id, task in state_tasks.items():
        record = ensure_task(task_id)
        record["agent_type"] = task.get("agent_type") or record["agent_type"]
        record["description"] = task.get("description") or record["description"]
        record["started_at"] = task.get("started_at") or record["started_at"]
        record["updated_at"] = task.get("finished_at") or task.get("started_at") or record["updated_at"]
        record["finished_at"] = task.get("finished_at") or record["finished_at"]
        record["pid"] = task.get("pid") or record["pid"]
        record["parent_task_id"] = task.get("parent_task_id") or record["parent_task_id"]
        record["lane"] = task.get("lane") or record.get("lane") or _infer_lane(record["agent_type"])
        state_status = task.get("status")
        if state_status:
            record["status"] = state_status
        if state_status in _TERMINAL_STATUSES:
            record["detached_at"] = task.get("finished_at") or record.get("detached_at")

    tasks = sorted(
        task_records.values(),
        key=lambda record: (
            record.get("started_at") or "",
            record.get("task_id") or "",
        ),
    )
    running_count = sum(1 for task in tasks if task["status"] == "running")
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_parent.setdefault(task.get("parent_task_id") or "mr1", []).append(task)
    for siblings in by_parent.values():
        siblings.sort(
            key=lambda task: (
                task.get("started_at") or "",
                task.get("task_id") or "",
            )
        )

    def assign_paths(parent_id: str, parent_path: list[int]) -> None:
        for index, task in enumerate(by_parent.get(parent_id, [])):
            path = parent_path + [index]
            task["path"] = path
            assign_paths(task["task_id"], path)

    assign_paths("mr1", [])

    events.sort(key=lambda event: event.get("timestamp") or "", reverse=True)
    conversation = sorted(state.get("conversation", []), key=lambda entry: entry.get("timestamp", ""))

    return {
        "generated_at": _now_iso(),
        "session": {
            "session_id": state.get("session_id"),
            "started_at": state.get("started_at"),
            "status": "running" if state else "idle",
        },
        "summary": {
            "task_count": len(tasks),
            "running_count": running_count,
            "decision_count": len(decisions),
        },
        "root": {
            "id": "mr1",
            "name": "MR1",
            "status": "running" if state else "idle",
        },
        "tasks": tasks,
        "events": events[:40],
        "recent_decisions": decisions[-10:],
        "conversation": conversation[-_MAX_CONVERSATION:],
    }


def _resolve_paths(
    project_root: Optional[str],
    state_path: Optional[str],
    tasks_dir: Optional[str],
) -> tuple[Path, Path]:
    if state_path and tasks_dir:
        return Path(state_path), Path(tasks_dir)

    if project_root:
        root = Path(project_root)
        return (
            Path(state_path) if state_path else root / "mr1" / "memory" / "active" / "mr1_state.json",
            Path(tasks_dir) if tasks_dir else root / "mr1" / "tasks",
        )

    return (
        Path(state_path) if state_path else _DEFAULT_STATE_PATH,
        Path(tasks_dir) if tasks_dir else _DEFAULT_TASKS_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MR1 visualization helpers")
    parser.add_argument("--snapshot", action="store_true", help="print a JSON snapshot")
    parser.add_argument("--project-root", help="project root containing the mr1 package")
    parser.add_argument("--state-path", help="override mr1_state.json path")
    parser.add_argument("--tasks-dir", help="override tasks directory path")
    args = parser.parse_args()

    if args.snapshot:
        state_path, tasks_dir = _resolve_paths(
            args.project_root,
            args.state_path,
            args.tasks_dir,
        )
        print(json.dumps(build_snapshot(state_path=state_path, tasks_dir=tasks_dir)))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
