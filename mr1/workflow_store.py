"""
Workflow persistence layer.

`workflow.json` per workflow directory is the single source of truth in
Phase 1. There is NO top-level index file; `list_workflows()` scans the
root directory. Phase 1 workflow counts are small enough that the scan
cost is negligible and we avoid the consistency burden of a separate
index.

Layout::

    <root>/<wf_id>/workflow.json
    <root>/<wf_id>/events.jsonl
    <root>/<wf_id>/tasks/<task_id>/stdout.log
    <root>/<wf_id>/tasks/<task_id>/stderr.log
    <root>/<wf_id>/tasks/<task_id>/result.json
    <root>/<wf_id>/tasks/<task_id>/attempts/<attempt_id>/stdout.log
    <root>/<wf_id>/tasks/<task_id>/attempts/<attempt_id>/stderr.log
    <root>/<wf_id>/tasks/<task_id>/attempts/<attempt_id>/result.json

All state mutation uses an atomic tmp → rename. A single store-level
`RLock` serialises mutation across the workflow.json and events.jsonl
files so "state persisted but event missing" and "event logged but state
not persisted" failure modes cannot occur mid-write.

Callers that must bundle a state change and its event atomically use the
`locked()` context manager, which holds the lock across an arbitrary
sequence of `save_workflow` + `append_event` calls.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from mr1.dataflow import Artifact, ResolvedTaskInput, TaskOutput
from mr1.workflow_models import Workflow, WorkflowEvent


_DEFAULT_ROOT = Path(__file__).resolve().parent / "memory" / "workflows"


class WorkflowStore:
    """
    Thread-safe persistent store for workflows and their events.

    The store never runs a scheduler or launches any process. It only
    reads from and writes to disk.
    """

    def __init__(self, root: Optional[Path] = None):
        self._root = Path(root) if root else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def workflow_dir(self, workflow_id: str) -> Path:
        return self._root / workflow_id

    def workflow_json_path(self, workflow_id: str) -> Path:
        return self.workflow_dir(workflow_id) / "workflow.json"

    def events_jsonl_path(self, workflow_id: str) -> Path:
        return self.workflow_dir(workflow_id) / "events.jsonl"

    def task_log_paths(self, workflow_id: str, task_id: str) -> tuple[Path, Path]:
        base = self.workflow_dir(workflow_id) / "tasks" / task_id
        base.mkdir(parents=True, exist_ok=True)
        return base / "stdout.log", base / "stderr.log"

    def task_attempt_dir(self, workflow_id: str, task_id: str, attempt_id: int) -> Path:
        base = self.workflow_dir(workflow_id) / "tasks" / task_id / "attempts" / str(attempt_id)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def task_attempt_log_paths(
        self,
        workflow_id: str,
        task_id: str,
        attempt_id: int,
    ) -> tuple[Path, Path]:
        base = self.task_attempt_dir(workflow_id, task_id, attempt_id)
        return base / "stdout.log", base / "stderr.log"

    def task_result_path(self, workflow_id: str, task_id: str) -> Path:
        base = self.workflow_dir(workflow_id) / "tasks" / task_id
        base.mkdir(parents=True, exist_ok=True)
        return base / "result.json"

    def task_attempt_result_path(self, workflow_id: str, task_id: str, attempt_id: int) -> Path:
        base = self.task_attempt_dir(workflow_id, task_id, attempt_id)
        return base / "result.json"

    def task_output_path(self, workflow_id: str, task_id: str) -> Path:
        base = self.workflow_dir(workflow_id) / "tasks" / task_id
        base.mkdir(parents=True, exist_ok=True)
        return base / "output.json"

    def task_inputs_path(self, workflow_id: str, task_id: str) -> Path:
        base = self.workflow_dir(workflow_id) / "tasks" / task_id
        base.mkdir(parents=True, exist_ok=True)
        return base / "inputs.json"

    def materialized_prompt_path(self, workflow_id: str, task_id: str) -> Path:
        base = self.workflow_dir(workflow_id) / "tasks" / task_id
        base.mkdir(parents=True, exist_ok=True)
        return base / "materialized_prompt.txt"

    def task_artifacts_dir(self, workflow_id: str, task_id: str) -> Path:
        base = self.workflow_dir(workflow_id) / "tasks" / task_id / "artifacts"
        base.mkdir(parents=True, exist_ok=True)
        return base

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    @contextmanager
    def locked(self) -> Iterator[None]:
        """
        Hold the store lock across a multi-step mutation (e.g. update a
        task, save the workflow, append the event).
        """
        with self._lock:
            yield

    # ------------------------------------------------------------------
    # Workflow persistence
    # ------------------------------------------------------------------

    def save_workflow(self, workflow: Workflow) -> None:
        """Atomic write of the full workflow.json."""
        with self._lock:
            wf_dir = self.workflow_dir(workflow.workflow_id)
            wf_dir.mkdir(parents=True, exist_ok=True)
            target = self.workflow_json_path(workflow.workflow_id)
            tmp = target.with_suffix(".json.tmp")
            payload = json.dumps(workflow.to_dict(), indent=2, sort_keys=False)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            tmp.replace(target)

    def load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        with self._lock:
            path = self.workflow_json_path(workflow_id)
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Workflow.from_dict(data)

    def list_workflows(self) -> list[Workflow]:
        """
        Load every workflow directory found under root.

        Directories without a `workflow.json` are skipped (they may be
        mid-write). Phase 1 data volumes are small enough that loading
        the full payload is fine; switch to a summary cache later if
        this becomes hot.
        """
        with self._lock:
            workflows: list[Workflow] = []
            if not self._root.exists():
                return workflows
            for entry in sorted(self._root.iterdir()):
                if not entry.is_dir():
                    continue
                wf_path = entry / "workflow.json"
                if not wf_path.exists():
                    continue
                try:
                    with open(wf_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    workflows.append(Workflow.from_dict(data))
                except (json.JSONDecodeError, KeyError):
                    continue
            return workflows

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def append_event(self, event: WorkflowEvent) -> None:
        """Append one JSON line to the workflow's events.jsonl."""
        with self._lock:
            wf_dir = self.workflow_dir(event.workflow_id)
            wf_dir.mkdir(parents=True, exist_ok=True)
            path = self.events_jsonl_path(event.workflow_id)
            line = json.dumps(event.to_dict(), separators=(",", ":"))
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def load_events(
        self,
        workflow_id: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[WorkflowEvent]:
        """
        Read back events for a workflow, optionally filtered by
        timestamp range and/or task.
        """
        with self._lock:
            path = self.events_jsonl_path(workflow_id)
            if not path.exists():
                return []
            events: list[WorkflowEvent] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = data.get("timestamp", "")
                    if since and ts < since:
                        continue
                    if until and ts > until:
                        continue
                    if task_id and data.get("task_id") != task_id:
                        continue
                    events.append(WorkflowEvent.from_dict(data))
            if limit is not None:
                events = events[-limit:]
            return events

    # ------------------------------------------------------------------
    # Task artefacts
    # ------------------------------------------------------------------

    def write_result(
        self,
        workflow_id: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> Path:
        """Write the final result.json for a task. Atomic."""
        return self._write_json_file(
            self.task_result_path(workflow_id, task_id),
            payload,
        )

    def write_attempt_result(
        self,
        workflow_id: str,
        task_id: str,
        attempt_id: int,
        payload: dict[str, Any],
    ) -> Path:
        return self._write_json_file(
            self.task_attempt_result_path(workflow_id, task_id, attempt_id),
            payload,
        )

    def write_task_output(
        self,
        workflow_id: str,
        task_id: str,
        output: TaskOutput | dict[str, Any],
    ) -> Path:
        payload = output.to_dict() if isinstance(output, TaskOutput) else dict(output)
        return self._write_json_file(
            self.task_output_path(workflow_id, task_id),
            payload,
        )

    def load_task_output(self, workflow_id: str, task_id: str) -> Optional[TaskOutput]:
        payload = self._read_json_file(self.task_output_path(workflow_id, task_id))
        return TaskOutput.from_dict(payload) if payload is not None else None

    def write_task_inputs(
        self,
        workflow_id: str,
        task_id: str,
        inputs: list[ResolvedTaskInput | dict[str, Any]],
    ) -> Path:
        payload = [
            item.to_dict() if isinstance(item, ResolvedTaskInput) else dict(item)
            for item in inputs
        ]
        return self._write_json_file(
            self.task_inputs_path(workflow_id, task_id),
            payload,
        )

    def load_task_inputs(self, workflow_id: str, task_id: str) -> Optional[list[ResolvedTaskInput]]:
        payload = self._read_json_file(self.task_inputs_path(workflow_id, task_id))
        if payload is None:
            return None
        return [ResolvedTaskInput.from_dict(item) for item in payload]

    def write_materialized_prompt(self, workflow_id: str, task_id: str, prompt: str) -> Path:
        with self._lock:
            target = self.materialized_prompt_path(workflow_id, task_id)
            tmp = target.with_suffix(".txt.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(prompt)
            tmp.replace(target)
            return target

    def register_artifact(
        self,
        workflow_id: str,
        task_id: str,
        artifact: Artifact | dict[str, Any],
    ) -> Artifact:
        resolved = artifact if isinstance(artifact, Artifact) else Artifact.from_dict(artifact)
        self.task_artifacts_dir(workflow_id, task_id)
        return resolved

    def read_result(self, workflow_id: str, task_id: str) -> Optional[dict[str, Any]]:
        payload = self._read_json_file(self.task_result_path(workflow_id, task_id))
        if payload is not None:
            return payload
        workflow = self.load_workflow(workflow_id)
        if workflow is None:
            return None
        task = workflow.tasks.get(task_id)
        if task is None or not task.result_path:
            return None
        return self._read_json_file(Path(task.result_path))

    def _write_json_file(self, target: Path, payload: Any) -> Path:
        with self._lock:
            tmp = target.with_suffix(f"{target.suffix}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(target)
            return target

    def _read_json_file(self, path: Path) -> Optional[Any]:
        with self._lock:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
