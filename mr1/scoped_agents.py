"""
Persistent scoped-agent registry and workflow access helpers.

This module is separate from `mr1.agents`, which continues to describe
static runtime worker profiles such as `kazi`.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mr1.workflow_models import Workflow
    from mr1.workflow_store import WorkflowStore


_DEFAULT_ROOT = Path(__file__).resolve().parent / "memory" / "agents"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _report_ts(iso: Optional[str]) -> str:
    if not iso:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    digits = "".join(ch for ch in iso if ch.isdigit())
    return digits or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def new_agent_id() -> str:
    return f"ag-{_ts_compact()}-{uuid.uuid4().hex[:6]}"


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class PersistentAgent:
    agent_id: str
    agent_type: str
    title: str
    tree_level: int
    parent_agent_id: Optional[str]
    status: str = "active"
    created_at: str = field(default_factory=_now_iso)
    owned_workflow_ids: list[str] = field(default_factory=list)
    mission: Optional[str] = None
    mode: str = "manual"
    run_status: str = "idle"
    current_iteration: int = 0
    last_step_at: Optional[str] = None
    last_action: Optional[dict[str, Any]] = None
    parent_request: Optional[str] = None
    last_run: Optional[dict[str, Any]] = None
    step_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "title": self.title,
            "tree_level": self.tree_level,
            "parent_agent_id": self.parent_agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "owned_workflow_ids": list(self.owned_workflow_ids),
            "mission": self.mission,
            "mode": self.mode,
            "run_status": self.run_status,
            "current_iteration": self.current_iteration,
            "last_step_at": self.last_step_at,
            "last_action": dict(self.last_action) if self.last_action is not None else None,
            "parent_request": self.parent_request,
            "last_run": dict(self.last_run) if self.last_run is not None else None,
            "step_context": dict(self.step_context),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersistentAgent":
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            title=data["title"],
            tree_level=int(data["tree_level"]),
            parent_agent_id=data.get("parent_agent_id"),
            status=data.get("status", "active"),
            created_at=data.get("created_at", _now_iso()),
            owned_workflow_ids=list(data.get("owned_workflow_ids", [])),
            mission=data.get("mission"),
            mode=data.get("mode", "manual"),
            run_status=data.get("run_status", "idle"),
            current_iteration=int(data.get("current_iteration", 0)),
            last_step_at=data.get("last_step_at"),
            last_action=dict(data["last_action"])
            if isinstance(data.get("last_action"), dict) else None,
            parent_request=data.get("parent_request"),
            last_run=dict(data["last_run"])
            if isinstance(data.get("last_run"), dict) else None,
            step_context=dict(data.get("step_context") or {}),
        )


class AgentScopeError(ValueError):
    """Raised when an agent tree operation is outside the caller scope."""


class PersistentAgentStore:
    def __init__(self, root: Optional[Path] = None):
        self._root = Path(root) if root else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def root_agent_id_path(self) -> Path:
        return self._root / ".root_agent_id"

    def agent_path(self, agent_id: str) -> Path:
        return self._root / f"{agent_id}.json"

    def agent_dir(self, agent_id: str) -> Path:
        path = self._root / agent_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def memory_path(self, agent_id: str) -> Path:
        return self.agent_dir(agent_id) / "memory.md"

    def logs_dir(self, agent_id: str) -> Path:
        path = self.agent_dir(agent_id) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def step_log_path(self, agent_id: str) -> Path:
        return self.logs_dir(agent_id) / "steps.jsonl"

    def run_logs_dir(self, agent_id: str) -> Path:
        path = self.logs_dir(agent_id) / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_log_path(self, agent_id: str, run_id: str) -> Path:
        return self.run_logs_dir(agent_id) / f"{run_id}.json"

    def run_summary_log_path(self, agent_id: str) -> Path:
        return self.logs_dir(agent_id) / "runs.jsonl"

    def report_dir(self, agent_id: str) -> Path:
        path = self.agent_dir(agent_id) / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_agent_files(self, agent_id: str) -> None:
        self.agent_dir(agent_id)
        self.logs_dir(agent_id)
        self.run_logs_dir(agent_id)
        self.report_dir(agent_id)
        memory_path = self.memory_path(agent_id)
        if not memory_path.exists():
            memory_path.write_text("", encoding="utf-8")

    def save_agent(self, agent: PersistentAgent) -> None:
        with self._lock:
            self.ensure_agent_files(agent.agent_id)
            target = self.agent_path(agent.agent_id)
            tmp = target.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(agent.to_dict(), handle, indent=2)
            tmp.replace(target)

    def load_agent(self, agent_id: str) -> Optional[PersistentAgent]:
        with self._lock:
            path = self.agent_path(agent_id)
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as handle:
                agent = PersistentAgent.from_dict(json.load(handle))
            self.ensure_agent_files(agent.agent_id)
            return agent

    def require_agent(self, agent_id: str) -> PersistentAgent:
        agent = self.load_agent(agent_id)
        if agent is None:
            raise ValueError(f"agent not found: {agent_id}")
        return agent

    def list_agents(self) -> list[PersistentAgent]:
        with self._lock:
            agents: list[PersistentAgent] = []
            for path in sorted(self._root.glob("ag-*.json")):
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        agent = PersistentAgent.from_dict(json.load(handle))
                    self.ensure_agent_files(agent.agent_id)
                    agents.append(agent)
                except (OSError, json.JSONDecodeError, KeyError, ValueError):
                    continue
            agents.sort(key=lambda item: (item.tree_level, item.created_at, item.agent_id))
            return agents

    def ensure_root_agent(self) -> PersistentAgent:
        with self._lock:
            pointer = self.root_agent_id_path
            if pointer.exists():
                agent_id = pointer.read_text(encoding="utf-8").strip()
                if agent_id:
                    existing = self.load_agent(agent_id)
                    if existing is not None:
                        return existing
            agent = PersistentAgent(
                agent_id=new_agent_id(),
                agent_type="mr1",
                title="MR1",
                tree_level=1,
                parent_agent_id=None,
            )
            self.save_agent(agent)
            tmp = pointer.with_suffix(".tmp")
            tmp.write_text(agent.agent_id, encoding="utf-8")
            tmp.replace(pointer)
            return agent

    @property
    def root_agent_id(self) -> str:
        return self.ensure_root_agent().agent_id

    def is_root_agent(self, agent_id: str) -> bool:
        return agent_id == self.root_agent_id

    def is_terminated(self, agent_id: str) -> bool:
        return self.require_agent(agent_id).status == "terminated"

    def descendant_ids(self, agent_id: str) -> set[str]:
        all_agents = {agent.agent_id: agent for agent in self.list_agents()}
        descendants: set[str] = set()
        frontier = [agent_id]
        while frontier:
            current = frontier.pop()
            for candidate in all_agents.values():
                if candidate.parent_agent_id != current:
                    continue
                if candidate.agent_id in descendants:
                    continue
                descendants.add(candidate.agent_id)
                frontier.append(candidate.agent_id)
        return descendants

    def is_visible(self, caller_agent_id: str, target_agent_id: str) -> bool:
        if self.is_root_agent(caller_agent_id):
            return self.load_agent(target_agent_id) is not None
        return target_agent_id == caller_agent_id or target_agent_id in self.descendant_ids(caller_agent_id)

    def list_visible_agents(self, caller_agent_id: str) -> list[PersistentAgent]:
        return [
            agent
            for agent in self.list_agents()
            if self.is_visible(caller_agent_id, agent.agent_id)
        ]

    def get_visible_agent(self, caller_agent_id: str, target_agent_id: str) -> PersistentAgent:
        agent = self.require_agent(target_agent_id)
        if not self.is_visible(caller_agent_id, target_agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        return agent

    def can_manage_agent(self, caller_agent_id: str, target_agent_id: str) -> bool:
        if self.is_root_agent(caller_agent_id):
            return self.load_agent(target_agent_id) is not None
        return target_agent_id == caller_agent_id or target_agent_id in self.descendant_ids(caller_agent_id)

    def create_child_agent(self, caller_agent_id: str, title: str) -> PersistentAgent:
        title = title.strip()
        if not title:
            raise ValueError("agent title must be non-empty")
        parent = self.require_agent(caller_agent_id)
        if parent.status == "terminated":
            raise ValueError(f"agent is terminated: {caller_agent_id}")
        agent = PersistentAgent(
            agent_id=new_agent_id(),
            agent_type="mrn",
            title=title,
            tree_level=parent.tree_level + 1,
            parent_agent_id=parent.agent_id,
        )
        self.save_agent(agent)
        return agent

    def terminate_agent(self, caller_agent_id: str, target_agent_id: str) -> PersistentAgent:
        if not self.can_manage_agent(caller_agent_id, target_agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        agent = self.require_agent(target_agent_id)
        if agent.status != "terminated":
            agent.status = "terminated"
            self.save_agent(agent)
        return agent

    def append_owned_workflow(self, agent_id: str, workflow_id: str) -> None:
        agent = self.require_agent(agent_id)
        if workflow_id not in agent.owned_workflow_ids:
            agent.owned_workflow_ids.append(workflow_id)
            self.save_agent(agent)

    def assign_mission(
        self,
        caller_agent_id: str,
        target_agent_id: str,
        mission: str,
    ) -> PersistentAgent:
        if not self.can_manage_agent(caller_agent_id, target_agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        agent = self.require_agent(target_agent_id)
        if agent.status == "terminated":
            raise ValueError(f"agent terminated: {target_agent_id}")
        agent.mission = mission.strip() if isinstance(mission, str) else None
        agent.run_status = "idle"
        agent.current_iteration = 0
        agent.last_step_at = None
        agent.last_action = None
        agent.parent_request = None
        self.save_agent(agent)
        return agent

    def read_memory(self, agent_id: str) -> str:
        path = self.memory_path(agent_id)
        with self._lock:
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return ""

    def write_report(
        self,
        agent_id: str,
        content: str,
        *,
        timestamp: Optional[str] = None,
    ) -> Path:
        with self._lock:
            path = self.report_dir(agent_id) / f"{_report_ts(timestamp)}.md"
            tmp = path.with_suffix(".md.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(content.rstrip() + "\n")
            tmp.replace(path)
            return path

    def capability_call_log_path(self, agent_id: str) -> Path:
        return self.logs_dir(agent_id) / "capability_calls.jsonl"

    def append_capability_call_log(self, agent_id: str, record: dict[str, Any]) -> Path:
        with self._lock:
            path = self.capability_call_log_path(agent_id)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return path

    def append_step_log(self, agent_id: str, record: dict[str, Any]) -> Path:
        with self._lock:
            path = self.step_log_path(agent_id)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return path

    def write_run_log(self, agent_id: str, run_id: str, record: dict[str, Any]) -> Path:
        with self._lock:
            path = self.run_log_path(agent_id, run_id)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
            tmp.replace(path)
            return path

    def append_run_summary(self, agent_id: str, record: dict[str, Any]) -> Path:
        with self._lock:
            path = self.run_summary_log_path(agent_id)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return path

    def normalize_workflow_ownership(self, workflow: "Workflow") -> "Workflow":
        root = self.ensure_root_agent()
        if getattr(workflow, "owner_agent_id", None) is None:
            workflow.owner_agent_id = root.agent_id
        owner = self.load_agent(workflow.owner_agent_id) or root
        if getattr(workflow, "owner_agent_title", None) is None:
            workflow.owner_agent_title = owner.title
        if not hasattr(workflow, "parent_agent_id") or workflow.parent_agent_id is None:
            workflow.parent_agent_id = owner.parent_agent_id
        return workflow

    def can_agent_access_workflow(self, agent_id: str, workflow: "Workflow") -> bool:
        self.normalize_workflow_ownership(workflow)
        if self.is_root_agent(agent_id):
            return True
        return (
            workflow.owner_agent_id == agent_id
            or workflow.owner_agent_id in self.descendant_ids(agent_id)
        )

    def list_reports(self, agent_id: str) -> list[Path]:
        report_dir = self.report_dir(agent_id)
        return sorted(report_dir.glob("*.md"), reverse=True)

    def write_workflow_report(self, workflow: "Workflow", store: "WorkflowStore") -> Optional[Path]:
        workflow = self.normalize_workflow_ownership(workflow)
        owner = self.load_agent(workflow.owner_agent_id)
        if owner is None or owner.agent_type != "mrn":
            return None
        path = self.report_dir(owner.agent_id) / f"{_report_ts(workflow.finished_at)}.md"
        if path.exists():
            return None
        lines = [
            f"# Workflow Report: {workflow.title}",
            "",
            f"- workflow_id: {workflow.workflow_id}",
            f"- owner_agent_id: {workflow.owner_agent_id}",
            f"- owner_agent_title: {workflow.owner_agent_title or owner.title}",
            f"- status: {workflow.status.value}",
            f"- finished_at: {workflow.finished_at or '-'}",
            "",
            "## Tasks",
        ]
        for label, task_id in workflow.label_to_task_id.items():
            task = workflow.tasks.get(task_id)
            if task is None:
                continue
            lines.append(
                f"- {label}: {task.status.value}"
                + (f" | {task.result_summary}" if task.result_summary else "")
            )
        lines.append("")
        lines.append("## Key Outputs")
        emitted_output = False
        for label, task_id in workflow.label_to_task_id.items():
            task = workflow.tasks.get(task_id)
            if task is None:
                continue
            output = store.load_task_output(workflow.workflow_id, task.task_id)
            if output is None:
                continue
            emitted_output = True
            lines.extend([
                f"### {label}",
                f"- status: {output.status}",
                f"- summary: {output.summary or '-'}",
                f"- text: {(output.text or '').strip()[:500] or '-'}",
                "",
            ])
        if not emitted_output:
            lines.append("- none")
        tmp = path.with_suffix(".md.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines).rstrip() + "\n")
        tmp.replace(path)
        return path
