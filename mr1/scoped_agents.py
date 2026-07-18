"""
Scoped agent registry and workflow access helpers.

Every record here has three independent properties: `role` (worker |
orchestrator), `mr_level` (depth in the MR1 hierarchy; MR1 itself is
mr_level=1), and `lifecycle` (ephemeral | task_scoped | project_scoped |
standing). See docs/architecture/AGENT_ONTOLOGY.md for the full model.

This module is separate from `mr1.agents`, which continues to describe
static runtime worker profiles such as `worker`.
"""

from __future__ import annotations

import json
import threading
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from mr1.event_log import EventLog

if TYPE_CHECKING:
    from mr1.workflow_models import Workflow
    from mr1.workflow_store import WorkflowStore


_DEFAULT_ROOT = Path(__file__).resolve().parent / "memory" / "agents"
ACTIVE_AGENT_STATUS = "active"
TERMINAL_AGENT_STATUSES = frozenset({"terminated", "archived", "dead"})
TERMINAL_RUN_STATUSES = frozenset({"terminated", "completed", "failed", "dead"})
MAX_AUTONOMOUS_CLEARANCE = 0.99

# Index file mapping normalized_casefold_title → agent_id.
# Terminated agents remain in the index: their titles are permanently reserved
# to prevent reuse. This is intentional and matches the current semantics.
_TITLE_INDEX_FILE = ".title_index.json"


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


def _normalize_lifecycle_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _title_has_control_chars(title: str) -> bool:
    return any(
        ch == "\x7f" or unicodedata.category(ch) == "Cc"
        for ch in title
    )


def _normalize_agent_title(title: str) -> str:
    return title.strip().casefold()


def derive_lifecycle_state(
    status: str | None,
    run_status: str | None,
) -> dict[str, Any]:
    normalized_status = _normalize_lifecycle_value(status)
    normalized_run_status = _normalize_lifecycle_value(run_status)
    status_is_terminal = normalized_status in TERMINAL_AGENT_STATUSES
    run_is_terminal = normalized_run_status in TERMINAL_RUN_STATUSES
    status_conflict = normalized_status == ACTIVE_AGENT_STATUS and run_is_terminal
    is_terminal = status_is_terminal or run_is_terminal
    is_live = normalized_status == ACTIVE_AGENT_STATUS and not run_is_terminal
    lifecycle_status = normalized_status or "-"
    if status_is_terminal:
        lifecycle_status = normalized_status
    elif status_conflict:
        lifecycle_status = "terminated"
    elif normalized_status:
        lifecycle_status = normalized_status
    elif run_is_terminal:
        lifecycle_status = "terminated"
    elif normalized_run_status:
        lifecycle_status = normalized_run_status
    return {
        "status": normalized_status or None,
        "run_status": normalized_run_status or None,
        "lifecycle_status": lifecycle_status,
        "is_live": is_live,
        "is_terminal": is_terminal,
        "status_conflict": status_conflict,
    }


# NOTE: `derive_lifecycle_state`'s "lifecycle_status" (active/idle/working/
# terminated, derived from `status` + `run_status`) is a *runtime activity*
# concept, distinct from `AgentRecord.lifecycle` (ephemeral/task_scoped/
# project_scoped/standing), which is a *design-time expected duration*
# property. The two are orthogonal; see docs/architecture/AGENT_ONTOLOGY.md.


def derive_agent_lifecycle(agent: "AgentRecord") -> dict[str, Any]:
    return derive_lifecycle_state(agent.status, agent.run_status)


def is_agent_terminal(agent: "AgentRecord") -> bool:
    return bool(derive_agent_lifecycle(agent)["is_terminal"])


def is_agent_live(agent: "AgentRecord") -> bool:
    return bool(derive_agent_lifecycle(agent)["is_live"])


def agent_display_status(agent: "AgentRecord") -> str:
    return str(derive_agent_lifecycle(agent)["lifecycle_status"])


def _clone_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clone_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_json_value(item) for item in value]
    return value


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_id_list(items: Any, *, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(items, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        candidate: Optional[str] = None
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            for key in keys:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    candidate = value.strip()
                    break
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _normalize_string_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _normalize_assignment_context(context: Any) -> dict[str, Any]:
    if not isinstance(context, dict):
        context = {}
    return {
        "assigned_clearance": context.get("assigned_clearance"),
        "agents": _normalize_id_list(
            context.get("agents"),
            keys=("agent_id", "id"),
        ),
        "messages": _normalize_id_list(
            context.get("messages"),
            keys=("message_id", "id"),
        ),
        "workflows": _normalize_id_list(
            context.get("workflows"),
            keys=("workflow_id", "id"),
        ),
        "delegated_subtask": _clean_text(context.get("delegated_subtask")),
        "reason_for_delegation": _clean_text(context.get("reason_for_delegation")),
        "mission": _clean_text(context.get("mission")),
        "responsibility": _clean_text(context.get("responsibility")),
        "in_scope": _normalize_string_list(context.get("in_scope")),
        "out_of_scope": _normalize_string_list(context.get("out_of_scope")),
        "constraints": _normalize_string_list(context.get("constraints")),
        "success_criteria": _normalize_string_list(context.get("success_criteria")),
        "expected_first_action": _clean_text(context.get("expected_first_action")),
        "escalation_rules": _clean_text(context.get("escalation_rules")),
    }


def build_assignment_packet(
    parent_agent: "AgentRecord",
    child_title: str,
    user_input: str,
    context: Any,
) -> dict[str, Any]:
    normalized_context = _normalize_assignment_context(context)
    normalized_title = _clean_text(child_title) or "Unnamed agent"
    full_parent_request = user_input if isinstance(user_input, str) else ""
    delegated_subtask = (
        normalized_context["delegated_subtask"]
        or _clean_text(user_input)
        or f"Own the delegated responsibility for {normalized_title}."
    )
    reason_for_delegation = (
        normalized_context["reason_for_delegation"]
        or "The parent agent delegated this work so the child can own a scoped responsibility and execute it persistently."
    )
    mission = (
        normalized_context["mission"]
        or f"Own the delegated subtask for {normalized_title} and deliver the requested outcome within the defined scope."
    )
    responsibility = (
        normalized_context["responsibility"]
        or "Own the requested domain/responsibility instead of treating it as a one-shot execution."
    )
    in_scope = normalized_context["in_scope"] or [
        delegated_subtask,
        "Plan, execute, and report on the delegated work within the child agent's scope.",
        "Create or manage workflows when structured execution is the best fit.",
    ]
    out_of_scope = normalized_context["out_of_scope"] or [
        "Do not exceed the delegated scope or assigned clearance.",
        "Do not redesign permissions, memory retrieval, or unrelated runtime systems unless explicitly delegated.",
        "Escalate instead of inventing missing decisions that materially change the assignment.",
    ]
    constraints = normalized_context["constraints"] or [
        "Preserve the full parent request without truncation.",
        "Treat assigned_clearance as informational; enforcement is handled elsewhere.",
        "Use relevant runtime references by ID only and fetch details through observation reads when needed.",
        "Prefer workflow creation when execution should become structured work.",
    ]
    success_criteria = normalized_context["success_criteria"] or [
        "The child can explain its mission, responsibility, scope boundaries, and escalation behavior.",
        "The child uses relevant context IDs instead of embedded full runtime objects.",
        "The delegated work stays within the assigned scope and escalates when blocked by missing decisions or clearance.",
    ]
    expected_first_action = (
        normalized_context["expected_first_action"]
        or "Review the full parent request and relevant context IDs, then either start direct work or create a workflow if structured execution is warranted."
    )
    escalation_rules = (
        normalized_context["escalation_rules"]
        or "Escalate to the parent when a missing decision, approval, or scope boundary blocks progress or would materially change the assignment."
    )
    assigned_clearance_raw = normalized_context["assigned_clearance"]
    assigned_clearance = (
        float(assigned_clearance_raw)
        if assigned_clearance_raw is not None else
        float(parent_agent.security_clearance)
    )
    return {
        "parent_agent_id": parent_agent.agent_id,
        "parent_level": int(parent_agent.mr_level),
        "child_level": int(parent_agent.mr_level) + 1,
        "assigned_clearance": assigned_clearance,
        "child_title": normalized_title,
        "full_parent_request": full_parent_request,
        "delegated_subtask": delegated_subtask,
        "reason_for_delegation": reason_for_delegation,
        "mission": mission,
        "responsibility": responsibility,
        "scope": {
            "in_scope": in_scope,
            "out_of_scope": out_of_scope,
        },
        "constraints": constraints,
        "success_criteria": success_criteria,
        "expected_first_action": expected_first_action,
        "relevant_context": {
            "agents": list(normalized_context["agents"]),
            "messages": list(normalized_context["messages"]),
            "workflows": list(normalized_context["workflows"]),
        },
        "escalation_rules": escalation_rules,
        "notes": "clearance is informational; enforcement handled elsewhere",
    }


def render_assignment_mission(assignment_packet: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(assignment_packet, dict):
        return None
    child_title = _clean_text(assignment_packet.get("child_title")) or "Unnamed agent"
    mission = _clean_text(assignment_packet.get("mission")) or "Own the delegated assignment."
    responsibility = _clean_text(assignment_packet.get("responsibility")) or "-"
    delegated_subtask = _clean_text(assignment_packet.get("delegated_subtask")) or "-"
    expected_first_action = _clean_text(assignment_packet.get("expected_first_action")) or "-"
    escalation_rules = _clean_text(assignment_packet.get("escalation_rules")) or "-"
    assigned_clearance = float(assignment_packet.get("assigned_clearance", 0.0))
    scope = assignment_packet.get("scope") if isinstance(assignment_packet.get("scope"), dict) else {}
    in_scope = _normalize_string_list(scope.get("in_scope"))
    out_of_scope = _normalize_string_list(scope.get("out_of_scope"))
    constraints = _normalize_string_list(assignment_packet.get("constraints"))
    success_criteria = _normalize_string_list(assignment_packet.get("success_criteria"))
    lines = [
        f"You are {child_title}, an orchestrator agent owning a scoped assignment.",
        f"Mission: {mission}",
        f"Responsibility: {responsibility}",
        f"Delegated subtask: {delegated_subtask}",
        f"Assigned clearance: {assigned_clearance:.2f}",
    ]
    if in_scope:
        lines.append("In scope:")
        lines.extend(f"- {item}" for item in in_scope)
    if out_of_scope:
        lines.append("Out of scope:")
        lines.extend(f"- {item}" for item in out_of_scope)
    if constraints:
        lines.append("Constraints:")
        lines.extend(f"- {item}" for item in constraints)
    if success_criteria:
        lines.append("Success criteria:")
        lines.extend(f"- {item}" for item in success_criteria)
    lines.append(f"Expected first action: {expected_first_action}")
    lines.append(f"Escalation: {escalation_rules}")
    return "\n".join(lines)


ALLOWED_ROLES = frozenset({"worker", "orchestrator"})
ALLOWED_LIFECYCLES = frozenset({"ephemeral", "task_scoped", "project_scoped", "standing"})

# Legacy `agent_type` values, retired as a category name but still needed to
# interpret pre-migration JSON. See AgentRecord.from_dict / migrate_legacy_agent_dict.
_LEGACY_AGENT_TYPE_TO_ROLE = {"mr1": "orchestrator", "mrn": "orchestrator", "kazi": "worker"}


def actor_category(role: str, mr_level: int) -> str:
    """Collapse (role, mr_level) into the 3-tier category capability_policy prices
    clearance against. MR1 (mr_level==1) is strictly more trusted than any other
    orchestrator at a deeper level — that distinction predates this refactor and is
    preserved here, not reintroduced as a new behavior."""
    if role == "orchestrator" and mr_level == 1:
        return "root_orchestrator"
    return role


@dataclass
class AgentRecord:
    agent_id: str
    role: str
    title: str
    mr_level: int
    parent_agent_id: Optional[str]
    lifecycle: str = "project_scoped"
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
    security_clearance: float = MAX_AUTONOMOUS_CLEARANCE
    scope_roots: list[str] = field(default_factory=list)
    scope_grants: list[dict[str, Any]] = field(default_factory=list)
    assignment_packet: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.security_clearance) <= 1.0:
            raise ValueError("security_clearance must be between 0.0 and 1.0")
        if self.role not in ALLOWED_ROLES:
            raise ValueError(f"invalid role: {self.role!r} (expected one of {sorted(ALLOWED_ROLES)})")
        if self.lifecycle not in ALLOWED_LIFECYCLES:
            raise ValueError(
                f"invalid lifecycle: {self.lifecycle!r} (expected one of {sorted(ALLOWED_LIFECYCLES)})"
            )
        if self.mr_level < 1:
            raise ValueError(f"mr_level must be >= 1, got {self.mr_level}")
        if self.mr_level == 1 and self.parent_agent_id is not None:
            raise ValueError("mr_level=1 (root) must not have a parent_agent_id")

    @property
    def actor_category(self) -> str:
        return actor_category(self.role, self.mr_level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "title": self.title,
            "mr_level": self.mr_level,
            "parent_agent_id": self.parent_agent_id,
            "lifecycle": self.lifecycle,
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
            "security_clearance": float(self.security_clearance),
            "scope_roots": list(self.scope_roots),
            "scope_grants": [dict(item) for item in self.scope_grants],
            "assignment_packet": _clone_json_value(self.assignment_packet)
            if isinstance(self.assignment_packet, dict) else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRecord":
        data = migrate_legacy_agent_dict(data)
        security_clearance = data.get("security_clearance")
        if security_clearance is None and data["role"] == "orchestrator":
            security_clearance = MAX_AUTONOMOUS_CLEARANCE
        return cls(
            agent_id=data["agent_id"],
            role=data["role"],
            title=data["title"],
            mr_level=int(data["mr_level"]),
            parent_agent_id=data.get("parent_agent_id"),
            lifecycle=data.get("lifecycle", "project_scoped"),
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
            security_clearance=float(security_clearance or 0.0),
            scope_roots=list(data.get("scope_roots", [])),
            scope_grants=[
                dict(item)
                for item in list(data.get("scope_grants", []))
                if isinstance(item, dict)
            ],
            assignment_packet=_clone_json_value(data["assignment_packet"])
            if isinstance(data.get("assignment_packet"), dict) else None,
        )


def migrate_legacy_agent_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Map a pre-ontology-refactor agent JSON payload (`agent_type` + `tree_level`,
    no `role`/`mr_level`/`lifecycle`) onto the current schema. Idempotent: a
    payload that already has `role`/`mr_level` is returned unchanged (aside from
    a shallow copy). Raises on legacy data this mapping can't interpret rather
    than silently dropping or misclassifying an agent."""
    if "role" in data and "mr_level" in data:
        return data
    data = dict(data)
    legacy_agent_type = data.pop("agent_type", None)
    if legacy_agent_type is not None:
        if legacy_agent_type not in _LEGACY_AGENT_TYPE_TO_ROLE:
            raise ValueError(f"cannot migrate unknown legacy agent_type: {legacy_agent_type!r}")
        data.setdefault("role", _LEGACY_AGENT_TYPE_TO_ROLE[legacy_agent_type])
    if "tree_level" in data:
        data["mr_level"] = data.pop("tree_level")
    if "role" not in data:
        raise ValueError("agent record has neither 'role' nor a recognizable legacy 'agent_type'")
    if "mr_level" not in data:
        raise ValueError("agent record has neither 'mr_level' nor a legacy 'tree_level'")
    if "lifecycle" not in data:
        data["lifecycle"] = "standing" if int(data["mr_level"]) == 1 else "project_scoped"
    return data


def migrate_agent_store_ontology(root: Path) -> dict[str, Any]:
    """One-time, idempotent migration of every agent JSON file under `root`
    (an agent store's root directory, e.g. `<runtime_root>/agents`) from the
    pre-ontology-refactor schema (`agent_type` + `tree_level`) to the current
    one (`role` + `mr_level` + `lifecycle`).

    Safe to run repeatedly: a file already in the new shape is left
    untouched (not even rewritten), so re-running finds nothing to migrate.
    Never deletes an agent file; a file that can't be migrated is reported
    in `failed` and left exactly as it was found.
    """
    migrated: list[str] = []
    already_current: list[str] = []
    failed: list[dict[str, str]] = []
    for path in sorted(Path(root).glob("ag-*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            failed.append({"agent_id": path.stem, "error": f"unreadable: {exc}"})
            continue
        if "role" in raw and "mr_level" in raw:
            already_current.append(path.stem)
            continue
        try:
            agent = AgentRecord.from_dict(raw)
        except (KeyError, ValueError) as exc:
            failed.append({"agent_id": path.stem, "error": str(exc)})
            continue
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(agent.to_dict(), handle, indent=2)
        tmp.replace(path)
        migrated.append(agent.agent_id)
    return {
        "root": str(root),
        "migrated": migrated,
        "already_current": already_current,
        "failed": failed,
    }


@dataclass(frozen=True)
class ScopeGrant:
    agent_id: str
    path: str
    granted_by: str
    reason: str
    timestamp: str

    def to_dict(self) -> dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "path": self.path,
            "granted_by": self.granted_by,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScopeGrant":
        return cls(
            agent_id=str(data["agent_id"]),
            path=str(data["path"]),
            granted_by=str(data["granted_by"]),
            reason=str(data["reason"]),
            timestamp=str(data["timestamp"]),
        )


class AgentScopeError(ValueError):
    """Raised when an agent tree operation is outside the caller scope."""


class AgentStore:
    def __init__(self, root: Optional[Path] = None):
        self._root = Path(root) if root else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._event_log = EventLog(self._root.parent / "events")

    @property
    def root(self) -> Path:
        return self._root

    @property
    def _title_index_path(self) -> Path:
        return self._root / _TITLE_INDEX_FILE

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

    def step_artifacts_dir(self, agent_id: str) -> Path:
        path = self.logs_dir(agent_id) / "steps"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def step_prompt_artifact_path(
        self,
        agent_id: str,
        step_id: str,
        *,
        attempt: str = "primary",
    ) -> Path:
        safe_step_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in step_id)
        suffix = "" if attempt == "primary" else f".{attempt}"
        return self.step_artifacts_dir(agent_id) / f"{safe_step_id}{suffix}.prompt.json"

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
        self.step_artifacts_dir(agent_id)
        self.run_logs_dir(agent_id)
        self.report_dir(agent_id)
        memory_path = self.memory_path(agent_id)
        if not memory_path.exists():
            memory_path.write_text("", encoding="utf-8")

    def _read_agent_file(self, path: Path) -> Optional[AgentRecord]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return AgentRecord.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    def _load_title_index_locked(self) -> dict[str, str]:
        """Return the title index {normalized_title: agent_id}, rebuilding if missing/corrupt."""
        path = self._title_index_path
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if isinstance(raw, dict):
                    return {str(k): str(v) for k, v in raw.items()}
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        return self._rebuild_title_index_locked()

    def _rebuild_title_index_locked(self) -> dict[str, str]:
        """Full scan → rebuild title index. O(n). Terminated agents are included
        so their titles remain permanently reserved (same semantics as before)."""
        index: dict[str, str] = {}
        for path in sorted(self._root.glob("ag-*.json")):
            agent = self._read_agent_file(path)
            if agent is None:
                continue
            normalized = _normalize_agent_title(agent.title)
            if normalized:
                index[normalized] = agent.agent_id
        self._save_title_index_locked(index)
        return index

    def _save_title_index_locked(self, index: dict[str, str]) -> None:
        path = self._title_index_path
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
        tmp.replace(path)

    def _find_title_conflict_locked(
        self,
        title: str,
        *,
        excluding_agent_id: str | None = None,
    ) -> Optional[AgentRecord]:
        """O(1) title conflict check via the title index (one index file read + at most
        one agent file read to verify). Falls back to a full rebuild if the index is
        stale (i.e., the indexed agent no longer has that title)."""
        normalized_title = _normalize_agent_title(title)
        if not normalized_title:
            return None
        index = self._load_title_index_locked()
        conflicting_id = index.get(normalized_title)
        if conflicting_id is None:
            return None
        if excluding_agent_id and conflicting_id == excluding_agent_id:
            return None
        # Verify the indexed agent still holds this title (self-healing for stale entries).
        candidate = self._read_agent_file(self.agent_path(conflicting_id))
        if candidate is None or _normalize_agent_title(candidate.title) != normalized_title:
            index.pop(normalized_title, None)
            self._save_title_index_locked(index)
            return None
        return candidate

    def save_agent(self, agent: AgentRecord) -> None:
        with self._lock:
            target = self.agent_path(agent.agent_id)
            existing = self._read_agent_file(target) if target.exists() else None
            title_changed = (
                existing is None
                or _normalize_agent_title(existing.title) != _normalize_agent_title(agent.title)
            )
            old_normalized = (
                _normalize_agent_title(existing.title) if existing is not None else None
            )
            new_normalized = _normalize_agent_title(agent.title)
            if title_changed:
                conflict = self._find_title_conflict_locked(
                    agent.title,
                    excluding_agent_id=agent.agent_id,
                )
                if conflict is not None:
                    raise ValueError(f"agent title already exists: {conflict.title}")
            self.ensure_agent_files(agent.agent_id)
            tmp = target.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(agent.to_dict(), handle, indent=2)
            tmp.replace(target)
            # Update title index atomically after the agent file is persisted.
            if title_changed:
                index = self._load_title_index_locked()
                if old_normalized is not None and index.get(old_normalized) == agent.agent_id:
                    del index[old_normalized]
                index[new_normalized] = agent.agent_id
                self._save_title_index_locked(index)

    def load_agent(self, agent_id: str) -> Optional[AgentRecord]:
        with self._lock:
            path = self.agent_path(agent_id)
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as handle:
                agent = AgentRecord.from_dict(json.load(handle))
            self.ensure_agent_files(agent.agent_id)
            return agent

    def require_agent(self, agent_id: str) -> AgentRecord:
        agent = self.load_agent(agent_id)
        if agent is None:
            raise ValueError(f"agent not found: {agent_id}")
        return agent

    def list_agents(self) -> list[AgentRecord]:
        with self._lock:
            agents: list[AgentRecord] = []
            for path in sorted(self._root.glob("ag-*.json")):
                agent = self._read_agent_file(path)
                if agent is None:
                    continue
                self.ensure_agent_files(agent.agent_id)
                agents.append(agent)
            agents.sort(key=lambda item: (item.mr_level, item.created_at, item.agent_id))
            return agents

    def ensure_root_agent(self) -> AgentRecord:
        with self._lock:
            pointer = self.root_agent_id_path
            if pointer.exists():
                agent_id = pointer.read_text(encoding="utf-8").strip()
                if agent_id:
                    existing = self.load_agent(agent_id)
                    if existing is not None:
                        if is_agent_terminal(existing):
                            existing.status = ACTIVE_AGENT_STATUS
                            existing.run_status = "idle"
                            self.save_agent(existing)
                        return existing
            agent = AgentRecord(
                agent_id=new_agent_id(),
                role="orchestrator",
                title="MR1",
                mr_level=1,
                parent_agent_id=None,
                lifecycle="standing",
                security_clearance=MAX_AUTONOMOUS_CLEARANCE,
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
        return is_agent_terminal(self.require_agent(agent_id))

    def workspace_root(self) -> Path:
        return self._root.parent

    def normalized_scope_roots(self, agent_id: str) -> list[str]:
        from mr1.capability_policy import normalize_path

        agent = self.require_agent(agent_id)
        return sorted(
            {
                str(normalize_path(item))
                for item in list(agent.scope_roots or [])
            }
        )

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

    def ancestor_ids(self, agent_id: str, *, include_self: bool = False) -> list[str]:
        lineage: list[str] = []
        current_id = agent_id if include_self else self.require_agent(agent_id).parent_agent_id
        while current_id:
            lineage.append(current_id)
            current = self.require_agent(current_id)
            current_id = current.parent_agent_id
        return lineage

    def is_ancestor(self, maybe_ancestor_id: str, agent_id: str, *, include_self: bool = False) -> bool:
        if include_self and maybe_ancestor_id == agent_id:
            return True
        return maybe_ancestor_id in self.ancestor_ids(agent_id)

    def can_agent_access_path(self, agent_id: str, path: str | Path) -> bool:
        from mr1.capability_policy import ScopeContext, normalize_path

        normalized = normalize_path(path)
        if self.is_root_agent(agent_id):
            scope = ScopeContext(
                allowed_roots=[self.workspace_root()],
                workspace_root=self.workspace_root(),
            )
            return scope.contains(normalized)
        scope = ScopeContext(
            allowed_roots=[Path(item) for item in self.normalized_scope_roots(agent_id)],
            workspace_root=self.workspace_root(),
        )
        return scope.contains(normalized)

    def is_visible(self, caller_agent_id: str, target_agent_id: str) -> bool:
        if self.is_root_agent(caller_agent_id):
            return self.load_agent(target_agent_id) is not None
        return target_agent_id == caller_agent_id or target_agent_id in self.descendant_ids(caller_agent_id)

    def list_visible_agents(self, caller_agent_id: str) -> list[AgentRecord]:
        return [
            agent
            for agent in self.list_agents()
            if self.is_visible(caller_agent_id, agent.agent_id)
        ]

    def get_visible_agent(self, caller_agent_id: str, target_agent_id: str) -> AgentRecord:
        agent = self.require_agent(target_agent_id)
        if not self.is_visible(caller_agent_id, target_agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        return agent

    def can_manage_agent(self, caller_agent_id: str, target_agent_id: str) -> bool:
        if self.is_root_agent(caller_agent_id):
            return self.load_agent(target_agent_id) is not None
        return target_agent_id == caller_agent_id or target_agent_id in self.descendant_ids(caller_agent_id)

    def create_child_agent(
        self,
        caller_agent_id: str,
        title: str,
        *,
        security_clearance: Optional[float] = None,
        lifecycle: str = "project_scoped",
    ) -> AgentRecord:
        title = title.strip()
        if not title:
            raise ValueError("agent title must be non-empty")
        if title.casefold() == "all":
            raise ValueError("agent title 'all' is reserved")
        if _title_has_control_chars(title):
            raise ValueError("agent title must not contain control characters")
        if lifecycle not in ALLOWED_LIFECYCLES:
            raise ValueError(f"invalid lifecycle: {lifecycle!r} (expected one of {sorted(ALLOWED_LIFECYCLES)})")
        parent = self.require_agent(caller_agent_id)
        if is_agent_terminal(parent):
            raise ValueError(f"agent is terminated: {caller_agent_id}")
        child_clearance = parent.security_clearance if security_clearance is None else float(security_clearance)
        if not 0.0 <= child_clearance <= 1.0:
            raise ValueError("security_clearance must be between 0.0 and 1.0")
        if child_clearance > parent.security_clearance:
            raise ValueError("child security_clearance cannot exceed parent.security_clearance")
        agent = AgentRecord(
            agent_id=new_agent_id(),
            role="orchestrator",
            title=title,
            mr_level=parent.mr_level + 1,
            parent_agent_id=parent.agent_id,
            lifecycle=lifecycle,
            security_clearance=child_clearance,
        )
        self.save_agent(agent)
        self._event_log.emit(
            event_type="agent_created",
            actor_id=parent.agent_id,
            actor_type=parent.actor_category,
            target_id=agent.agent_id,
            target_type="agent",
            status="created",
            summary=f"agent created: {agent.title}",
            record_path=str(self.agent_path(agent.agent_id)),
            metadata={
                "title": agent.title,
                "mr_level": agent.mr_level,
                "role": agent.role,
                "lifecycle": agent.lifecycle,
                "parent_agent_id": parent.agent_id,
            },
        )
        return agent

    def terminate_agent(self, caller_agent_id: str, target_agent_id: str) -> AgentRecord:
        if not self.can_manage_agent(caller_agent_id, target_agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        agent = self.require_agent(target_agent_id)
        if self.is_root_agent(target_agent_id):
            raise ValueError("cannot terminate root agent")
        lifecycle = derive_agent_lifecycle(agent)
        if agent.status != "terminated" or agent.run_status != "terminated":
            agent.status = "terminated"
            if not lifecycle["is_terminal"] or agent.run_status != "terminated":
                agent.run_status = "terminated"
            self.save_agent(agent)
        return agent

    def append_owned_workflow(self, agent_id: str, workflow_id: str) -> None:
        agent = self.require_agent(agent_id)
        if workflow_id not in agent.owned_workflow_ids:
            agent.owned_workflow_ids.append(workflow_id)
            self.save_agent(agent)

    def list_scope_grants(self, agent_id: str) -> list[ScopeGrant]:
        agent = self.require_agent(agent_id)
        grants: list[ScopeGrant] = []
        for item in list(agent.scope_grants or []):
            try:
                grants.append(ScopeGrant.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
        grants.sort(key=lambda item: (item.path, item.timestamp, item.granted_by))
        return grants

    def grant_scope(
        self,
        granting_agent_id: str,
        target_agent_id: str,
        path: str | Path,
        *,
        reason: str,
    ) -> ScopeGrant:
        from mr1.capability_policy import normalize_path

        granter = self.require_agent(granting_agent_id)
        if not self.is_root_agent(granting_agent_id) and granter.security_clearance < 1.0:
            raise AgentScopeError("access denied: insufficient security clearance for scope grant")
        normalized_path = str(normalize_path(path))
        if not (self.is_root_agent(granting_agent_id) or self.can_agent_access_path(granting_agent_id, normalized_path)):
            raise AgentScopeError("access denied: granting agent lacks access to requested scope")
        if (
            granting_agent_id == target_agent_id
            and not self.can_agent_access_path(target_agent_id, normalized_path)
        ):
            raise AgentScopeError("access denied: agent cannot self-grant outside current scope")
        target = self.require_agent(target_agent_id)
        target.scope_roots = sorted(set(list(target.scope_roots) + [normalized_path]))
        grants_by_path = {item.path: item for item in self.list_scope_grants(target_agent_id)}
        grants_by_path[normalized_path] = ScopeGrant(
            agent_id=target_agent_id,
            path=normalized_path,
            granted_by=granting_agent_id,
            reason=reason.strip() or "scope grant",
            timestamp=_now_iso(),
        )
        target.scope_grants = [
            item.to_dict()
            for item in sorted(grants_by_path.values(), key=lambda item: item.path)
        ]
        self.save_agent(target)
        self._event_log.emit(
            event_type="agent_scope_granted",
            actor_id=granting_agent_id,
            actor_type=granter.actor_category,
            target_id=target.agent_id,
            target_type="agent",
            status="granted",
            summary=f"scope granted to {target.agent_id}",
            record_path=str(self.agent_path(target.agent_id)),
            metadata={
                "path": normalized_path,
                "reason": grants_by_path[normalized_path].reason,
            },
        )
        return grants_by_path[normalized_path]

    def revoke_scope(
        self,
        granting_agent_id: str,
        target_agent_id: str,
        path: str | Path,
    ) -> str:
        from mr1.capability_policy import normalize_path

        granter = self.require_agent(granting_agent_id)
        if not self.is_root_agent(granting_agent_id) and granter.security_clearance < 1.0:
            raise AgentScopeError("access denied: insufficient security clearance for scope revoke")
        normalized_path = str(normalize_path(path))
        target = self.require_agent(target_agent_id)
        target.scope_roots = [
            item
            for item in list(target.scope_roots or [])
            if str(normalize_path(item)) != normalized_path
        ]
        target.scope_grants = [
            item.to_dict()
            for item in self.list_scope_grants(target_agent_id)
            if item.path != normalized_path
        ]
        self.save_agent(target)
        self._event_log.emit(
            event_type="agent_scope_revoked",
            actor_id=granting_agent_id,
            actor_type=granter.actor_category,
            target_id=target.agent_id,
            target_type="agent",
            status="revoked",
            summary=f"scope revoked from {target.agent_id}",
            record_path=str(self.agent_path(target.agent_id)),
            metadata={"path": normalized_path},
        )
        return normalized_path

    def assign_mission(
        self,
        caller_agent_id: str,
        target_agent_id: str,
        mission: str,
        *,
        parent_request: Optional[str] = None,
        assignment_packet: Optional[dict[str, Any]] = None,
    ) -> AgentRecord:
        if not self.can_manage_agent(caller_agent_id, target_agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        agent = self.require_agent(target_agent_id)
        if is_agent_terminal(agent):
            raise ValueError(f"agent terminated: {target_agent_id}")
        normalized_assignment_packet = (
            _clone_json_value(assignment_packet)
            if isinstance(assignment_packet, dict) else None
        )
        resolved_parent_request = parent_request
        if resolved_parent_request is None and normalized_assignment_packet is not None:
            packet_parent_request = normalized_assignment_packet.get("full_parent_request")
            if isinstance(packet_parent_request, str):
                resolved_parent_request = packet_parent_request
        agent.mission = mission.strip() if isinstance(mission, str) else None
        agent.run_status = "idle"
        agent.current_iteration = 0
        agent.last_step_at = None
        agent.last_action = None
        agent.parent_request = (
            resolved_parent_request
            if isinstance(resolved_parent_request, str) else None
        )
        agent.assignment_packet = normalized_assignment_packet
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

    def capability_audits_dir(self, agent_id: str) -> Path:
        path = self.logs_dir(agent_id) / "capability_audits"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def capability_audit_path(self, agent_id: str, audit_id: str) -> Path:
        return self.capability_audits_dir(agent_id) / f"{audit_id}.json"

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

    def write_step_prompt_artifact(
        self,
        agent_id: str,
        step_id: str,
        record: dict[str, Any],
        *,
        attempt: str = "primary",
    ) -> Path:
        with self._lock:
            path = self.step_prompt_artifact_path(agent_id, step_id, attempt=attempt)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
            tmp.replace(path)
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
        if owner is None or owner.mr_level == 1:
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
