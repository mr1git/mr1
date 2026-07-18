"""
Deterministic derived graph memory built from the unified event timeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.capability_policy import CapabilityAuditRecord, CapabilityRequest, normalize_path
from mr1.event_log import EventLog, SystemEvent
from mr1.workflow_models import Task, Workflow
from mr1.workflow_store import WorkflowStore


NODE_TYPES = frozenset({
    "Agent",
    "Workflow",
    "WorkflowTemplate",
    "Task",
    "Capability",
    "ApprovalRequest",
    "FailureMode",
    "Project",
    "File",
})

EDGE_TYPES = frozenset({
    "CREATED_AGENT",
    "RAN_WORKFLOW",
    "HAS_TEMPLATE",
    "CONTAINS_TASK",
    "USED_CAPABILITY",
    "BLOCKED_BY_CAPABILITY",
    "REQUESTED_APPROVAL",
    "APPROVED_BY",
    "DENIED_BY",
    "FAILED_WITH",
    "SUCCEEDED_WITH",
    "TOUCHED_FILE",
    "BELONGS_TO_PROJECT",
})

UNKNOWN_WORKFLOW_TEMPLATE_ID = "workflow_template:unknown"
UNKNOWN_WORKFLOW_TEMPLATE_NAME = "unknown"

_INTERNAL_MEMORY_DIR_NAMES = frozenset({
    "agents",
    "messages",
    "capability_approvals",
    "graph",
    "events",
    "workflows",
})


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _normalized_path_string(path: str | Path) -> str:
    return str(normalize_path(path))


def _edge_identity(source_id: str, edge_type: str, target_id: str) -> str:
    return _canonical_json({
        "edge_type": edge_type,
        "source_id": _normalized_text(source_id),
        "target_id": _normalized_text(target_id),
    })


def agent_node_id(agent_id: str) -> str:
    return f"agent:{agent_id.strip()}"


def workflow_node_id(workflow_id: str) -> str:
    return f"workflow:{workflow_id.strip()}"


def workflow_template_node_id(template_hash: str) -> str:
    value = template_hash.strip()
    if value.startswith("workflow_template:"):
        return value
    return f"workflow_template:{value}"


def task_node_id(workflow_id: str, task_id: str) -> str:
    return f"task:{workflow_id.strip()}:{task_id.strip()}"


def capability_node_id(capability_name: str) -> str:
    return f"capability:{_normalized_text(capability_name)}"


def approval_node_id(approval_request_id: str) -> str:
    return f"approval:{approval_request_id.strip()}"


def failure_node_id(normalized_failure_key: str) -> str:
    return f"failure:{normalized_failure_key.strip()}"


def project_node_id(normalized_project_key: str) -> str:
    return f"project:{normalized_project_key.strip()}"


def file_node_id(path: str | Path) -> str:
    normalized = _normalized_path_string(path)
    return f"file:{_hash_text(normalized)}"


def edge_id(source_id: str, edge_type: str, target_id: str) -> str:
    return f"edge:{_hash_text(_edge_identity(source_id, edge_type, target_id))}"


@dataclass
class MemoryNode:
    node_id: str
    node_type: str
    name: str
    created_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"invalid node_type '{self.node_type}'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "stats": dict(self.stats),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            name=data["name"],
            created_at=data.get("created_at"),
            last_seen_at=data.get("last_seen_at"),
            stats=dict(data.get("stats", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MemoryEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    first_seen_at: str
    last_seen_at: str
    count: int = 1
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"invalid edge_type '{self.edge_type}'")
        if self.count < 1:
            raise ValueError("edge count must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "count": self.count,
            "stats": dict(self.stats),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEdge":
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data["edge_type"],
            first_seen_at=data["first_seen_at"],
            last_seen_at=data["last_seen_at"],
            count=int(data.get("count", 1)),
            stats=dict(data.get("stats", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MemoryGraph:
    nodes: dict[str, MemoryNode] = field(default_factory=dict)
    edges: dict[str, MemoryEdge] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {
                node_id: self.nodes[node_id].to_dict()
                for node_id in sorted(self.nodes)
            },
            "edges": {
                item_id: self.edges[item_id].to_dict()
                for item_id in sorted(self.edges)
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryGraph":
        nodes = {
            node_id: MemoryNode.from_dict(payload)
            for node_id, payload in dict(data.get("nodes", {})).items()
        }
        edges = {
            item_id: MemoryEdge.from_dict(payload)
            for item_id, payload in dict(data.get("edges", {})).items()
        }
        return cls(nodes=nodes, edges=edges)


@dataclass
class IngestionResult:
    processed_events: int
    last_processed_event_index: int
    nodes_created: int
    nodes_updated: int
    edges_created: int
    edges_updated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_events": self.processed_events,
            "last_processed_event_index": self.last_processed_event_index,
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "edges_updated": self.edges_updated,
        }


class MemoryGraphStore:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def graph_path(self) -> Path:
        return self._root / "graph.json"

    @property
    def cursor_path(self) -> Path:
        return self._root / "cursor.json"

    def load_graph(self) -> MemoryGraph:
        if not self.graph_path.exists():
            return MemoryGraph()
        try:
            with open(self.graph_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("graph payload must be an object")
            return MemoryGraph.from_dict(payload)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt graph file: {self.graph_path}") from exc

    def save_graph(self, graph: MemoryGraph) -> None:
        self._write_json_file(self.graph_path, graph.to_dict())

    def load_cursor(self) -> int:
        if not self.cursor_path.exists():
            return 0
        try:
            with open(self.cursor_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("cursor payload must be an object")
            value = int(payload["last_processed_event_index"])
            if value < 0:
                raise ValueError("cursor must be >= 0")
            return value
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt cursor file: {self.cursor_path}") from exc

    def save_cursor(self, last_processed_event_index: int) -> None:
        self._write_json_file(
            self.cursor_path,
            {"last_processed_event_index": int(last_processed_event_index)},
        )

    def upsert_node(
        self,
        graph: MemoryGraph,
        *,
        node_id: str,
        node_type: str,
        name: str,
        timestamp: Optional[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        existing = graph.nodes.get(node_id)
        if existing is None:
            graph.nodes[node_id] = MemoryNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                created_at=timestamp,
                last_seen_at=timestamp,
                stats={},
                metadata=dict(metadata or {}),
            )
            return True
        existing.name = name or existing.name
        if timestamp and not existing.created_at:
            existing.created_at = timestamp
        if timestamp:
            existing.last_seen_at = timestamp
        if metadata:
            existing.metadata.update(dict(metadata))
        return False

    def upsert_edge(
        self,
        graph: MemoryGraph,
        *,
        source_id: str,
        target_id: str,
        edge_type: str,
        timestamp: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        item_id = edge_id(source_id, edge_type, target_id)
        existing = graph.edges.get(item_id)
        if existing is None:
            graph.edges[item_id] = MemoryEdge(
                edge_id=item_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                first_seen_at=timestamp,
                last_seen_at=timestamp,
                count=1,
                stats={},
                metadata=dict(metadata or {}),
            )
            return True
        existing.last_seen_at = timestamp
        existing.count += 1
        if metadata:
            existing.metadata.update(dict(metadata))
        return False

    def increment_node_stat(
        self,
        graph: MemoryGraph,
        node_id: str,
        stat_name: str,
        amount: int = 1,
    ) -> None:
        node = graph.nodes.get(node_id)
        if node is None:
            return
        current = node.stats.get(stat_name, 0)
        try:
            current_value = int(current)
        except (TypeError, ValueError):
            current_value = 0
        node.stats[stat_name] = current_value + amount

    def increment_edge_stat(
        self,
        graph: MemoryGraph,
        item_id: str,
        stat_name: str,
        amount: int = 1,
    ) -> None:
        edge = graph.edges.get(item_id)
        if edge is None:
            return
        current = edge.stats.get(stat_name, 0)
        try:
            current_value = int(current)
        except (TypeError, ValueError):
            current_value = 0
        edge.stats[stat_name] = current_value + amount

    def _write_json_file(self, target: Path, payload: Any) -> None:
        tmp = target.with_suffix(f"{target.suffix}.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp.replace(target)


def compute_workflow_template_id(workflow_record_or_spec: Any) -> str:
    canonical = _canonical_workflow_structure(workflow_record_or_spec)
    if canonical is None:
        return UNKNOWN_WORKFLOW_TEMPLATE_ID
    digest = _hash_text(_canonical_json(canonical))
    return workflow_template_node_id(digest)


def _canonical_workflow_structure(workflow_record_or_spec: Any) -> Optional[dict[str, Any]]:
    spec = _workflow_like_to_spec(workflow_record_or_spec)
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None

    normalized_tasks: list[dict[str, Any]] = []
    alias_by_label: dict[str, str] = {}
    task_by_label: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tasks, start=1):
        if not isinstance(raw, dict):
            return None
        label = raw.get("label")
        if not isinstance(label, str) or not label.strip():
            return None
        alias = f"t{index}"
        alias_by_label[label] = alias
        task_by_label[label] = raw

    for label, alias in alias_by_label.items():
        raw = task_by_label[label]
        task_kind = str(raw.get("task_kind", "agent"))
        item: dict[str, Any] = {
            "alias": alias,
            "task_kind": task_kind,
            "depends_on": sorted(
                alias_by_label.get(dep, dep)
                for dep in list(raw.get("depends_on") or [])
                if isinstance(dep, str)
            ),
            "dependency_policy": str(raw.get("dependency_policy", "all_succeeded")),
        }
        if task_kind == "agent":
            item["agent_type"] = str(raw.get("agent_type", "worker"))
        elif task_kind == "tool":
            item["tool_type"] = str(raw.get("tool_type") or "")
            item["shape_config"] = _structural_config_subset(dict(raw.get("tool_config", {})))
        elif task_kind == "watcher":
            item["watcher_type"] = str(raw.get("watcher_type") or "")
            item["shape_config"] = _structural_config_subset(dict(raw.get("watch_config", {})))
            if raw.get("condition") is not None:
                item["condition"] = _normalize_shape_value(raw.get("condition"))
        run_if = raw.get("run_if")
        if isinstance(run_if, dict):
            normalized_run_if = {
                key: _normalize_shape_value(value)
                for key, value in sorted(run_if.items())
                if key != "ref"
            }
            if isinstance(run_if.get("ref"), str):
                normalized_run_if["ref"] = _remap_ref(run_if["ref"], alias_by_label)
            item["run_if"] = normalized_run_if
        inputs = raw.get("inputs") or []
        if inputs:
            refs = []
            for input_item in inputs:
                if not isinstance(input_item, dict):
                    continue
                source = input_item.get("from")
                if isinstance(source, str):
                    refs.append(_remap_ref(source, alias_by_label))
            if refs:
                item["inputs"] = sorted(refs)
        if raw.get("timeout_s") is not None:
            item["timeout_s"] = raw.get("timeout_s")
        normalized_tasks.append(item)

    return {"tasks": normalized_tasks}


def _workflow_like_to_spec(workflow_record_or_spec: Any) -> dict[str, Any]:
    if isinstance(workflow_record_or_spec, Workflow):
        tasks = []
        for label, task_id_value in workflow_record_or_spec.label_to_task_id.items():
            task = workflow_record_or_spec.tasks.get(task_id_value)
            if task is None:
                continue
            tasks.append(_task_to_spec(task, workflow_record_or_spec))
        return {"title": workflow_record_or_spec.title, "tasks": tasks}
    if isinstance(workflow_record_or_spec, dict):
        tasks = workflow_record_or_spec.get("tasks")
        if isinstance(tasks, dict):
            ordered_tasks = []
            label_to_task_id = dict(workflow_record_or_spec.get("label_to_task_id", {}))
            workflow_id_value = str(workflow_record_or_spec.get("workflow_id") or "")
            for label, task_id_value in label_to_task_id.items():
                payload = tasks.get(task_id_value)
                if not isinstance(payload, dict):
                    continue
                task = Task.from_dict(payload)
                if not task.workflow_id and workflow_id_value:
                    task.workflow_id = workflow_id_value
                ordered_tasks.append(_task_to_spec(task, Workflow.from_dict({
                    "workflow_id": workflow_record_or_spec.get("workflow_id", "wf-unknown"),
                    "title": workflow_record_or_spec.get("title", ""),
                    "tasks": tasks,
                    "label_to_task_id": label_to_task_id,
                })))
            return {"title": workflow_record_or_spec.get("title", ""), "tasks": ordered_tasks}
        return {"title": workflow_record_or_spec.get("title", ""), "tasks": list(tasks or [])}
    return {}


def _task_to_spec(task: Task, workflow: Workflow) -> dict[str, Any]:
    item: dict[str, Any] = {
        "label": task.label,
        "title": task.title,
        "task_kind": task.task_kind,
    }
    depends_on = []
    for dep_id in task.depends_on:
        dep_label = _label_for_task_id(workflow, dep_id)
        depends_on.append(dep_label or dep_id)
    if depends_on:
        item["depends_on"] = depends_on
    if task.dependency_policy != "all_succeeded":
        item["dependency_policy"] = task.dependency_policy
    if task.run_if is not None:
        item["run_if"] = dict(task.run_if)
    if task.inputs:
        item["inputs"] = [input_item.to_dict() for input_item in task.inputs]
    if task.timeout_s is not None:
        item["timeout_s"] = task.timeout_s
    if task.task_kind == "agent":
        item["agent_type"] = task.agent_type or "worker"
    elif task.task_kind == "tool":
        item["tool_type"] = task.tool_type
        item["tool_config"] = dict(task.tool_config)
    elif task.task_kind == "watcher":
        item["watcher_type"] = task.watcher_type
        item["watch_config"] = dict(task.watch_config)
        if task.condition is not None:
            item["condition"] = dict(task.condition)
    return item


def _normalize_shape_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_shape_value(value[key])
            for key in sorted(value)
            if key not in {"prompt", "title", "path", "cwd", "content"}
        }
    if isinstance(value, list):
        return [_normalize_shape_value(item) for item in value]
    return value


def _structural_config_subset(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_shape_value(value)
        for key, value in sorted(config.items())
        if key not in {"path", "cwd", "content", "argv", "at", "event"}
    }


def _remap_ref(value: str, alias_by_label: dict[str, str]) -> str:
    if "." not in value:
        return alias_by_label.get(value, value)
    head, tail = value.split(".", 1)
    return f"{alias_by_label.get(head, head)}.{tail}"


def _label_for_task_id(workflow: Workflow, task_id_value: str) -> Optional[str]:
    for label, candidate in workflow.label_to_task_id.items():
        if candidate == task_id_value:
            return label
    return None


def update_graph_from_events(
    event_log: EventLog,
    graph_store: MemoryGraphStore,
) -> IngestionResult:
    graph = graph_store.load_graph()
    cursor = graph_store.load_cursor()
    workflow_store = WorkflowStore(root=graph_store.root.parent / "workflows")
    capability_registry = default_capability_registry()
    events = [
        event
        for event in event_log.list_events()
        if event.event_index > cursor
    ]

    nodes_created = 0
    nodes_updated = 0
    edges_created = 0
    edges_updated = 0
    last_processed_event_index = cursor

    for event in events:
        counts = _process_event(
            graph,
            event,
            graph_store=graph_store,
            workflow_store=workflow_store,
            capability_registry=capability_registry,
        )
        nodes_created += counts["nodes_created"]
        nodes_updated += counts["nodes_updated"]
        edges_created += counts["edges_created"]
        edges_updated += counts["edges_updated"]
        last_processed_event_index = event.event_index

    _recompute_scores(graph)
    graph_store.save_graph(graph)
    graph_store.save_cursor(last_processed_event_index)
    return IngestionResult(
        processed_events=len(events),
        last_processed_event_index=last_processed_event_index,
        nodes_created=nodes_created,
        nodes_updated=nodes_updated,
        edges_created=edges_created,
        edges_updated=edges_updated,
    )


def _process_event(
    graph: MemoryGraph,
    event: SystemEvent,
    *,
    graph_store: MemoryGraphStore,
    workflow_store: WorkflowStore,
    capability_registry: CapabilityRegistry,
) -> dict[str, int]:
    counts = {
        "nodes_created": 0,
        "nodes_updated": 0,
        "edges_created": 0,
        "edges_updated": 0,
    }

    def ensure_node(
        node_id_value: Optional[str],
        node_type: str,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        if not node_id_value:
            return None
        created = graph_store.upsert_node(
            graph,
            node_id=node_id_value,
            node_type=node_type,
            name=name,
            timestamp=event.timestamp,
            metadata=metadata,
        )
        counts["nodes_created" if created else "nodes_updated"] += 1
        return node_id_value

    def ensure_edge(
        source_id_value: Optional[str],
        edge_type: str,
        target_id_value: Optional[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        if not source_id_value or not target_id_value:
            return None
        created = graph_store.upsert_edge(
            graph,
            source_id=source_id_value,
            target_id=target_id_value,
            edge_type=edge_type,
            timestamp=event.timestamp,
            metadata=metadata,
        )
        counts["edges_created" if created else "edges_updated"] += 1
        return edge_id(source_id_value, edge_type, target_id_value)

    workflow = _load_workflow_for_event(event, workflow_store)

    if event.event_type == "agent_created":
        parent_id = ensure_node(
            agent_node_id(event.actor_id) if event.actor_id else None,
            "Agent",
            event.actor_id or "",
        )
        child_id = ensure_node(
            agent_node_id(event.target_id) if event.target_id else None,
            "Agent",
            event.metadata.get("title") or event.target_id or "",
            metadata={
                "agent_id": event.target_id,
                "parent_agent_id": event.metadata.get("parent_agent_id"),
                "mr_level": event.metadata.get("mr_level"),
                "title": event.metadata.get("title"),
            },
        )
        ensure_edge(parent_id, "CREATED_AGENT", child_id)

    elif event.event_type == "workflow_created":
        workflow_id_value = ensure_node(
            workflow_node_id(event.workflow_id) if event.workflow_id else None,
            "Workflow",
            event.metadata.get("title") or event.workflow_id or "",
            metadata={
                "workflow_id": event.workflow_id,
                "title": event.metadata.get("title"),
                "owner_agent_id": event.metadata.get("owner_agent_id"),
                "task_count": event.metadata.get("task_count"),
            },
        )
        actor_id_value = ensure_node(
            agent_node_id(event.actor_id) if event.actor_id else None,
            "Agent",
            event.actor_id or "",
        )
        ensure_edge(actor_id_value, "RAN_WORKFLOW", workflow_id_value)
        template_id_value = _ensure_workflow_template(
            graph,
            graph_store,
            event=event,
            workflow=workflow,
            counts=counts,
        )
        if workflow_id_value and template_id_value:
            ensure_edge(workflow_id_value, "HAS_TEMPLATE", template_id_value)
            graph.nodes[workflow_id_value].metadata["template_id"] = template_id_value

    elif event.event_type in {"workflow_task_started", "workflow_task_completed", "workflow_task_failed"}:
        workflow_id_value = ensure_node(
            workflow_node_id(event.workflow_id) if event.workflow_id else None,
            "Workflow",
            workflow.title if workflow is not None else (event.workflow_id or ""),
            metadata={"workflow_id": event.workflow_id},
        )
        task_id_value = _ensure_task_node(
            graph,
            graph_store,
            event=event,
            workflow=workflow,
            counts=counts,
        )
        ensure_edge(workflow_id_value, "CONTAINS_TASK", task_id_value)
        if task_id_value and event.event_type == "workflow_task_completed":
            graph_store.increment_node_stat(graph, task_id_value, "success_count")
        if task_id_value and event.event_type == "workflow_task_failed":
            graph_store.increment_node_stat(graph, task_id_value, "failure_count")
            failure_id_value = _ensure_failure_mode(
                graph,
                graph_store,
                event=event,
                counts=counts,
            )
            ensure_edge(task_id_value, "FAILED_WITH", failure_id_value)

    elif event.event_type == "workflow_completed":
        workflow_id_value = ensure_node(
            workflow_node_id(event.workflow_id) if event.workflow_id else None,
            "Workflow",
            workflow.title if workflow is not None else (event.workflow_id or ""),
            metadata={"workflow_id": event.workflow_id},
        )
        if workflow_id_value:
            graph_store.increment_node_stat(graph, workflow_id_value, "success_count")
            template_id_value = _workflow_template_for_workflow_node(graph, workflow_id_value)
            if template_id_value:
                ensure_node(template_id_value, "WorkflowTemplate", _template_name(template_id_value))
                graph_store.increment_node_stat(graph, template_id_value, "success_count")
                ensure_edge(workflow_id_value, "SUCCEEDED_WITH", template_id_value)

    elif event.event_type == "workflow_failed":
        workflow_id_value = ensure_node(
            workflow_node_id(event.workflow_id) if event.workflow_id else None,
            "Workflow",
            workflow.title if workflow is not None else (event.workflow_id or ""),
            metadata={"workflow_id": event.workflow_id},
        )
        if workflow_id_value:
            graph_store.increment_node_stat(graph, workflow_id_value, "failure_count")
            template_id_value = _workflow_template_for_workflow_node(graph, workflow_id_value)
            if template_id_value:
                ensure_node(template_id_value, "WorkflowTemplate", _template_name(template_id_value))
                graph_store.increment_node_stat(graph, template_id_value, "failure_count")
        failure_id_value = _ensure_failure_mode(
            graph,
            graph_store,
            event=event,
            counts=counts,
        )
        ensure_edge(workflow_id_value, "FAILED_WITH", failure_id_value)

    elif event.event_type == "capability_requested":
        capability_id_value = ensure_node(
            capability_node_id(event.target_id) if event.target_id else None,
            "Capability",
            event.target_id or "",
            metadata={"capability_name": event.target_id},
        )
        if capability_id_value:
            graph_store.increment_node_stat(graph, capability_id_value, "request_count")
        actor_id_value = ensure_node(
            agent_node_id(event.actor_id) if event.actor_id else None,
            "Agent",
            event.actor_id or "",
        )
        ensure_edge(actor_id_value, "USED_CAPABILITY", capability_id_value)
        _attach_touched_paths(
            graph,
            graph_store,
            event=event,
            capability_registry=capability_registry,
            counts=counts,
        )

    elif event.event_type == "capability_blocked":
        capability_id_value = ensure_node(
            capability_node_id(event.target_id) if event.target_id else None,
            "Capability",
            event.target_id or "",
            metadata={"capability_name": event.target_id},
        )
        if capability_id_value:
            graph_store.increment_node_stat(graph, capability_id_value, "blocked_count")
        actor_id_value = ensure_node(
            agent_node_id(event.actor_id) if event.actor_id else None,
            "Agent",
            event.actor_id or "",
        )
        if actor_id_value:
            graph_store.increment_node_stat(graph, actor_id_value, "blocked_count")
        ensure_edge(actor_id_value, "BLOCKED_BY_CAPABILITY", capability_id_value)
        _attach_touched_paths(
            graph,
            graph_store,
            event=event,
            capability_registry=capability_registry,
            counts=counts,
        )

    elif event.event_type == "capability_executed":
        capability_id_value = ensure_node(
            capability_node_id(event.target_id) if event.target_id else None,
            "Capability",
            event.target_id or "",
            metadata={"capability_name": event.target_id},
        )
        if capability_id_value:
            graph_store.increment_node_stat(graph, capability_id_value, "execution_count")
        _attach_touched_paths(
            graph,
            graph_store,
            event=event,
            capability_registry=capability_registry,
            counts=counts,
        )

    elif event.event_type == "capability_failed":
        capability_id_value = ensure_node(
            capability_node_id(event.target_id) if event.target_id else None,
            "Capability",
            event.target_id or "",
            metadata={"capability_name": event.target_id},
        )
        if capability_id_value:
            graph_store.increment_node_stat(graph, capability_id_value, "failure_count")
        failure_id_value = _ensure_failure_mode(
            graph,
            graph_store,
            event=event,
            counts=counts,
        )
        ensure_edge(capability_id_value, "FAILED_WITH", failure_id_value)
        _attach_touched_paths(
            graph,
            graph_store,
            event=event,
            capability_registry=capability_registry,
            counts=counts,
        )

    elif event.event_type == "approval_requested":
        approval_id_value = ensure_node(
            approval_node_id(event.approval_request_id) if event.approval_request_id else None,
            "ApprovalRequest",
            event.approval_request_id or "",
            metadata={
                "approval_request_id": event.approval_request_id,
                "capability_name": event.metadata.get("capability_name"),
                "designated_approver_id": event.metadata.get("designated_approver_id"),
                "requested_scope_path": event.metadata.get("requested_scope_path"),
                "status": "pending",
            },
        )
        actor_id_value = ensure_node(
            agent_node_id(event.actor_id) if event.actor_id else None,
            "Agent",
            event.actor_id or "",
        )
        if actor_id_value:
            graph_store.increment_node_stat(graph, actor_id_value, "approval_requested_count")
        ensure_edge(actor_id_value, "REQUESTED_APPROVAL", approval_id_value)
        _attach_path_touch(
            graph,
            graph_store,
            raw_path=event.metadata.get("requested_scope_path"),
            timestamp=event.timestamp,
            actor_id=event.actor_id,
            workflow_id=event.workflow_id,
            counts=counts,
        )

    elif event.event_type == "approval_approved":
        approval_id_value = ensure_node(
            approval_node_id(event.approval_request_id) if event.approval_request_id else None,
            "ApprovalRequest",
            event.approval_request_id or "",
            metadata={"status": "approved"},
        )
        actor_id_value = ensure_node(
            agent_node_id(event.actor_id) if event.actor_id else None,
            "Agent",
            event.actor_id or "",
        )
        if actor_id_value:
            graph_store.increment_node_stat(graph, actor_id_value, "approval_approved_count")
        ensure_edge(approval_id_value, "APPROVED_BY", actor_id_value)

    elif event.event_type == "approval_denied":
        approval_id_value = ensure_node(
            approval_node_id(event.approval_request_id) if event.approval_request_id else None,
            "ApprovalRequest",
            event.approval_request_id or "",
            metadata={"status": "denied"},
        )
        actor_id_value = ensure_node(
            agent_node_id(event.actor_id) if event.actor_id else None,
            "Agent",
            event.actor_id or "",
        )
        if actor_id_value:
            graph_store.increment_node_stat(graph, actor_id_value, "approval_denied_count")
        ensure_edge(approval_id_value, "DENIED_BY", actor_id_value)

    return counts


def _ensure_task_node(
    graph: MemoryGraph,
    graph_store: MemoryGraphStore,
    *,
    event: SystemEvent,
    workflow: Optional[Workflow],
    counts: dict[str, int],
) -> Optional[str]:
    if not event.workflow_id or not event.task_id:
        return None
    label = event.task_id
    metadata: dict[str, Any] = {
        "workflow_id": event.workflow_id,
        "task_id": event.task_id,
    }
    if workflow is not None:
        task = workflow.tasks.get(event.task_id)
        if task is not None:
            label = task.label or task.task_id
            metadata.update({
                "label": task.label,
                "title": task.title,
                "task_kind": task.task_kind,
                "dependency_policy": task.dependency_policy,
                "agent_type": task.agent_type,
                "tool_type": task.tool_type,
                "watcher_type": task.watcher_type,
            })
    else:
        metadata.update({
            "task_kind": event.metadata.get("task_kind"),
            "attempt_id": event.metadata.get("attempt_id"),
        })
    node_id_value = task_node_id(event.workflow_id, event.task_id)
    created = graph_store.upsert_node(
        graph,
        node_id=node_id_value,
        node_type="Task",
        name=label,
        timestamp=event.timestamp,
        metadata=metadata,
    )
    counts["nodes_created" if created else "nodes_updated"] += 1
    return node_id_value


def _ensure_workflow_template(
    graph: MemoryGraph,
    graph_store: MemoryGraphStore,
    *,
    event: SystemEvent,
    workflow: Optional[Workflow],
    counts: dict[str, int],
) -> str:
    template_id_value = UNKNOWN_WORKFLOW_TEMPLATE_ID
    canonical = None
    if workflow is not None:
        template_id_value = compute_workflow_template_id(workflow)
        canonical = _canonical_workflow_structure(workflow)
    created = graph_store.upsert_node(
        graph,
        node_id=template_id_value,
        node_type="WorkflowTemplate",
        name=_template_name(template_id_value),
        timestamp=event.timestamp,
        metadata={
            "task_count": len((canonical or {}).get("tasks", [])),
            "canonical_structure": canonical,
        } if canonical is not None else {},
    )
    counts["nodes_created" if created else "nodes_updated"] += 1
    return template_id_value


def _ensure_failure_mode(
    graph: MemoryGraph,
    graph_store: MemoryGraphStore,
    *,
    event: SystemEvent,
    counts: dict[str, int],
) -> Optional[str]:
    normalized_key, display_name, metadata = _failure_mode_from_event(event)
    if not normalized_key:
        return None
    node_id_value = failure_node_id(normalized_key)
    created = graph_store.upsert_node(
        graph,
        node_id=node_id_value,
        node_type="FailureMode",
        name=display_name,
        timestamp=event.timestamp,
        metadata=metadata,
    )
    counts["nodes_created" if created else "nodes_updated"] += 1
    graph_store.increment_node_stat(graph, node_id_value, "occurrence_count")
    return node_id_value


def _failure_mode_from_event(event: SystemEvent) -> tuple[Optional[str], str, dict[str, Any]]:
    metadata = dict(event.metadata)
    reason = None
    for key in ("reason", "error", "error_type", "failure_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            reason = value.strip()
            break
    if reason is None and isinstance(event.summary, str) and event.summary.strip():
        reason = event.summary.strip()
    status = _normalized_text(event.status)
    event_type = _normalized_text(event.event_type)
    error_type = _normalized_text(metadata.get("error_type") or metadata.get("failure_type") or "")
    reason_key = _normalized_text(reason or "")
    if not any([status, event_type, error_type, reason_key]):
        return None, "", {}
    display_name = reason or event.event_type
    normalized_key = _hash_text(_canonical_json({
        "error_type": error_type,
        "event_type": event_type,
        "reason": reason_key,
        "status": status,
    }))
    return normalized_key, display_name, {
        "event_type": event.event_type,
        "status": event.status,
        "error_type": metadata.get("error_type") or metadata.get("failure_type"),
        "reason": reason,
    }


def _load_workflow_for_event(event: SystemEvent, workflow_store: WorkflowStore) -> Optional[Workflow]:
    if event.record_path and str(event.record_path).endswith("workflow.json"):
        path = Path(event.record_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return Workflow.from_dict(json.load(handle))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
    if event.workflow_id:
        return workflow_store.load_workflow(event.workflow_id)
    return None


def _workflow_template_for_workflow_node(graph: MemoryGraph, workflow_id_value: str) -> Optional[str]:
    node = graph.nodes.get(workflow_id_value)
    if node is None:
        return None
    template_id_value = node.metadata.get("template_id")
    if isinstance(template_id_value, str) and template_id_value:
        return template_id_value
    return None


def _template_name(template_id_value: str) -> str:
    if template_id_value == UNKNOWN_WORKFLOW_TEMPLATE_ID:
        return UNKNOWN_WORKFLOW_TEMPLATE_NAME
    return template_id_value.split(":", 1)[1]


def _attach_touched_paths(
    graph: MemoryGraph,
    graph_store: MemoryGraphStore,
    *,
    event: SystemEvent,
    capability_registry: CapabilityRegistry,
    counts: dict[str, int],
) -> None:
    audit_record = _load_audit_record(event.record_path)
    if audit_record is None:
        return
    request_payload = audit_record.request
    try:
        request = CapabilityRequest.from_dict(request_payload)
    except (KeyError, TypeError, ValueError):
        return
    try:
        capability = capability_registry.describe_capability(request.capability_name)
    except ValueError:
        return
    path_fields = list(capability.get("path_arg_fields", []))
    for field_name in path_fields:
        _attach_path_touch(
            graph,
            graph_store,
            raw_path=request.args.get(field_name),
            timestamp=event.timestamp,
            actor_id=event.actor_id or request.actor_id,
            workflow_id=event.workflow_id or request.workflow_id,
            counts=counts,
            allowed_roots=request.scope.allowed_roots,
            workspace_root=request.scope.workspace_root,
        )


def _load_audit_record(record_path: Optional[str]) -> Optional[CapabilityAuditRecord]:
    if not record_path:
        return None
    path = Path(record_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return CapabilityAuditRecord.from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _attach_path_touch(
    graph: MemoryGraph,
    graph_store: MemoryGraphStore,
    *,
    raw_path: Any,
    timestamp: str,
    actor_id: Optional[str],
    workflow_id: Optional[str],
    counts: dict[str, int],
    allowed_roots: Optional[list[Path]] = None,
    workspace_root: Optional[Path] = None,
) -> None:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        return
    normalized = normalize_path(raw_path)
    if _is_internal_record_path(normalized, graph_store.root.parent):
        return
    file_id_value = file_node_id(normalized)
    file_created = graph_store.upsert_node(
        graph,
        node_id=file_id_value,
        node_type="File",
        name=str(normalized),
        timestamp=timestamp,
        metadata={"path": str(normalized)},
    )
    counts["nodes_created" if file_created else "nodes_updated"] += 1
    graph_store.increment_node_stat(graph, file_id_value, "touch_count")

    actor_node_id_value = agent_node_id(actor_id) if actor_id else None
    if actor_node_id_value:
        created = graph_store.upsert_node(
            graph,
            node_id=actor_node_id_value,
            node_type="Agent",
            name=actor_id or "",
            timestamp=timestamp,
            metadata=None,
        )
        counts["nodes_created" if created else "nodes_updated"] += 1
        created_edge = graph_store.upsert_edge(
            graph,
            source_id=actor_node_id_value,
            target_id=file_id_value,
            edge_type="TOUCHED_FILE",
            timestamp=timestamp,
            metadata=None,
        )
        counts["edges_created" if created_edge else "edges_updated"] += 1

    workflow_node_id_value = workflow_node_id(workflow_id) if workflow_id else None
    if workflow_node_id_value:
        created = graph_store.upsert_node(
            graph,
            node_id=workflow_node_id_value,
            node_type="Workflow",
            name=workflow_id or "",
            timestamp=timestamp,
            metadata={"workflow_id": workflow_id},
        )
        counts["nodes_created" if created else "nodes_updated"] += 1
        created_edge = graph_store.upsert_edge(
            graph,
            source_id=workflow_node_id_value,
            target_id=file_id_value,
            edge_type="TOUCHED_FILE",
            timestamp=timestamp,
            metadata=None,
        )
        counts["edges_created" if created_edge else "edges_updated"] += 1

    project_root = _derive_project_root(
        normalized,
        allowed_roots=allowed_roots or [],
        workspace_root=workspace_root,
    )
    if project_root is None:
        return
    project_key = _normalized_text(project_root.name or str(project_root))
    if not project_key:
        return
    project_id_value = project_node_id(project_key)
    created = graph_store.upsert_node(
        graph,
        node_id=project_id_value,
        node_type="Project",
        name=project_root.name or str(project_root),
        timestamp=timestamp,
        metadata={"root_path": str(project_root)},
    )
    counts["nodes_created" if created else "nodes_updated"] += 1
    graph_store.increment_node_stat(graph, project_id_value, "touch_count")
    created_edge = graph_store.upsert_edge(
        graph,
        source_id=file_id_value,
        target_id=project_id_value,
        edge_type="BELONGS_TO_PROJECT",
        timestamp=timestamp,
        metadata=None,
    )
    counts["edges_created" if created_edge else "edges_updated"] += 1


def _is_internal_record_path(path: Path, runtime_root: Path) -> bool:
    for name in _INTERNAL_MEMORY_DIR_NAMES:
        root = runtime_root / name
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _derive_project_root(
    path: Path,
    *,
    allowed_roots: list[Path],
    workspace_root: Optional[Path],
) -> Optional[Path]:
    for root in sorted((normalize_path(item) for item in allowed_roots), key=lambda item: len(str(item)), reverse=True):
        if path == root or _path_within(path, root):
            return root
    if workspace_root is not None:
        normalized_workspace = normalize_path(workspace_root)
        if path == normalized_workspace or _path_within(path, normalized_workspace):
            return normalized_workspace
    return path.parent if path.parent != path else None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _recompute_scores(graph: MemoryGraph) -> None:
    for node in graph.nodes.values():
        if node.node_type == "WorkflowTemplate":
            success_count = int(node.stats.get("success_count", 0))
            failure_count = int(node.stats.get("failure_count", 0))
            blocked_count = int(node.stats.get("blocked_count", 0))
            node.stats["credibility_score"] = success_count / (
                success_count + failure_count + blocked_count + 1
            )
        elif node.node_type == "Capability":
            execution_count = int(node.stats.get("execution_count", 0))
            blocked_count = int(node.stats.get("blocked_count", 0))
            failure_count = int(node.stats.get("failure_count", 0))
            node.stats["reliability_score"] = execution_count / (
                execution_count + blocked_count + failure_count + 1
            )


def top_workflow_templates(graph: MemoryGraph, limit: int = 10) -> list[dict[str, Any]]:
    items = [
        _node_summary(node)
        for node in graph.nodes.values()
        if node.node_type == "WorkflowTemplate"
    ]
    items.sort(
        key=lambda item: (
            -float(item["stats"].get("credibility_score", 0.0)),
            -int(item["stats"].get("success_count", 0)),
            item["node_id"],
        )
    )
    return items[:limit]


def capability_stats(graph: MemoryGraph, limit: Optional[int] = None) -> list[dict[str, Any]]:
    items = [
        _node_summary(node)
        for node in graph.nodes.values()
        if node.node_type == "Capability"
    ]
    items.sort(
        key=lambda item: (
            -int(item["stats"].get("request_count", 0)),
            item["node_id"],
        )
    )
    return items if limit is None else items[:limit]


def failure_modes(graph: MemoryGraph, limit: int = 10) -> list[dict[str, Any]]:
    items = [
        _node_summary(node)
        for node in graph.nodes.values()
        if node.node_type == "FailureMode"
    ]
    items.sort(
        key=lambda item: (
            -int(item["stats"].get("occurrence_count", 0)),
            item["node_id"],
        )
    )
    return items[:limit]


def agent_summary(graph: MemoryGraph, agent_id_value: str) -> dict[str, Any]:
    node = graph.nodes.get(agent_node_id(agent_id_value))
    if node is None:
        raise ValueError(f"agent not found: {agent_id_value}")
    return _summary_for_node(graph, node)


def workflow_template_summary(graph: MemoryGraph, template_id_value: str) -> dict[str, Any]:
    node_id_value = workflow_template_node_id(template_id_value)
    node = graph.nodes.get(node_id_value)
    if node is None:
        raise ValueError(f"workflow template not found: {template_id_value}")
    return _summary_for_node(graph, node)


def project_summary(graph: MemoryGraph, project_id_value: str) -> dict[str, Any]:
    node_id_value = project_id_value if project_id_value.startswith("project:") else project_node_id(project_id_value)
    node = graph.nodes.get(node_id_value)
    if node is None:
        raise ValueError(f"project not found: {project_id_value}")
    return _summary_for_node(graph, node)


def file_summary(graph: MemoryGraph, file_id_value: str) -> dict[str, Any]:
    node = graph.nodes.get(file_id_value)
    if node is None:
        raise ValueError(f"file not found: {file_id_value}")
    return _summary_for_node(graph, node)


def graph_stats(graph: MemoryGraph, *, last_processed_event_index: int) -> dict[str, Any]:
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for node in graph.nodes.values():
        node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
    for item in graph.edges.values():
        edge_counts[item.edge_type] = edge_counts.get(item.edge_type, 0) + 1
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_types": dict(sorted(node_counts.items())),
        "edge_types": dict(sorted(edge_counts.items())),
        "last_processed_event_index": last_processed_event_index,
    }


def show_node(graph: MemoryGraph, node_id_value: str) -> dict[str, Any]:
    node = graph.nodes.get(node_id_value)
    if node is None:
        raise ValueError(f"node not found: {node_id_value}")
    incident_edges = [
        edge.to_dict()
        for edge in graph.edges.values()
        if edge.source_id == node_id_value or edge.target_id == node_id_value
    ]
    incident_edges.sort(key=lambda item: (item["edge_type"], item["source_id"], item["target_id"]))
    return {
        "node": node.to_dict(),
        "incident_edges": incident_edges,
    }


def _node_summary(node: MemoryNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "name": node.name,
        "created_at": node.created_at,
        "last_seen_at": node.last_seen_at,
        "stats": dict(node.stats),
        "metadata": dict(node.metadata),
    }


def _summary_for_node(graph: MemoryGraph, node: MemoryNode) -> dict[str, Any]:
    related_edges = [
        edge.to_dict()
        for edge in graph.edges.values()
        if edge.source_id == node.node_id or edge.target_id == node.node_id
    ]
    related_edges.sort(key=lambda item: (item["edge_type"], item["source_id"], item["target_id"]))
    neighbors = []
    for edge in related_edges:
        other_id = edge["target_id"] if edge["source_id"] == node.node_id else edge["source_id"]
        other = graph.nodes.get(other_id)
        neighbors.append({
            "edge_id": edge["edge_id"],
            "edge_type": edge["edge_type"],
            "direction": "out" if edge["source_id"] == node.node_id else "in",
            "other_node_id": other_id,
            "other_node_type": other.node_type if other is not None else None,
            "other_name": other.name if other is not None else None,
            "count": edge["count"],
        })
    return {
        "node": _node_summary(node),
        "neighbors": neighbors,
    }
