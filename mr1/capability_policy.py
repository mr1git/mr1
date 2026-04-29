"""
Deterministic capability policy, scope, approval, and audit helpers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_store import WorkflowStore


POLICY_ID = "capability-policy-v1"
POLICY_VERSION = 1

_ALLOWED_DECISION_STATUSES = frozenset({"allowed", "denied", "requires_approval"})
_ALLOWED_FINAL_EFFECTS = frozenset({"executed", "blocked", "skipped"})
_ALLOWED_INVOCATION_MODES = frozenset({"direct", "workflow"})
_ALLOWED_ACTOR_TYPES = frozenset({"mr1", "mrn", "kazi", "workflow"})
_ALLOWED_KINDS = frozenset({"tool", "watcher", "agent"})
_ALLOWED_ACTOR_THRESHOLDS = {
    "mr1": {"direct": 0.50, "workflow": 1.00},
    "mrn": {"direct": 0.20, "workflow": 1.00},
    "kazi": {"direct": 0.00, "workflow": 1.00},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ScopeContext:
    allowed_roots: list[Path]
    workspace_root: Path

    def __post_init__(self) -> None:
        workspace_root = normalize_path(self.workspace_root)
        normalized_roots = sorted(
            {
                str(normalize_path(root))
                for root in self.allowed_roots
            }
        )
        object.__setattr__(self, "workspace_root", workspace_root)
        object.__setattr__(
            self,
            "allowed_roots",
            [Path(root) for root in normalized_roots],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_roots": [str(root) for root in self.allowed_roots],
            "workspace_root": str(self.workspace_root),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScopeContext":
        return cls(
            allowed_roots=[
                Path(item)
                for item in list(data.get("allowed_roots", []))
            ],
            workspace_root=Path(data["workspace_root"]),
        )

    def contains(self, path: Path) -> bool:
        normalized = normalize_path(path)
        return any(
            normalized == root or _path_within_root(normalized, root)
            for root in self.allowed_roots
        )


@dataclass(frozen=True)
class CapabilityMetadata:
    name: str
    kind: str
    effect: str
    risk_score: float
    direct_allowed: bool
    workflow_allowed: bool
    requires_scope: bool
    blocking: bool
    is_filesystem: bool
    is_network: bool
    is_execution: bool
    is_blocking: bool
    path_arg_fields: list[str]
    metadata_note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"invalid capability kind '{self.kind}' for {self.name}")
        if not 0.0 <= float(self.risk_score) <= 1.0:
            raise ValueError(f"invalid risk_score for capability '{self.name}'")
        if not isinstance(self.path_arg_fields, list):
            raise ValueError(f"path_arg_fields must be a list for capability '{self.name}'")
        if any(not isinstance(item, str) or not item for item in self.path_arg_fields):
            raise ValueError(f"path_arg_fields must contain non-empty strings for capability '{self.name}'")
        if self.is_filesystem and not self.path_arg_fields:
            raise ValueError(
                f"filesystem capability '{self.name}' must declare non-empty path_arg_fields"
            )
        if (
            not self.is_filesystem
            and self.path_arg_fields
            and self.metadata_note != "path-scoped-non-filesystem"
        ):
            raise ValueError(
                f"non-filesystem capability '{self.name}' cannot declare path_arg_fields without explicit justification"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "kind": self.kind,
            "effect": self.effect,
            "risk_score": self.risk_score,
            "direct_allowed": self.direct_allowed,
            "workflow_allowed": self.workflow_allowed,
            "requires_scope": self.requires_scope,
            "blocking": self.blocking,
            "is_filesystem": self.is_filesystem,
            "is_network": self.is_network,
            "is_execution": self.is_execution,
            "is_blocking": self.is_blocking,
            "path_arg_fields": list(self.path_arg_fields),
        }
        if self.metadata_note is not None:
            payload["metadata_note"] = self.metadata_note
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityMetadata":
        return cls(
            name=data["name"],
            kind=data["kind"],
            effect=data["effect"],
            risk_score=float(data["risk_score"]),
            direct_allowed=bool(data["direct_allowed"]),
            workflow_allowed=bool(data["workflow_allowed"]),
            requires_scope=bool(data["requires_scope"]),
            blocking=bool(data["blocking"]),
            is_filesystem=bool(data["is_filesystem"]),
            is_network=bool(data["is_network"]),
            is_execution=bool(data["is_execution"]),
            is_blocking=bool(data["is_blocking"]),
            path_arg_fields=list(data["path_arg_fields"]),
            metadata_note=data.get("metadata_note"),
        )


@dataclass(frozen=True)
class CapabilityRequest:
    actor_id: str
    actor_type: str
    invocation_mode: str
    capability_name: str
    args: dict[str, Any]
    scope: ScopeContext
    step_id: Optional[str] = None
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.actor_type not in _ALLOWED_ACTOR_TYPES:
            raise ValueError(f"invalid actor_type '{self.actor_type}'")
        if self.invocation_mode not in _ALLOWED_INVOCATION_MODES:
            raise ValueError(f"invalid invocation_mode '{self.invocation_mode}'")
        if not isinstance(self.args, dict):
            raise ValueError("capability args must be a JSON object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "invocation_mode": self.invocation_mode,
            "capability_name": self.capability_name,
            "args": dict(self.args),
            "scope": self.scope.to_dict(),
            "step_id": self.step_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityRequest":
        return cls(
            actor_id=data["actor_id"],
            actor_type=data["actor_type"],
            invocation_mode=data["invocation_mode"],
            capability_name=data["capability_name"],
            args=dict(data.get("args", {})),
            scope=ScopeContext.from_dict(dict(data["scope"])),
            step_id=data.get("step_id"),
            workflow_id=data.get("workflow_id"),
            task_id=data.get("task_id"),
        )


@dataclass(frozen=True)
class CapabilityDecision:
    allowed: bool
    status: str
    reason: str
    risk_score: float
    policy_id: str
    policy_version: int
    final_effect: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_DECISION_STATUSES:
            raise ValueError(f"invalid capability decision status '{self.status}'")
        if self.final_effect not in _ALLOWED_FINAL_EFFECTS:
            raise ValueError(f"invalid capability decision final_effect '{self.final_effect}'")
        if self.final_effect == "skipped":
            raise ValueError("Phase 15 must not emit final_effect='skipped'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "final_effect": self.final_effect,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CapabilityApprovalRequest:
    approval_request_id: str
    requesting_actor_id: str
    capability_name: str
    invocation_mode: str
    args: dict[str, Any]
    risk_score: float
    reason: str
    scope_summary: dict[str, Any]
    original_request: CapabilityRequest
    original_step_id: Optional[str] = None
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_request_id": self.approval_request_id,
            "requesting_actor_id": self.requesting_actor_id,
            "capability_name": self.capability_name,
            "invocation_mode": self.invocation_mode,
            "args": dict(self.args),
            "risk_score": self.risk_score,
            "reason": self.reason,
            "scope_summary": dict(self.scope_summary),
            "original_request": self.original_request.to_dict(),
            "original_step_id": self.original_step_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityApprovalRequest":
        return cls(
            approval_request_id=data["approval_request_id"],
            requesting_actor_id=data["requesting_actor_id"],
            capability_name=data["capability_name"],
            invocation_mode=data["invocation_mode"],
            args=dict(data.get("args", {})),
            risk_score=float(data["risk_score"]),
            reason=data["reason"],
            scope_summary=dict(data.get("scope_summary", {})),
            original_request=CapabilityRequest.from_dict(dict(data["original_request"])),
            original_step_id=data.get("original_step_id"),
            workflow_id=data.get("workflow_id"),
            task_id=data.get("task_id"),
            created_at=data.get("created_at", _now_iso()),
        )


@dataclass
class CapabilityAuditRecord:
    capability_name: str
    request: dict[str, Any]
    metadata: dict[str, Any]
    decision: dict[str, Any]
    execution_result: Optional[dict[str, Any]]
    error: Optional[str]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_name": self.capability_name,
            "request": dict(self.request),
            "metadata": dict(self.metadata),
            "decision": dict(self.decision),
            "execution_result": (
                dict(self.execution_result)
                if isinstance(self.execution_result, dict) else
                self.execution_result
            ),
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityAuditRecord":
        execution_result = data.get("execution_result")
        return cls(
            capability_name=data["capability_name"],
            request=dict(data["request"]),
            metadata=dict(data["metadata"]),
            decision=dict(data["decision"]),
            execution_result=(
                dict(execution_result)
                if isinstance(execution_result, dict) else
                execution_result
            ),
            error=data.get("error"),
            timestamp=data["timestamp"],
        )


_CAPABILITY_METADATA_RAW: dict[str, dict[str, Any]] = {
    "read_file": {
        "effect": "read",
        "risk_score": 0.10,
        "direct_allowed": True,
        "workflow_allowed": True,
        "requires_scope": True,
        "blocking": False,
        "is_filesystem": True,
        "is_network": False,
        "is_execution": False,
        "is_blocking": False,
        "path_arg_fields": ["path"],
    },
    "write_file": {
        "effect": "write",
        "risk_score": 0.65,
        "direct_allowed": False,
        "workflow_allowed": True,
        "requires_scope": True,
        "blocking": False,
        "is_filesystem": True,
        "is_network": False,
        "is_execution": False,
        "is_blocking": False,
        "path_arg_fields": ["path"],
    },
    "shell_command": {
        "effect": "execute",
        "risk_score": 1.00,
        "direct_allowed": False,
        "workflow_allowed": True,
        "requires_scope": True,
        "blocking": False,
        "is_filesystem": False,
        "is_network": False,
        "is_execution": True,
        "is_blocking": False,
        "path_arg_fields": ["cwd"],
        "metadata_note": "path-scoped-non-filesystem",
    },
    "file_exists": {
        "effect": "read",
        "risk_score": 0.00,
        "direct_allowed": True,
        "workflow_allowed": True,
        "requires_scope": True,
        "blocking": True,
        "is_filesystem": True,
        "is_network": False,
        "is_execution": False,
        "is_blocking": True,
        "path_arg_fields": ["path"],
    },
    "time_reached": {
        "effect": "wait",
        "risk_score": 0.00,
        "direct_allowed": True,
        "workflow_allowed": True,
        "requires_scope": False,
        "blocking": True,
        "is_filesystem": False,
        "is_network": False,
        "is_execution": False,
        "is_blocking": True,
        "path_arg_fields": [],
    },
    "manual_event": {
        "effect": "wait",
        "risk_score": 0.00,
        "direct_allowed": False,
        "workflow_allowed": True,
        "requires_scope": False,
        "blocking": True,
        "is_filesystem": False,
        "is_network": False,
        "is_execution": False,
        "is_blocking": True,
        "path_arg_fields": [],
    },
    "condition_script": {
        "effect": "execute",
        "risk_score": 0.75,
        "direct_allowed": True,
        "workflow_allowed": True,
        "requires_scope": True,
        "blocking": True,
        "is_filesystem": True,
        "is_network": False,
        "is_execution": True,
        "is_blocking": True,
        "path_arg_fields": ["path"],
    },
    "kazi": {
        "effect": "execute",
        "risk_score": 1.00,
        "direct_allowed": False,
        "workflow_allowed": False,
        "requires_scope": False,
        "blocking": False,
        "is_filesystem": False,
        "is_network": False,
        "is_execution": True,
        "is_blocking": False,
        "path_arg_fields": [],
    },
    "workflow_compiler": {
        "effect": "execute",
        "risk_score": 1.00,
        "direct_allowed": False,
        "workflow_allowed": False,
        "requires_scope": False,
        "blocking": False,
        "is_filesystem": False,
        "is_network": False,
        "is_execution": True,
        "is_blocking": False,
        "path_arg_fields": [],
    },
}


def metadata_for_capability(name: str, kind: str) -> CapabilityMetadata:
    payload = _CAPABILITY_METADATA_RAW.get(name)
    if payload is None:
        raise ValueError(f"missing capability metadata for '{name}'")
    return CapabilityMetadata(
        name=name,
        kind=kind,
        effect=payload["effect"],
        risk_score=float(payload["risk_score"]),
        direct_allowed=bool(payload["direct_allowed"]),
        workflow_allowed=bool(payload["workflow_allowed"]),
        requires_scope=bool(payload["requires_scope"]),
        blocking=bool(payload["blocking"]),
        is_filesystem=bool(payload["is_filesystem"]),
        is_network=bool(payload["is_network"]),
        is_execution=bool(payload["is_execution"]),
        is_blocking=bool(payload["is_blocking"]),
        path_arg_fields=list(payload["path_arg_fields"]),
        metadata_note=payload.get("metadata_note"),
    )


def build_scope_context(
    *,
    actor_id: str,
    workspace_root: Path,
    scoped_agent_store: PersistentAgentStore,
    workflow_store: Optional[WorkflowStore] = None,
    workflow_id: Optional[str] = None,
    task_id: Optional[str] = None,
    explicit_roots: Optional[list[str | Path]] = None,
) -> ScopeContext:
    normalized_workspace_root = normalize_path(workspace_root)
    allowed_roots: list[Path] = []
    if scoped_agent_store.is_root_agent(actor_id):
        allowed_roots.append(normalized_workspace_root)
    else:
        agent = scoped_agent_store.require_agent(actor_id)
        allowed_roots.extend(
            normalize_path(item)
            for item in list(getattr(agent, "scope_roots", []) or [])
        )
    if explicit_roots:
        allowed_roots.extend(normalize_path(item) for item in explicit_roots)
    if workflow_id and workflow_store is not None:
        allowed_roots.append(normalize_path(workflow_store.workflow_dir(workflow_id)))
        if task_id:
            allowed_roots.append(
                normalize_path(workflow_store.workflow_dir(workflow_id) / "tasks" / task_id)
            )
    return ScopeContext(
        allowed_roots=allowed_roots,
        workspace_root=normalized_workspace_root,
    )


def normalize_args_for_policy(
    args: dict[str, Any],
    metadata: CapabilityMetadata,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if key in metadata.path_arg_fields and isinstance(value, (str, Path)):
            normalized[key] = str(normalize_path(value))
        elif isinstance(value, dict):
            normalized[key] = {
                inner_key: value[inner_key]
                for inner_key in sorted(value)
            }
        elif isinstance(value, list):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return normalized


def approval_hash_input(
    request: CapabilityRequest,
    metadata: CapabilityMetadata,
) -> dict[str, Any]:
    return {
        "actor_id": request.actor_id,
        "capability_name": request.capability_name,
        "invocation_mode": request.invocation_mode,
        "normalized_args": normalize_args_for_policy(request.args, metadata),
        "scope_roots": sorted(str(root) for root in request.scope.allowed_roots),
        "workflow_id": request.workflow_id,
        "task_id": request.task_id,
        "step_id": request.step_id,
    }


def approval_request_id_for(
    request: CapabilityRequest,
    metadata: CapabilityMetadata,
) -> str:
    raw = json.dumps(
        approval_hash_input(request, metadata),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"cap_approval_{digest}"


class CapabilityApprovalStore:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def approval_path(self, approval_request_id: str) -> Path:
        return self._root / f"{approval_request_id}.json"

    def exists(self, approval_request_id: str) -> bool:
        return self.approval_path(approval_request_id).exists()

    def save(self, approval: CapabilityApprovalRequest) -> Path:
        path = self.approval_path(approval.approval_request_id)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(approval.to_dict(), handle, indent=2, sort_keys=True)
        tmp.replace(path)
        return path


class CapabilityAuditWriter:
    def write(self, path: Path, record: CapabilityAuditRecord) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
        tmp.replace(path)
        return path


class PolicyEngine:
    def __init__(self, *, policy_id: str = POLICY_ID, policy_version: int = POLICY_VERSION):
        self._policy_id = policy_id
        self._policy_version = policy_version

    def evaluate(
        self,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        *,
        config_schema: Optional[dict[str, Any]] = None,
    ) -> CapabilityDecision:
        schema = dict(config_schema or {})
        missing_field = self._missing_required_arg(request.args, schema)
        if missing_field is not None:
            return self._decision(
                allowed=False,
                status="denied",
                reason=f"missing_required_arg:{missing_field}",
                risk_score=metadata.risk_score,
            )
        if request.invocation_mode == "direct" and not metadata.direct_allowed:
            return self._decision(
                allowed=False,
                status="denied",
                reason="capability_not_allowed_in_direct_mode",
                risk_score=metadata.risk_score,
            )
        if request.invocation_mode == "workflow" and not metadata.workflow_allowed:
            return self._decision(
                allowed=False,
                status="denied",
                reason="capability_not_allowed_in_workflow_mode",
                risk_score=metadata.risk_score,
            )
        thresholds = _ALLOWED_ACTOR_THRESHOLDS.get(request.actor_type)
        if thresholds is None:
            return self._decision(
                allowed=False,
                status="denied",
                reason="unknown_actor_policy",
                risk_score=metadata.risk_score,
            )
        max_risk = thresholds[request.invocation_mode]
        if metadata.risk_score > max_risk:
            return self._decision(
                allowed=False,
                status="denied",
                reason=f"risk_exceeds_{request.invocation_mode}_threshold",
                risk_score=metadata.risk_score,
                metadata_payload={"threshold": max_risk},
            )
        scope_decision = self._check_scope(request, metadata)
        if scope_decision is not None:
            return scope_decision
        return self._decision(
            allowed=True,
            status="allowed",
            reason="allowed",
            risk_score=metadata.risk_score,
        )

    def _missing_required_arg(
        self,
        args: dict[str, Any],
        config_schema: dict[str, Any],
    ) -> Optional[str]:
        for field, payload in config_schema.items():
            if not isinstance(payload, dict):
                continue
            if not payload.get("required"):
                continue
            if field not in args or args.get(field) in (None, ""):
                return field
        return None

    def _check_scope(
        self,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
    ) -> Optional[CapabilityDecision]:
        if not metadata.requires_scope:
            return None
        if not metadata.path_arg_fields and not metadata.is_filesystem:
            return None
        if not metadata.path_arg_fields:
            return self._decision(
                allowed=False,
                status="requires_approval",
                reason="scope_validation_not_meaningful",
                risk_score=metadata.risk_score,
            )
        for field in metadata.path_arg_fields:
            raw_value = request.args.get(field)
            if raw_value in (None, ""):
                return self._decision(
                    allowed=False,
                    status="denied",
                    reason=f"missing_required_arg:{field}",
                    risk_score=metadata.risk_score,
                )
            if not isinstance(raw_value, (str, Path)):
                return self._decision(
                    allowed=False,
                    status="denied",
                    reason=f"invalid_path_arg:{field}",
                    risk_score=metadata.risk_score,
                )
            normalized = normalize_path(raw_value)
            if not request.scope.contains(normalized):
                return self._decision(
                    allowed=False,
                    status="requires_approval",
                    reason="outside_actor_scope",
                    risk_score=metadata.risk_score,
                    metadata_payload={field: str(normalized)},
                )
        return None

    def _decision(
        self,
        *,
        allowed: bool,
        status: str,
        reason: str,
        risk_score: float,
        metadata_payload: Optional[dict[str, Any]] = None,
    ) -> CapabilityDecision:
        return CapabilityDecision(
            allowed=allowed,
            status=status,
            reason=reason,
            risk_score=risk_score,
            policy_id=self._policy_id,
            policy_version=self._policy_version,
            final_effect="executed" if allowed else "blocked",
            metadata=dict(metadata_payload or {}),
        )


def build_approval_request(
    request: CapabilityRequest,
    metadata: CapabilityMetadata,
    decision: CapabilityDecision,
) -> CapabilityApprovalRequest:
    approval_request_id = approval_request_id_for(request, metadata)
    return CapabilityApprovalRequest(
        approval_request_id=approval_request_id,
        requesting_actor_id=request.actor_id,
        capability_name=request.capability_name,
        invocation_mode=request.invocation_mode,
        args=normalize_args_for_policy(request.args, metadata),
        risk_score=decision.risk_score,
        reason=decision.reason,
        scope_summary=request.scope.to_dict(),
        original_request=request,
        original_step_id=request.step_id,
        workflow_id=request.workflow_id,
        task_id=request.task_id,
    )


def maybe_route_approval_request(
    approval: CapabilityApprovalRequest,
    *,
    approval_store: CapabilityApprovalStore,
    message_store: MessageStore,
    scoped_agent_store: PersistentAgentStore,
) -> tuple[str, bool]:
    already_exists = approval_store.exists(approval.approval_request_id)
    approval_store.save(approval)
    if already_exists:
        return approval.approval_request_id, False
    root_agent_id = scoped_agent_store.root_agent_id
    body = json.dumps(approval.to_dict(), indent=2, sort_keys=True)
    message_store.create_message(
        from_agent_id=approval.requesting_actor_id,
        to_agent_id=root_agent_id,
        kind="request",
        subject=f"Capability approval requested: {approval.capability_name}",
        body=body,
        workflow_id=approval.workflow_id,
        task_id=approval.task_id,
    )
    return approval.approval_request_id, True
