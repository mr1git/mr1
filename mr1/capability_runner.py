"""
Direct synchronous capability invocation with deterministic policy enforcement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.capability_policy import (
    CapabilityApprovalStore,
    CapabilityAuditRecord,
    CapabilityAuditWriter,
    CapabilityMetadata,
    CapabilityRequest,
    PolicyEngine,
    build_approval_request,
    build_scope_context,
    capability_audit_index_entry,
)
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore
from mr1.tools import _read_file_pure
from mr1.watchers import (
    _evaluate_condition_script_pure,
    _evaluate_file_exists_pure,
    _evaluate_time_reached_pure,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CapabilityResult:
    status: str  # succeeded | failed | denied | requires_approval
    output: dict[str, Any]
    error: Optional[str]
    duration_ms: int
    capability: str
    decision: dict[str, Any] = field(default_factory=dict)
    approval_request_id: Optional[str] = None
    audit_record_path: Optional[str] = None


class CapabilityRunner:
    def __init__(
        self,
        *,
        capability_registry: Optional[CapabilityRegistry] = None,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
        message_store: Optional[MessageStore] = None,
        workspace_root: Optional[Path] = None,
    ):
        self._registry = capability_registry or default_capability_registry()
        self._agents = scoped_agent_store or PersistentAgentStore()
        self._workspace_root = Path(workspace_root) if workspace_root else self._agents.root.parent
        self._message_store = message_store or MessageStore(
            root=self._agents.root.parent / "messages",
            scoped_agent_store=self._agents,
        )
        self._policy_engine = PolicyEngine()
        self._approval_store = CapabilityApprovalStore(
            self._agents.root.parent / "capability_approvals"
        )
        self._audit_writer = CapabilityAuditWriter()

    def run_capability(
        self,
        name: str,
        config: dict[str, Any],
        caller_agent_id: str,
        mode: str = "direct",
        *,
        step_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> CapabilityResult:
        try:
            meta = self._registry.describe_capability(name)
        except ValueError:
            raise ValueError(f"capability not found: {name}")
        caller_type = self._resolve_caller_type(caller_agent_id)
        if caller_type not in {"mr1", "mrn"}:
            raise ValueError("access denied")
        metadata = CapabilityMetadata.from_dict(meta)
        request = CapabilityRequest(
            actor_id=caller_agent_id,
            actor_type=caller_type,
            invocation_mode=mode,
            capability_name=name,
            args=dict(config),
            scope=build_scope_context(
                actor_id=caller_agent_id,
                workspace_root=self._workspace_root,
                scoped_agent_store=self._agents,
                workflow_id=workflow_id,
                task_id=task_id,
            ),
            step_id=step_id,
            workflow_id=workflow_id,
            task_id=task_id,
        )
        decision = self._policy_engine.evaluate(
            request,
            metadata,
            config_schema=meta.get("config_schema", {}),
            approved_request=self._approval_store.active_approval_for_request(request, metadata),
        )
        audit_id = self._new_audit_id(caller_agent_id)
        audit_path = self._agents.capability_audit_path(caller_agent_id, audit_id)
        record = CapabilityAuditRecord(
            capability_name=name,
            request=request.to_dict(),
            metadata=metadata.to_dict(),
            decision=decision.to_dict(),
            execution_result=None,
            error=None,
            timestamp=_now_iso(),
        )
        self._audit_writer.write(audit_path, record)

        if not decision.allowed:
            approval_request_id = None
            if decision.status == "requires_approval":
                approval = build_approval_request(request, metadata, decision)
                approval_request_id, _ = self._route_approval(approval)
            output = {
                "status": decision.status,
                "reason": decision.reason,
            }
            if approval_request_id is not None:
                output["approval_request_id"] = approval_request_id
            record.execution_result = dict(output)
            self._audit_writer.write(audit_path, record)
            self._append_audit_index(
                actor_id=caller_agent_id,
                audit_id=audit_id,
                audit_path=audit_path,
                request=request,
                metadata=metadata,
                decision=decision.to_dict(),
                execution_status=decision.status,
                approval_request_id=approval_request_id,
            )
            return CapabilityResult(
                status=decision.status,
                output=output,
                error=None,
                duration_ms=0,
                capability=name,
                decision=decision.to_dict(),
                approval_request_id=approval_request_id,
                audit_record_path=str(audit_path),
            )

        meta_timeout = 30
        config_timeout = config.get("timeout_s")
        if isinstance(config_timeout, int) and config_timeout > 0:
            meta_timeout = config_timeout
        started = time.monotonic()
        output: dict[str, Any] = {}
        error: Optional[str] = None
        status = "succeeded"
        try:
            output, error = self._dispatch(name, config, meta_timeout)
            if error is not None:
                status = "failed"
        except Exception as exc:
            error = str(exc)
            status = "failed"
        duration_ms = int((time.monotonic() - started) * 1000)
        record.execution_result = dict(output)
        record.error = error
        self._audit_writer.write(audit_path, record)
        approval_request_id = decision.metadata.get("approval_request_id")
        if status == "succeeded" and isinstance(approval_request_id, str) and approval_request_id:
            self._approval_store.mark_used(
                approval_request_id,
                audit_id=audit_id,
            )
        self._append_audit_index(
            actor_id=caller_agent_id,
            audit_id=audit_id,
            audit_path=audit_path,
            request=request,
            metadata=metadata,
            decision=decision.to_dict(),
            execution_status=status,
            error=error,
            approval_request_id=approval_request_id,
        )
        return CapabilityResult(
            status=status,
            output=output,
            error=error,
            duration_ms=duration_ms,
            capability=name,
            decision=decision.to_dict(),
            audit_record_path=str(audit_path),
        )

    def _resolve_caller_type(self, caller_agent_id: str) -> str:
        return "mr1" if self._agents.is_root_agent(caller_agent_id) else "mrn"

    def _route_approval(self, approval) -> tuple[str, bool]:
        from mr1.capability_policy import maybe_route_approval_request

        return maybe_route_approval_request(
            approval,
            approval_store=self._approval_store,
            message_store=self._message_store,
            scoped_agent_store=self._agents,
        )

    def _new_audit_id(self, caller_agent_id: str) -> str:
        base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        candidate = f"cap_audit_{base}"
        path = self._agents.capability_audit_path(caller_agent_id, candidate)
        suffix = 1
        while path.exists():
            candidate = f"cap_audit_{base}_{suffix}"
            path = self._agents.capability_audit_path(caller_agent_id, candidate)
            suffix += 1
        return candidate

    def _append_audit_index(
        self,
        *,
        actor_id: str,
        audit_id: str,
        audit_path: Path,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        execution_status: str,
        error: Optional[str] = None,
        approval_request_id: Optional[str] = None,
    ) -> None:
        self._agents.append_capability_call_log(
            actor_id,
            capability_audit_index_entry(
                audit_id=audit_id,
                audit_path=audit_path,
                request=request,
                metadata=metadata,
                decision=decision,
                execution_status=execution_status,
                error=error,
                approval_request_id=approval_request_id,
            ),
        )

    def _dispatch(
        self,
        name: str,
        config: dict[str, Any],
        timeout_s: int,
    ) -> tuple[dict[str, Any], Optional[str]]:
        if name == "read_file":
            path_str = config.get("path", "")
            max_bytes = config.get("max_bytes", 65536)
            r = _read_file_pure(path_str, max_bytes)
            output = {**r["data"], "text": r["text"]}
            return output, r["error"]

        if name == "file_exists":
            path_str = config.get("path", "")
            r = _evaluate_file_exists_pure(path_str)
            output = {
                "exists": r["metadata"].get("exists", False),
                "state": r["state"],
                "message": r["message"],
                **r["metadata"],
            }
            return output, None

        if name == "time_reached":
            at_str = config.get("at", "")
            now = datetime.now(timezone.utc)
            r = _evaluate_time_reached_pure(at_str, now)
            output = {
                "reached": r["state"] == "satisfied",
                "at": r["metadata"].get("at", at_str),
                "now": r["metadata"].get("now", now.isoformat()),
                "state": r["state"],
                "message": r["message"],
            }
            return output, None

        if name == "condition_script":
            path_str = config.get("path", "")
            r = _evaluate_condition_script_pure(path_str, timeout_s)
            state = r["state"]
            if state in ("timed_out", "failed"):
                return {**r["metadata"], "state": state, "message": r["message"]}, r["message"]
            output = {
                "satisfied": state == "satisfied",
                "state": state,
                "message": r["message"],
                **r["metadata"],
            }
            return output, None

        raise ValueError(f"capability not found: {name}")
