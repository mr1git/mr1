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
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.memory_feedback import (
    evaluate_memory_feedback_due_for_runtime_root,
    run_memory_curate,
    run_memory_graph_update,
    update_insight_feedback,
)
from mr1.memory_queries import (
    memory_search,
    memory_graph_agent_summary,
    memory_graph_capabilities,
    memory_graph_failures,
    memory_graph_top_workflows,
    memory_insight_show,
    memory_insights_search,
)
from mr1.memory_retrieval import update_memory_retrieval
from mr1.scoped_agents import PersistentAgentStore
from mr1.memory_curator import InsightStore, evaluate_memory_curation_due_for_runtime_root
from mr1.memory_graph import MemoryGraphStore
from mr1.tools import _read_file_pure
from mr1.workflow_store import WorkflowStore
from mr1.watchers import (
    _evaluate_condition_script_pure,
    _evaluate_file_exists_pure,
    _evaluate_memory_curation_due_pure,
    _evaluate_memory_feedback_due_pure,
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


class CapabilityValidationError(ValueError):
    """Structured validation failure for direct capability calls."""


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
        self._event_log = EventLog(self._agents.root.parent / "events")

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
        self._event_log.emit(
            event_type="capability_requested",
            actor_id=caller_agent_id,
            actor_type=caller_type,
            target_id=name,
            target_type="capability",
            status="requested",
            summary=f"capability requested: {name}",
            workflow_id=workflow_id,
            task_id=task_id,
            step_id=step_id,
            audit_id=audit_id,
            record_path=str(audit_path),
            metadata={"mode": mode},
        )

        if not decision.allowed:
            approval_request_id = None
            approval = None
            if decision.status == "requires_approval":
                approval = build_approval_request(request, metadata, decision)
                approval_request_id = approval.approval_request_id
            self._event_log.emit(
                event_type="capability_blocked",
                actor_id=caller_agent_id,
                actor_type=caller_type,
                target_id=name,
                target_type="capability",
                status=decision.status,
                summary=f"capability blocked: {name}",
                workflow_id=workflow_id,
                task_id=task_id,
                step_id=step_id,
                approval_request_id=approval_request_id,
                audit_id=audit_id,
                record_path=str(audit_path),
                metadata={
                    "reason": decision.reason,
                    "decision_status": decision.status,
                },
            )
            if approval is not None:
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
        self._event_log.emit(
            event_type="capability_allowed",
            actor_id=caller_agent_id,
            actor_type=caller_type,
            target_id=name,
            target_type="capability",
            status="allowed",
            summary=f"capability allowed: {name}",
            workflow_id=workflow_id,
            task_id=task_id,
            step_id=step_id,
            audit_id=audit_id,
            record_path=str(audit_path),
            metadata={"reason": decision.reason},
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
        except CapabilityValidationError as exc:
            output = {
                "error_type": "validation_error",
                "message": str(exc),
            }
            error = str(exc)
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
        self._event_log.emit(
            event_type="capability_executed" if status == "succeeded" else "capability_failed",
            actor_id=caller_agent_id,
            actor_type=caller_type,
            target_id=name,
            target_type="capability",
            status=status,
            summary=(
                f"capability executed: {name}"
                if status == "succeeded" else
                f"capability failed: {name}"
            ),
            workflow_id=workflow_id,
            task_id=task_id,
            step_id=step_id,
            approval_request_id=approval_request_id if isinstance(approval_request_id, str) else None,
            audit_id=audit_id,
            record_path=str(audit_path),
            metadata={
                "duration_ms": duration_ms,
                "error": error,
            },
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

        if name == "memory_curation_due":
            runtime_root = config.get("runtime_root")
            if isinstance(runtime_root, str) and runtime_root.strip():
                result = _evaluate_memory_curation_due_pure(runtime_root)
            else:
                result = evaluate_memory_curation_due_for_runtime_root(self._workspace_root).to_dict()
                result["state"] = "satisfied" if result["due"] else "not_satisfied"
                result["message"] = (
                    f"memory curation due: {result['important_event_count']} important event(s)"
                    if result["due"] else
                    "memory curation not due"
                )
            output = {
                "due": bool(result["due"]),
                "latest_event_index": int(result["latest_event_index"]),
                "last_curated_event_index": int(result["last_curated_event_index"]),
                "important_event_count": int(result["important_event_count"]),
                "important_event_types": list(result["important_event_types"]),
                "suggested_event_window": list(result["suggested_event_window"]),
                "state": result["state"],
                "message": result["message"],
            }
            return output, None

        if name == "memory_feedback_due":
            runtime_root = config.get("runtime_root")
            if isinstance(runtime_root, str) and runtime_root.strip():
                result = _evaluate_memory_feedback_due_pure(runtime_root)
            else:
                result = evaluate_memory_feedback_due_for_runtime_root(self._workspace_root).to_dict()
                result["state"] = "satisfied" if result["due"] else "not_satisfied"
                result["message"] = (
                    f"memory feedback due: {result['relevant_event_count']} relevant event(s)"
                    if result["due"] else
                    "memory feedback not due"
                )
            output = {
                "due": bool(result["due"]),
                "latest_event_index": int(result["latest_event_index"]),
                "last_evaluated_event_index": int(result["last_evaluated_event_index"]),
                "relevant_event_count": int(result["relevant_event_count"]),
                "relevant_event_types": list(result["relevant_event_types"]),
                "suggested_event_window": list(result["suggested_event_window"]),
                "state": result["state"],
                "message": result["message"],
            }
            return output, None

        if name == "memory_insights_search":
            try:
                output = memory_insights_search(
                    self._workspace_root,
                    query=config.get("query"),
                    types=config.get("types"),
                    status=config.get("status"),
                    limit=config.get("limit"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_search":
            try:
                output = memory_search(
                    self._workspace_root,
                    query=config.get("query"),
                    types=config.get("types"),
                    limit=config.get("limit"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_insight_show":
            try:
                output = memory_insight_show(
                    self._workspace_root,
                    insight_id=config.get("insight_id"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_graph_top_workflows":
            try:
                output = memory_graph_top_workflows(
                    self._workspace_root,
                    limit=config.get("limit"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_graph_capabilities":
            try:
                output = memory_graph_capabilities(
                    self._workspace_root,
                    limit=config.get("limit"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_graph_failures":
            try:
                output = memory_graph_failures(
                    self._workspace_root,
                    limit=config.get("limit"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_graph_agent_summary":
            try:
                output = memory_graph_agent_summary(
                    self._workspace_root,
                    agent_id=config.get("agent_id"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_graph_update":
            try:
                output = run_memory_graph_update(
                    event_log=EventLog(self._workspace_root / "events"),
                    graph_store=MemoryGraphStore(self._workspace_root / "graph"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_curate":
            try:
                output = run_memory_curate(
                    event_log=EventLog(self._workspace_root / "events"),
                    graph_store=MemoryGraphStore(self._workspace_root / "graph"),
                    insight_store=InsightStore(self._workspace_root / "insights"),
                )
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_feedback_update":
            try:
                output = update_insight_feedback(
                    event_log=EventLog(self._workspace_root / "events"),
                    insight_store=InsightStore(self._workspace_root / "insights"),
                    workflow_store=WorkflowStore(root=self._workspace_root / "workflows"),
                ).to_dict()
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        if name == "memory_retrieval_update":
            try:
                output = update_memory_retrieval(self._workspace_root).to_dict()
            except ValueError as exc:
                raise CapabilityValidationError(str(exc)) from exc
            return output, None

        raise ValueError(f"capability not found: {name}")
