from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from mr1.capabilities import CapabilityRegistry
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
    maybe_route_approval_request,
)
from mr1.clock import default_clock
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.scoped_agents import AgentStore
from mr1.workflow_models import Task, TaskStatus, Workflow
from mr1.workflow_store import WorkflowStore


class CapabilityGate:
    def __init__(
        self,
        *,
        store: WorkflowStore,
        capabilities: CapabilityRegistry,
        scoped_agents: AgentStore,
        message_store: MessageStore,
        policy_engine: PolicyEngine,
        approval_store: CapabilityApprovalStore,
        audit_writer: CapabilityAuditWriter,
        timeline: EventLog,
        workspace_root: Path,
        now_fn: Any,
        consent_store: Optional[Any] = None,
        clock: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._capabilities = capabilities
        self._scoped_agents = scoped_agents
        self._message_store = message_store
        self._policy_engine = policy_engine
        self._approval_store = approval_store
        self._audit_writer = audit_writer
        self._timeline = timeline
        self._workspace_root = workspace_root
        self._now = now_fn
        self._consent_store = consent_store
        self._clock = clock or default_clock()

    @staticmethod
    def objective_id_for_workflow(workflow: Workflow) -> Optional[str]:
        """
        The mission a workflow belongs to, stamped at submission.

        `submit_workflow(workflow_metadata={"objective_id": ...})` is the only
        way this gets set, which is what makes autonomous side effects
        attributable without new plumbing through CapabilityRequest.
        """
        value = (workflow.metadata or {}).get("objective_id")
        return value if isinstance(value, str) and value else None

    def _consent_grants_for(self, workflow: Workflow) -> list[Any]:
        objective_id = self.objective_id_for_workflow(workflow)
        if not objective_id or self._consent_store is None:
            return []
        try:
            return self._consent_store.list_active(objective_id=objective_id)
        except Exception:
            # A consent store that cannot be read authorizes nothing.
            return []

    def workflow_actor_type(self, workflow: Workflow) -> str:
        owner = self._scoped_agents.load_agent(workflow.owner_agent_id or "")
        if owner is None:
            return "root_orchestrator"
        return owner.actor_category

    def capability_name_for_task(self, task: Task) -> str:
        if task.task_kind == "tool":
            return task.tool_type or ""
        if task.task_kind == "watcher":
            return task.watcher_type or ""
        raise ValueError(f"task is not a policy-managed capability: {task.task_id}")

    def capability_args_for_task(self, task: Task) -> dict[str, Any]:
        if task.task_kind == "tool":
            return dict(task.tool_config)
        if task.task_kind == "watcher":
            return dict(task.watch_config)
        raise ValueError(f"task is not a policy-managed capability: {task.task_id}")

    def current_attempt_policy_audit_path(self, task: Task) -> Optional[Path]:
        if task.current_attempt <= 0 or task.current_attempt > len(task.attempts):
            return None
        attempt = task.attempts[task.current_attempt - 1]
        return Path(attempt.policy_audit_path) if attempt.policy_audit_path else None

    def policy_audit_path(self, workflow: Workflow, task: Task, attempt_id: int) -> Path:
        return self._store.task_attempt_dir(
            workflow.workflow_id,
            task.task_id,
            attempt_id,
        ) / "capability_audit.json"

    def write_policy_audit(
        self,
        audit_path: Path,
        *,
        capability_name: str,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        execution_result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Path:
        record = CapabilityAuditRecord(
            capability_name=capability_name,
            request=request.to_dict(),
            metadata=metadata.to_dict(),
            decision=dict(decision),
            execution_result=execution_result,
            error=error,
            timestamp=self._now(),
        )
        return self._audit_writer.write(audit_path, record)

    def finalize_policy_audit(
        self,
        audit_path: Path,
        *,
        execution_result: dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        try:
            with open(audit_path, "r", encoding="utf-8") as handle:
                record = CapabilityAuditRecord.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return
        record.execution_result = dict(execution_result)
        record.error = error
        self._audit_writer.write(audit_path, record)

    def append_policy_audit_index(
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
        self._scoped_agents.append_capability_call_log(
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

    def append_policy_audit_index_from_path(
        self,
        audit_path: Path,
        *,
        execution_status: str,
        error: Optional[str] = None,
        approval_request_id: Optional[str] = None,
    ) -> None:
        try:
            with open(audit_path, "r", encoding="utf-8") as handle:
                record = CapabilityAuditRecord.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return
        request = CapabilityRequest.from_dict(record.request)
        metadata = CapabilityMetadata.from_dict(record.metadata)
        self.append_policy_audit_index(
            actor_id=request.actor_id,
            audit_id=self.policy_audit_id(audit_path),
            audit_path=audit_path,
            request=request,
            metadata=metadata,
            decision=record.decision,
            execution_status=execution_status,
            error=error or record.error,
            approval_request_id=approval_request_id,
        )

    @staticmethod
    def policy_audit_id(audit_path: Path) -> str:
        digest = hashlib.sha256(str(audit_path).encode("utf-8")).hexdigest()[:16]
        return f"wf_cap_audit_{digest}"

    def evaluate_task_policy(
        self,
        workflow: Workflow,
        task: Task,
        attempt_id: int,
    ) -> tuple[CapabilityRequest, CapabilityMetadata, dict[str, Any], Path]:
        capability_name = self.capability_name_for_task(task)
        capability = self._capabilities.describe_capability(capability_name)
        metadata = CapabilityMetadata.from_dict(capability)
        request = CapabilityRequest(
            actor_id=workflow.owner_agent_id or self._scoped_agents.root_agent_id,
            actor_type=self.workflow_actor_type(workflow),
            actor_clearance=float(
                self._scoped_agents.require_agent(
                    workflow.owner_agent_id or self._scoped_agents.root_agent_id
                ).security_clearance
            ),
            invocation_mode="workflow",
            capability_name=capability_name,
            args=self.capability_args_for_task(task),
            scope=build_scope_context(
                actor_id=workflow.owner_agent_id or self._scoped_agents.root_agent_id,
                workspace_root=self._workspace_root,
                scoped_agent_store=self._scoped_agents,
                workflow_store=self._store,
                workflow_id=workflow.workflow_id,
                task_id=task.task_id,
            ),
            workflow_id=workflow.workflow_id,
            task_id=task.task_id,
            objective_id=self.objective_id_for_workflow(workflow),
        )
        decision = self._policy_engine.evaluate(
            request,
            metadata,
            config_schema=capability.get("config_schema", {}),
            approval_request=self._approval_store.approval_for_request(request, metadata),
            consent_grants=self._consent_grants_for(workflow),
            now=self._clock.now(),
        ).to_dict()
        audit_path = self.policy_audit_path(workflow, task, attempt_id)
        self.write_policy_audit(
            audit_path,
            capability_name=capability_name,
            request=request,
            metadata=metadata,
            decision=decision,
        )
        audit_id = self.policy_audit_id(audit_path)
        self._timeline.emit(
            event_type="capability_requested",
            actor_id=request.actor_id,
            actor_type=request.actor_type,
            target_id=request.capability_name,
            target_type="capability",
            status="requested",
            summary=f"capability requested: {request.capability_name}",
            workflow_id=workflow.workflow_id,
            task_id=task.task_id,
            audit_id=audit_id,
            record_path=str(audit_path),
            metadata={"mode": request.invocation_mode},
        )
        if decision["allowed"]:
            grant_id = decision.get("metadata", {}).get("consent_grant_id")
            self._timeline.emit(
                event_type="capability_allowed",
                actor_id=request.actor_id,
                actor_type=request.actor_type,
                target_id=request.capability_name,
                target_type="capability",
                status="allowed",
                summary=f"capability allowed: {request.capability_name}",
                workflow_id=workflow.workflow_id,
                task_id=task.task_id,
                audit_id=audit_id,
                record_path=str(audit_path),
                metadata={
                    "reason": decision["reason"],
                    "consent_grant_id": grant_id,
                    "objective_id": request.objective_id,
                },
            )
            if grant_id and self._consent_store is not None:
                # Count the unattended execution against the grant that allowed
                # it. This is the accountability metric: what MR1 did without
                # asking, and under whose authority.
                self._consent_store.record_use(
                    grant_id,
                    workflow_id=workflow.workflow_id,
                    task_id=task.task_id,
                    audit_id=audit_id,
                )
        return request, metadata, decision, audit_path

    def policy_block_result_payload(
        self,
        workflow: Workflow,
        task: Task,
        request: CapabilityRequest,
        decision: dict[str, Any],
        *,
        target_status: TaskStatus,
        approval_request_id: Optional[str] = None,
        audit_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        error_type = "approval_required" if target_status is TaskStatus.BLOCKED else "policy_block"
        payload = {
            "task_id": task.task_id,
            "workflow_id": workflow.workflow_id,
            "attempt_id": task.current_attempt,
            "status": target_status.value,
            "summary": decision["reason"],
            "text": "",
            "data": {
                "request": request.to_dict(),
                "decision": dict(decision),
                "approval_request_id": approval_request_id,
            },
            "metrics": {},
            "error": decision["reason"],
            "error_type": error_type,
            "failure_type": error_type,
            "retryable": False,
        }
        if audit_path is not None:
            payload["audit_record_path"] = str(audit_path)
        return payload

    def handle_policy_block_for_task(
        self,
        workflow: Workflow,
        task: Task,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        audit_path: Path,
    ) -> tuple[dict[str, Any], Optional[str], str]:
        approval_request_id = None
        approval = None
        if decision["status"] == "requires_approval":
            approval = build_approval_request(
                request,
                metadata,
                self._policy_engine.evaluate(
                    request,
                    metadata,
                    config_schema=self._capabilities.describe_capability(
                        request.capability_name
                    ).get("config_schema", {}),
                ),
            )
            approval_request_id = approval.approval_request_id
        self._timeline.emit(
            event_type="capability_blocked",
            actor_id=request.actor_id,
            actor_type=request.actor_type,
            target_id=request.capability_name,
            target_type="capability",
            status=decision["status"],
            summary=f"capability blocked: {request.capability_name}",
            workflow_id=workflow.workflow_id,
            task_id=task.task_id,
            approval_request_id=approval_request_id,
            audit_id=self.policy_audit_id(audit_path),
            record_path=str(audit_path),
            metadata={
                "reason": decision["reason"],
                "decision_status": decision["status"],
            },
        )
        routing_outcome = None
        if approval is not None:
            approval_request_id, _ = maybe_route_approval_request(
                approval,
                approval_store=self._approval_store,
                message_store=self._message_store,
                scoped_agent_store=self._scoped_agents,
            )
            routed = self._approval_store.require(approval_request_id)
            routing_outcome = (
                "needs_user_approval"
                if routed.designated_approver_id == self._scoped_agents.root_agent_id
                else "needs_escalation"
            )
        blocked_result = {
            "status": decision["status"],
            "reason": decision["reason"],
        }
        if approval_request_id is not None:
            blocked_result["approval_request_id"] = approval_request_id
        if routing_outcome is not None:
            blocked_result["routing_outcome"] = routing_outcome
        self.write_policy_audit(
            audit_path,
            capability_name=request.capability_name,
            request=request,
            metadata=metadata,
            decision=decision,
            execution_result=blocked_result,
        )
        self.append_policy_audit_index(
            actor_id=request.actor_id,
            audit_id=self.policy_audit_id(audit_path),
            audit_path=audit_path,
            request=request,
            metadata=metadata,
            decision=decision,
            execution_status=decision["status"],
            approval_request_id=approval_request_id,
        )
        return (
            self.policy_block_result_payload(
                workflow,
                task,
                request,
                decision,
                target_status=(
                    TaskStatus.BLOCKED
                    if decision["status"] == "requires_approval" else
                    TaskStatus.FAILED
                ),
                approval_request_id=approval_request_id,
                audit_path=audit_path,
            ),
            approval_request_id,
            decision["reason"],
        )
