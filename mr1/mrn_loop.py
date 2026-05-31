"""
Bounded persistent MRn step execution.

This module implements a single deterministic reasoning/action cycle
for one persistent scoped MRn agent. It does not run autonomous loops,
recursive delegation, or messaging delivery.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from mr1.agents import AgentRuntimeError, parse_agent_json_envelope
from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.capability_runner import CapabilityResult, CapabilityRunner
from mr1.core import Dispatcher, PermissionDenied
from mr1.event_log import EventLog, bind_correlation_id, mrn_step_correlation_id
from mr1.kazi_runner import MockRunner
from mr1.messages import ALLOWED_MESSAGE_KINDS, MessageStore, normalize_message_kind
from mr1.orchestrator.loop.actions import (
    ACTION_DEFAULT_STATUS as _ACTION_DEFAULT_STATUS,
    ALLOWED_MRN_ACTIONS,
    ALLOWED_STATUSES as _ALLOWED_STATUSES,
    dispatch_action as _dispatch_action,
    parse_and_validate_action as _parse_and_validate_action_impl,
)
from mr1.orchestrator.loop.actions._text import (
    _MESSAGE_BODY_LIMIT,
    _MESSAGE_BODY_TRUNCATION_SUFFIX,
    _compact,
    _json_dumps,
    _truncate_message_body,
)
from mr1.orchestrator.identity import (
    _MRN_CONFIG_PATH,
    _PKG_ROOT,
    _now_iso as _orchestrator_now_iso,
)
from mr1.orchestrator.prompts import MRN_SYSTEM_PROMPT as _SYSTEM_PROMPT
from mr1.scoped_agents import (
    AgentScopeError,
    PersistentAgent,
    PersistentAgentStore,
    is_agent_terminal,
)
from mr1.scheduler import Scheduler, WorkflowSpecError
from mr1.tools import ToolRegistry, default_tool_registry
from mr1.watchers import WatcherRegistry, default_watcher_registry
from mr1.workflow_authoring import WorkflowAuthoringService
from mr1.workflow_compiler import WorkflowCompilerClient, WorkflowCompilerFailure
from mr1.workflow_models import Provenance, Workflow
from mr1.workflow_schema import WorkflowSchemaRegistry, default_workflow_schema_registry
from mr1.workflow_store import WorkflowStore


_DEFAULT_TIMEOUT_S = 300
_ALLOWED_MESSAGE_KIND_TEXT = " | ".join(f'"{kind}"' for kind in sorted(ALLOWED_MESSAGE_KINDS))
_ALLOWED_ACTIONS = ALLOWED_MRN_ACTIONS


ReasonerFn = Callable[[PersistentAgent, str, str], str]


@dataclass(frozen=True)
class MRnStepResult:
    agent_id: str
    iteration: int
    action: str
    status_before: str
    status_after: str
    reason: str
    message: str
    workflow_id: Optional[str] = None
    report_path: Optional[str] = None
    message_id: Optional[str] = None
    parent_request: Optional[str] = None
    error: Optional[str] = None
    created_workflow_id: Optional[str] = None
    created_workflow_status: Optional[str] = None
    created_parent_message_id: Optional[str] = None
    message_to_agent_id: Optional[str] = None
    confirmation_required: bool = False
    workflow_submitted: bool = False
    capability_result: Optional[dict[str, Any]] = None
    stored_as: Optional[str] = None
    prompt_artifact_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "iteration": self.iteration,
            "action": self.action,
            "status_before": self.status_before,
            "status_after": self.status_after,
            "reason": self.reason,
            "message": self.message,
            "workflow_id": self.workflow_id,
            "report_path": self.report_path,
            "message_id": self.message_id,
            "parent_request": self.parent_request,
            "error": self.error,
            "created_workflow_id": self.created_workflow_id,
            "created_workflow_status": self.created_workflow_status,
            "created_parent_message_id": self.created_parent_message_id,
            "message_to_agent_id": self.message_to_agent_id,
            "confirmation_required": self.confirmation_required,
            "workflow_submitted": self.workflow_submitted,
            "capability_result": self.capability_result,
            "stored_as": self.stored_as,
            "prompt_artifact_path": self.prompt_artifact_path,
        }


def _load_mrn_config(path: Path = _MRN_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _runtime_agent_type_for_level(level: int) -> str:
    return "mr1" if level <= 1 else f"mr{level}"


def _model_for_level(config: dict[str, Any], level: int) -> str:
    level_models = config.get("level_models", {})
    return level_models.get(level, config.get("default_model", "haiku"))


def run_mrn_step_agent(agent: PersistentAgent, system_prompt: str, prompt: str) -> str:
    config = _load_mrn_config()
    model = _model_for_level(config, agent.tree_level)
    allowed_tools = list(config.get("allowed_tools", []))
    cmd = ["claude", "-p", f"{system_prompt}\n\n{prompt}", "--model", model, "--output-format", "json"]
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])
    cli_flags = [token for token in cmd[1:] if token.startswith("-")]
    dispatcher = Dispatcher()
    agent_type = _runtime_agent_type_for_level(agent.tree_level)
    try:
        dispatcher.validate_full_spawn(agent_type, cli_flags, allowed_tools)
    except PermissionDenied as exc:
        raise ValueError(str(exc)) from exc
    timeout_s = int(config.get("timeout_s") or _DEFAULT_TIMEOUT_S)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"mrn step timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise ValueError(f"failed to run mrn step agent: {exc}") from exc
    raw_output = proc.stdout or ""
    if proc.returncode != 0:
        detail = (proc.stderr or raw_output).strip() or f"exit {proc.returncode}"
        raise ValueError(f"mrn step agent failed: {detail}")
    try:
        parsed = parse_agent_json_envelope(raw_output)
    except AgentRuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if parsed["is_error"]:
        raise ValueError(parsed["text"] or "mrn step agent returned an error")
    return parsed["text"]


class MRnStepRunner:
    def __init__(
        self,
        *,
        workflow_store: Optional[WorkflowStore] = None,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
        workflow_authoring_service: Optional[WorkflowAuthoringService] = None,
        workflow_compiler_client: Optional[WorkflowCompilerClient] = None,
        workflow_compiler: Optional[Callable[[str, str], str]] = None,
        workflow_authoring_backend: str = "local",
        capability_registry: Optional[CapabilityRegistry] = None,
        workflow_schema_registry: Optional[WorkflowSchemaRegistry] = None,
        watcher_registry: Optional[WatcherRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        message_store: Optional[MessageStore] = None,
        reasoner: Optional[ReasonerFn] = None,
        require_confirmation_for_workflows: bool = False,
        capability_runner: Optional[CapabilityRunner] = None,
    ):
        self._workflow_store = workflow_store or WorkflowStore()
        self._scoped_agents = scoped_agent_store or PersistentAgentStore(
            root=self._workflow_store.root.parent / "agents"
        )
        self._workflow_compiler = workflow_compiler
        self._workflow_authoring_backend = workflow_authoring_backend
        self._capability_registry = capability_registry or default_capability_registry()
        self._workflow_schema_registry = (
            workflow_schema_registry or default_workflow_schema_registry()
        )
        self._watcher_registry = watcher_registry or default_watcher_registry()
        self._tool_registry = tool_registry or default_tool_registry()
        self._message_store = message_store or MessageStore(
            root=self._workflow_store.root.parent / "messages",
            scoped_agent_store=self._scoped_agents,
        )
        self._workflow_compiler_client = (
            workflow_compiler_client or
            self._build_default_workflow_compiler_client()
        )
        self._reasoner = reasoner or run_mrn_step_agent
        self._require_confirmation_for_workflows = bool(require_confirmation_for_workflows)
        self._capability_runner = capability_runner or CapabilityRunner(
            capability_registry=self._capability_registry,
            scoped_agent_store=self._scoped_agents,
        )
        self._event_log = EventLog(self._workflow_store.root.parent / "events")
        self._workflow_authoring = (
            workflow_authoring_service or
            self._build_default_authoring_service()
        )

    def _build_default_workflow_compiler_client(self) -> WorkflowCompilerClient:
        return WorkflowCompilerClient(
            compiler=self._workflow_compiler,
            capability_registry=self._capability_registry,
            workflow_schema_registry=self._workflow_schema_registry,
            scoped_agent_store=self._scoped_agents,
            watcher_registry=self._watcher_registry,
            tool_registry=self._tool_registry,
        )

    def _build_default_authoring_service(self) -> WorkflowAuthoringService:
        scheduler = Scheduler(
            self._workflow_store,
            MockRunner(),
            auto_tick=False,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
        )
        return WorkflowAuthoringService(
            scheduler,
            self._workflow_store,
            compiler=self._workflow_compiler,
            authoring_backend=self._workflow_authoring_backend,
            workflow_compiler_client=self._workflow_compiler_client,
        )

    def step(
        self,
        agent_id: str,
        caller_agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> MRnStepResult:
        resolved_caller = caller_agent_id or self._scoped_agents.root_agent_id
        if not self._scoped_agents.can_manage_agent(resolved_caller, agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        agent = self._scoped_agents.require_agent(agent_id)
        if is_agent_terminal(agent):
            raise ValueError(f"agent terminated: {agent_id}")
        if not (agent.mission and agent.mission.strip()):
            raise ValueError(f"no mission assigned: {agent_id}")

        next_iteration = agent.current_iteration + 1
        step_id = f"{agent.agent_id}:{next_iteration}"
        correlation_id = mrn_step_correlation_id(agent.agent_id, next_iteration)
        self._event_log.emit(
            event_type="mrn_step_started",
            actor_id=agent.agent_id,
            actor_type=agent.agent_type,
            target_id=agent.agent_id,
            target_type="agent",
            status="started",
            summary="mrn step started",
            correlation_id=correlation_id,
            step_id=step_id,
            record_path=str(self._scoped_agents.step_log_path(agent.agent_id)),
            metadata={"iteration": next_iteration},
        )

        prompt = self._build_step_prompt(agent)
        prompt_artifact_path = self._write_prompt_artifact(
            agent,
            step_id=step_id,
            run_id=run_id,
            caller_agent_id=resolved_caller,
            system_prompt=_SYSTEM_PROMPT,
            prompt=prompt,
            correlation_id=correlation_id,
        )
        with bind_correlation_id(correlation_id):
            raw = self._reasoner(agent, _SYSTEM_PROMPT, prompt)
            try:
                action = self._parse_and_validate_action(raw)
            except ValueError as exc:
                corrected_raw = self._reasoner(
                    agent,
                    _SYSTEM_PROMPT,
                    self._build_correction_prompt(prompt, raw, str(exc)),
                )
                try:
                    action = self._parse_and_validate_action(corrected_raw)
                except ValueError as second_exc:
                    return self._persist_blocked_step(
                        agent,
                        reason=f"invalid mrn action: {second_exc}",
                        raw_action=corrected_raw,
                        prompt_artifact_path=str(prompt_artifact_path),
                    )

            step_call_count = [0]  # mutable counter, scoped to this step() invocation
            try:
                return self._execute_action(
                    agent,
                    action,
                    step_call_count,
                    prompt_artifact_path=str(prompt_artifact_path),
                )
            except (AgentScopeError, WorkflowCompilerFailure, WorkflowSpecError, RuntimeError, ValueError) as exc:
                return self._persist_blocked_step(
                    agent,
                    reason=str(exc),
                    raw_action=_json_dumps(action),
                    prompt_artifact_path=str(prompt_artifact_path),
                )

    def _build_step_prompt(self, agent: PersistentAgent) -> str:
        parts = [
            "Mission:",
            agent.mission or "",
        ]
        assignment_prompt = self._build_assignment_prompt(agent)
        if assignment_prompt:
            parts.extend([
                "Assignment:",
                assignment_prompt,
            ])
        parts.extend([
            "Scoped context:",
            self._build_scoped_context(agent),
        ])
        return "\n\n".join(parts)

    def _build_assignment_prompt(self, agent: PersistentAgent) -> str:
        assignment = (
            agent.assignment_packet
            if isinstance(agent.assignment_packet, dict) else None
        )
        if assignment is None:
            return ""
        scope = assignment.get("scope") if isinstance(assignment.get("scope"), dict) else {}
        relevant_context = (
            assignment.get("relevant_context")
            if isinstance(assignment.get("relevant_context"), dict) else {}
        )
        in_scope = scope.get("in_scope") if isinstance(scope.get("in_scope"), list) else []
        out_of_scope = scope.get("out_of_scope") if isinstance(scope.get("out_of_scope"), list) else []
        agent_ids = relevant_context.get("agents") if isinstance(relevant_context.get("agents"), list) else []
        message_ids = relevant_context.get("messages") if isinstance(relevant_context.get("messages"), list) else []
        workflow_ids = relevant_context.get("workflows") if isinstance(relevant_context.get("workflows"), list) else []
        lines = [
            f"child_title: {assignment.get('child_title') or agent.title}",
            f"clearance: {float(assignment.get('assigned_clearance', agent.security_clearance)):.2f}",
            f"mission: {_compact(assignment.get('mission'), limit=400)}",
            f"responsibility: {_compact(assignment.get('responsibility'), limit=400)}",
            f"delegated_subtask: {_compact(assignment.get('delegated_subtask'), limit=400)}",
            "full_parent_request:",
            assignment.get("full_parent_request") or "",
            "in_scope:",
        ]
        lines.extend(
            f"- {item}" for item in in_scope if isinstance(item, str) and item.strip()
        )
        if lines[-1] == "in_scope:":
            lines.append("- none")
        lines.append("out_of_scope:")
        lines.extend(
            f"- {item}" for item in out_of_scope if isinstance(item, str) and item.strip()
        )
        if lines[-1] == "out_of_scope:":
            lines.append("- none")
        lines.append(f"escalation_rules: {_compact(assignment.get('escalation_rules'), limit=400)}")
        lines.append(
            "relevant_context_ids: "
            f"agents={agent_ids or []} "
            f"messages={message_ids or []} "
            f"workflows={workflow_ids or []}"
        )
        return "\n".join(lines)

    def _build_scoped_context(self, agent: PersistentAgent) -> str:
        payload = {
            "agent": {
                "agent_id": agent.agent_id,
                "title": agent.title,
                "tree_level": agent.tree_level,
                "mission": agent.mission,
                "security_clearance": agent.security_clearance,
                "mode": agent.mode,
                "run_status": agent.run_status,
                "current_iteration": agent.current_iteration,
                "owned_workflow_ids": list(agent.owned_workflow_ids),
                "step_context": dict(agent.step_context) if agent.step_context else {},
            },
            "visible_workflows": self._workflow_summaries(agent),
            "recent_reports": self._recent_reports(agent),
            "unread_messages": self._inbox_messages(agent),
            "recent_sent_messages": self._outbox_messages(agent),
            "parent_messages": self._parent_messages(agent),
            "recent_events": self._recent_events(agent),
            "memory": _compact(self._scoped_agents.read_memory(agent.agent_id), limit=1000),
            "capabilities": self._capability_summaries(),
            "workflow_schema": self._workflow_schema_summary(),
        }
        return _json_dumps(payload)

    def _workflow_summaries(self, agent: PersistentAgent) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        for workflow in self._workflow_store.list_workflows():
            workflow = self._scoped_agents.normalize_workflow_ownership(workflow)
            if not self._scoped_agents.can_agent_access_workflow(agent.agent_id, workflow):
                continue
            visible.append({
                "workflow_id": workflow.workflow_id,
                "title": workflow.title,
                "status": workflow.status.value,
                "owner_agent_id": workflow.owner_agent_id,
                "task_count": len(workflow.tasks),
                "task_statuses": [
                    {
                        "label": task.label,
                        "status": task.status.value,
                        "summary": task.result_summary,
                    }
                    for task in workflow.tasks.values()
                ][:8],
            })
        visible.sort(key=lambda item: item["workflow_id"], reverse=True)
        return visible[:10]

    def _recent_reports(self, agent: PersistentAgent) -> list[dict[str, Any]]:
        reports = []
        for path in self._scoped_agents.list_reports(agent.agent_id)[:5]:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            reports.append({
                "name": path.name,
                "excerpt": _compact(text, limit=240),
            })
        return reports

    def _recent_events(self, agent: PersistentAgent) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for workflow in self._workflow_store.list_workflows():
            workflow = self._scoped_agents.normalize_workflow_ownership(workflow)
            if not self._scoped_agents.can_agent_access_workflow(agent.agent_id, workflow):
                continue
            for event in self._workflow_store.load_events(workflow.workflow_id, limit=5):
                events.append({
                    "timestamp": event.timestamp,
                    "workflow_id": event.workflow_id,
                    "task_id": event.task_id,
                    "event_type": event.event_type,
                    "message": _compact(event.message, limit=120),
                })
        events.sort(key=lambda item: item["timestamp"], reverse=True)
        return events[:10]

    def _message_summary(
        self,
        message,
        *,
        include_body: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "message_id": message.message_id,
            "from_agent_id": message.from_agent_id,
            "to_agent_id": message.to_agent_id,
            "kind": message.kind,
            "subject": message.subject,
            "status": message.status,
            "created_at": message.created_at,
        }
        if include_body:
            payload["body"] = _truncate_message_body(message.body)
        else:
            payload["body_excerpt"] = _compact(message.body, limit=160)
        return payload

    def _inbox_messages(self, agent: PersistentAgent) -> list[dict[str, Any]]:
        messages = [
            self._message_summary(message, include_body=True)
            for message in self._message_store.list_inbox(agent.agent_id)
            if message.status == "unread"
        ]
        return messages[:5]

    def _outbox_messages(self, agent: PersistentAgent) -> list[dict[str, Any]]:
        return [
            self._message_summary(message)
            for message in self._message_store.list_outbox(agent.agent_id)[:5]
        ]

    def _parent_messages(self, agent: PersistentAgent) -> list[dict[str, Any]]:
        parent_id = agent.parent_agent_id
        if not parent_id:
            return []
        summaries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for message in self._message_store.list_inbox(agent.agent_id)[:5]:
            if message.from_agent_id != parent_id:
                continue
            if message.message_id in seen:
                continue
            summaries.append(self._message_summary(message, include_body=True))
            seen.add(message.message_id)
        for message in self._message_store.list_outbox(agent.agent_id)[:5]:
            if message.to_agent_id != parent_id:
                continue
            if message.message_id in seen:
                continue
            summaries.append(self._message_summary(message, include_body=True))
            seen.add(message.message_id)
        summaries.sort(key=lambda item: (item["created_at"], item["message_id"]), reverse=True)
        return summaries[:5]

    def _capability_summaries(self) -> list[dict[str, str]]:
        summaries = []
        for item in self._capability_registry.describe_all():
            summaries.append({
                "name": item["name"],
                "type": item["type"],
                "description": _compact(item.get("description"), limit=120),
            })
        return summaries

    def _workflow_schema_summary(self) -> dict[str, Any]:
        schema = self._workflow_schema_registry.describe_all()
        return {
            "sections": sorted(schema),
            "task_kinds": sorted(schema.get("tasks", {}).keys()) if isinstance(schema.get("tasks"), dict) else [],
        }

    def _build_correction_prompt(self, prompt: str, invalid_output: str, error: str) -> str:
        return "\n\n".join([
            "Return corrected JSON only.",
            f"Validation error:\n{error}",
            "Original prompt:",
            prompt,
            "Invalid output:",
            invalid_output,
        ])

    def _write_prompt_artifact(
        self,
        agent: PersistentAgent,
        *,
        step_id: str,
        run_id: Optional[str],
        caller_agent_id: str,
        system_prompt: str,
        prompt: str,
        correlation_id: str,
    ) -> Path:
        payload = {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "agent_title": agent.title,
            "tree_level": agent.tree_level,
            "parent_agent_id": agent.parent_agent_id,
            "caller_agent_id": caller_agent_id,
            "run_id": run_id,
            "step_id": step_id,
            "correlation_id": correlation_id,
            "mission": agent.mission,
            "run_status_before": agent.run_status,
            "current_iteration_before": agent.current_iteration,
            "parent_request": agent.parent_request,
            "owned_workflow_ids": list(agent.owned_workflow_ids),
            "step_context": dict(agent.step_context),
            "assignment_packet": (
                dict(agent.assignment_packet)
                if isinstance(agent.assignment_packet, dict) else None
            ),
            "system_prompt": system_prompt,
            "prompt": prompt,
            "full_payload": f"{system_prompt}\n\n{prompt}",
        }
        return self._scoped_agents.write_step_prompt_artifact(
            agent.agent_id,
            step_id,
            payload,
        )

    def _parse_and_validate_action(self, raw: str) -> dict[str, Any]:
        return _parse_and_validate_action_impl(raw)

    def _execute_action(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        step_call_count: list[int],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        return _dispatch_action(
            self,
            agent,
            action,
            step_call_count,
            prompt_artifact_path=prompt_artifact_path,
        )

    def _persist_step(
        self,
        agent: PersistentAgent,
        *,
        action: dict[str, Any],
        status_after: str,
        message: str,
        workflow_id: Optional[str] = None,
        report_path: Optional[str] = None,
        message_id: Optional[str] = None,
        parent_request: Optional[str] = None,
        error: Optional[str] = None,
        workflow_summary: Optional[dict[str, Any]] = None,
        created_workflow_id: Optional[str] = None,
        created_workflow_status: Optional[str] = None,
        created_parent_message_id: Optional[str] = None,
        message_to_agent_id: Optional[str] = None,
        confirmation_required: bool = False,
        workflow_submitted: bool = False,
        capability_result: Optional[dict[str, Any]] = None,
        stored_as: Optional[str] = None,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        updated = self._scoped_agents.require_agent(agent.agent_id)
        status_before = updated.run_status
        iteration = updated.current_iteration + 1
        step_id = f"{updated.agent_id}:{iteration}"
        timestamp = updated.last_step_at = self._now_iso()
        updated.current_iteration = iteration
        updated.run_status = status_after
        if parent_request is not None:
            updated.parent_request = parent_request
        normalized_last_action = {
            "action": action["action"],
            "reason": action["reason"],
            "next_status": status_after,
        }
        if workflow_id is not None:
            normalized_last_action["workflow_id"] = workflow_id
        if report_path is not None:
            normalized_last_action["report_path"] = report_path
        if message_id is not None:
            normalized_last_action["message_id"] = message_id
        if created_workflow_id is not None:
            normalized_last_action["created_workflow_id"] = created_workflow_id
        if created_parent_message_id is not None:
            normalized_last_action["created_parent_message_id"] = created_parent_message_id
        if message_to_agent_id is not None:
            normalized_last_action["message_to_agent_id"] = message_to_agent_id
        if prompt_artifact_path is not None:
            normalized_last_action["prompt_artifact_path"] = prompt_artifact_path
        if confirmation_required:
            normalized_last_action["confirmation_required"] = True
        if workflow_submitted:
            normalized_last_action["workflow_submitted"] = True
        if parent_request is not None:
            normalized_last_action["parent_request"] = parent_request
        if error is not None:
            normalized_last_action["error"] = error
        updated.last_action = normalized_last_action
        self._scoped_agents.save_agent(updated)
        log_record: dict[str, Any] = {
            "timestamp": timestamp,
            "iteration": iteration,
            "action": action["action"],
            "reason": action["reason"],
            "status_before": status_before,
            "status_after": status_after,
            "workflow_id": workflow_id,
            "report_path": report_path,
            "message_id": message_id,
            "parent_request": parent_request,
            "error": error,
            "created_workflow_id": created_workflow_id,
            "created_workflow_status": created_workflow_status,
            "created_parent_message_id": created_parent_message_id,
            "message_to_agent_id": message_to_agent_id,
            "confirmation_required": confirmation_required,
            "workflow_submitted": workflow_submitted,
            "capability_result": capability_result,
            "stored_as": stored_as,
            "prompt_artifact_path": prompt_artifact_path,
        }
        if workflow_summary is not None:
            log_record["workflow_summary"] = workflow_summary
        self._scoped_agents.append_step_log(updated.agent_id, log_record)
        self._event_log.emit(
            event_type="mrn_step_completed",
            actor_id=updated.agent_id,
            actor_type=updated.agent_type,
            target_id=updated.agent_id,
            target_type="agent",
            status=status_after,
            summary=message,
            step_id=step_id,
            workflow_id=workflow_id,
            message_id=message_id,
            record_path=str(self._scoped_agents.step_log_path(updated.agent_id)),
            metadata={
                "iteration": iteration,
                "action": action["action"],
                "reason": action["reason"],
                "error": error,
                "prompt_artifact_path": prompt_artifact_path,
            },
        )
        return MRnStepResult(
            agent_id=updated.agent_id,
            iteration=iteration,
            action=action["action"],
            status_before=status_before,
            status_after=status_after,
            reason=action["reason"],
            message=message,
            workflow_id=workflow_id,
            report_path=report_path,
            message_id=message_id,
            parent_request=parent_request,
            error=error,
            created_workflow_id=created_workflow_id,
            created_workflow_status=created_workflow_status,
            created_parent_message_id=created_parent_message_id,
            message_to_agent_id=message_to_agent_id,
            confirmation_required=confirmation_required,
            workflow_submitted=workflow_submitted,
            capability_result=capability_result,
            stored_as=stored_as,
            prompt_artifact_path=prompt_artifact_path,
        )

    def _send_agent_message(
        self,
        agent: PersistentAgent,
        *,
        kind: str,
        subject: str,
        body: str,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        to_agent_id: Optional[str] = None,
    ):
        resolved_to_agent_id = to_agent_id or agent.parent_agent_id
        if not resolved_to_agent_id or not self._message_store.can_agent_send_message(
            agent.agent_id,
            resolved_to_agent_id,
        ):
            raise AgentScopeError("access denied: recipient not in agent scope")
        return self._message_store.create_message(
            from_agent_id=agent.agent_id,
            to_agent_id=resolved_to_agent_id,
            kind=kind,
            subject=subject,
            body=body,
            workflow_id=workflow_id,
            task_id=task_id,
        )

    def _persist_blocked_step(
        self,
        agent: PersistentAgent,
        *,
        reason: str,
        raw_action: str,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        action = {
            "action": "invalid",
            "reason": reason,
            "next_status": "blocked",
        }
        return self._persist_step(
            agent,
            action=action,
            status_after="blocked",
            message="step blocked",
            error=reason,
            workflow_summary={"raw_action": _compact(raw_action, limit=400)},
            prompt_artifact_path=prompt_artifact_path,
        )

    def _emit_mrn_reported(self, agent: PersistentAgent, result: MRnStepResult) -> None:
        if not result.report_path:
            return
        self._event_log.emit(
            event_type="mrn_reported",
            actor_id=agent.agent_id,
            actor_type=agent.agent_type,
            target_id=agent.agent_id,
            target_type="agent",
            status="reported",
            summary="mrn report emitted",
            step_id=f"{agent.agent_id}:{result.iteration}",
            workflow_id=result.workflow_id,
            message_id=result.message_id,
            record_path=result.report_path,
            metadata={"report_path": result.report_path},
        )

    @staticmethod
    def _now_iso() -> str:
        return _orchestrator_now_iso()
