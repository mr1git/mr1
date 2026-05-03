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
from mr1.messages import MessageStore
from mr1.scoped_agents import AgentScopeError, PersistentAgent, PersistentAgentStore
from mr1.scheduler import Scheduler, WorkflowSpecError
from mr1.tools import ToolRegistry, default_tool_registry
from mr1.watchers import WatcherRegistry, default_watcher_registry
from mr1.workflow_authoring import WorkflowAuthoringService
from mr1.workflow_compiler import WorkflowCompilerClient, WorkflowCompilerFailure
from mr1.workflow_models import Provenance, Workflow
from mr1.workflow_schema import WorkflowSchemaRegistry, default_workflow_schema_registry
from mr1.workflow_store import WorkflowStore


_PKG_ROOT = Path(__file__).resolve().parent
_MRN_CONFIG_PATH = _PKG_ROOT / "agents" / "mrn.yml"
_DEFAULT_TIMEOUT_S = 300
_MESSAGE_BODY_LIMIT = 4096
_MESSAGE_BODY_TRUNCATION_SUFFIX = "...[truncated, use message_id for full]"
ALLOWED_MRN_ACTIONS = frozenset({
    "create_workflow",
    "inspect_workflow",
    "write_report",
    "send_message",
    "ask_parent",
    "idle",
    "call_capability",
})
_ALLOWED_ACTIONS = ALLOWED_MRN_ACTIONS
_ALLOWED_STATUSES = frozenset({
    "idle",
    "working",
    "waiting",
    "reporting",
    "blocked",
    "terminated",
})
_ACTION_DEFAULT_STATUS = {
    "create_workflow": "working",
    "inspect_workflow": "working",
    "write_report": "reporting",
    "send_message": "waiting",
    "ask_parent": "waiting",
    "idle": "idle",
    "call_capability": "working",
}

_SYSTEM_PROMPT = """\
You are MRn, a persistent scoped orchestrator inside MR1.

You must return JSON only with this exact shape:
{
  "action": "create_workflow" | "inspect_workflow" | "write_report" | "send_message" | "ask_parent" | "idle" | "call_capability",
  "reason": "short reason",
  "workflow_request": "optional natural-language workflow request",
  "workflow_context": "optional extra context",
  "workflow_id": "optional existing workflow id",
  "report": "optional markdown report",
  "message_kind": "optional message kind",
  "message_subject": "optional message subject",
  "message_body": "optional message body",
  "to_agent_id": "optional message recipient",
  "parent_request": "optional question/request for parent",
  "capability": "optional direct-callable capability name for call_capability",
  "config": {},
  "store_as": "optional step_context key to store call_capability result",
  "next_status": "idle" | "working" | "waiting" | "reporting" | "blocked"
}

Rules:
- Return exactly one JSON object and nothing else.
- Do not use markdown fences.
- Choose exactly one action.
- Stay inside your scoped workflows and reports.
- You cannot access workflows outside your scope.
- You may request workflow creation, but runtime validation decides whether submission is allowed.
- If you do not have enough information, use "ask_parent" or "idle".
- For call_capability, set "capability" to a direct-callable capability name; optionally set "config" and "store_as".
"""

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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _compact(text: Any, *, limit: int = 240) -> str:
    if not isinstance(text, str):
        return "-"
    normalized = " ".join(text.split())
    if not normalized:
        return "-"
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _truncate_message_body(text: str, *, limit: int = _MESSAGE_BODY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_MESSAGE_BODY_TRUNCATION_SUFFIX))
    return text[:keep] + _MESSAGE_BODY_TRUNCATION_SUFFIX


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload:
        raise ValueError("empty output")
    if payload.startswith("```"):
        parts = payload.split("```")
        for part in parts:
            part = part.strip()
            if not part or part == "json":
                continue
            payload = part.removeprefix("json").strip()
            break
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    return data


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
        if agent.status == "terminated":
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
        return "\n\n".join([
            "Mission:",
            agent.mission or "",
            "Scoped context:",
            self._build_scoped_context(agent),
        ])

    def _build_scoped_context(self, agent: PersistentAgent) -> str:
        payload = {
            "agent": {
                "agent_id": agent.agent_id,
                "title": agent.title,
                "tree_level": agent.tree_level,
                "mission": agent.mission,
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
        action = _extract_json_object(raw)
        action_name = action.get("action")
        if action_name not in _ALLOWED_ACTIONS:
            raise ValueError("field 'action' must be one of: create_workflow, inspect_workflow, write_report, send_message, ask_parent, idle, call_capability")
        reason = action.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("field 'reason' must be a non-empty string")
        next_status = action.get("next_status", _ACTION_DEFAULT_STATUS[action_name])
        if next_status not in _ALLOWED_STATUSES:
            raise ValueError("field 'next_status' must be one of: idle, working, waiting, reporting, blocked, terminated")

        normalized = {
            "action": action_name,
            "reason": reason.strip(),
            "workflow_request": action.get("workflow_request"),
            "workflow_context": action.get("workflow_context"),
            "workflow_id": action.get("workflow_id"),
            "report": action.get("report"),
            "message_kind": action.get("message_kind"),
            "message_subject": action.get("message_subject"),
            "message_body": action.get("message_body"),
            "to_agent_id": action.get("to_agent_id"),
            "parent_request": action.get("parent_request"),
            "capability": action.get("capability"),
            "config": action.get("config"),
            "store_as": action.get("store_as"),
            "next_status": next_status,
        }

        if action_name == "create_workflow":
            if not isinstance(normalized["workflow_request"], str) or not normalized["workflow_request"].strip():
                raise ValueError("create_workflow requires workflow_request")
        elif action_name == "inspect_workflow":
            if not isinstance(normalized["workflow_id"], str) or not normalized["workflow_id"].strip():
                raise ValueError("inspect_workflow requires workflow_id")
        elif action_name == "write_report":
            if not isinstance(normalized["report"], str) or not normalized["report"].strip():
                raise ValueError("write_report requires report")
        elif action_name == "send_message":
            if not isinstance(normalized["message_kind"], str) or not normalized["message_kind"].strip():
                raise ValueError("send_message requires message_kind")
            if not isinstance(normalized["message_subject"], str) or not normalized["message_subject"].strip():
                raise ValueError("send_message requires message_subject")
            if not isinstance(normalized["message_body"], str) or not normalized["message_body"].strip():
                raise ValueError("send_message requires message_body")
        elif action_name == "ask_parent":
            if not isinstance(normalized["parent_request"], str) or not normalized["parent_request"].strip():
                raise ValueError("ask_parent requires parent_request")
            normalized["next_status"] = "waiting"
        elif action_name == "call_capability":
            if not isinstance(normalized["capability"], str) or not normalized["capability"].strip():
                raise ValueError("call_capability requires capability")
            if normalized["config"] is None:
                normalized["config"] = {}
            elif not isinstance(normalized["config"], dict):
                raise ValueError("call_capability config must be a JSON object")
            if normalized["store_as"] is not None:
                if not isinstance(normalized["store_as"], str) or not normalized["store_as"].strip():
                    raise ValueError("call_capability store_as must be a non-empty string when present")
        elif action_name == "idle":
            extra_fields = (
                normalized["workflow_request"],
                normalized["workflow_context"],
                normalized["workflow_id"],
                normalized["report"],
                normalized["message_kind"],
                normalized["message_subject"],
                normalized["message_body"],
                normalized["to_agent_id"],
                normalized["parent_request"],
            )
            if any(value not in (None, "") for value in extra_fields):
                raise ValueError("idle requires no extra fields")
            normalized["next_status"] = "idle"
        return normalized

    def _execute_action(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        step_call_count: list[int],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        if action["action"] == "call_capability":
            return self._execute_call_capability(
                agent,
                action,
                step_call_count,
                prompt_artifact_path=prompt_artifact_path,
            )
        if action["action"] == "create_workflow":
            return self._execute_create_workflow(
                agent,
                action,
                prompt_artifact_path=prompt_artifact_path,
            )
        if action["action"] == "inspect_workflow":
            return self._execute_inspect_workflow(
                agent,
                action,
                prompt_artifact_path=prompt_artifact_path,
            )
        if action["action"] == "write_report":
            return self._execute_write_report(
                agent,
                action,
                prompt_artifact_path=prompt_artifact_path,
            )
        if action["action"] == "send_message":
            return self._execute_send_message(
                agent,
                action,
                prompt_artifact_path=prompt_artifact_path,
            )
        if action["action"] == "ask_parent":
            return self._execute_ask_parent(
                agent,
                action,
                prompt_artifact_path=prompt_artifact_path,
            )
        return self._execute_idle(
            agent,
            action,
            prompt_artifact_path=prompt_artifact_path,
        )

    def _execute_call_capability(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        step_call_count: list[int],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        step_call_count[0] += 1

        capability_name = action["capability"].strip()
        config = dict(action.get("config") or {})
        store_as = action.get("store_as")
        step_id = f"{agent.agent_id}:{agent.current_iteration + 1}"

        result = self._capability_runner.run_capability(
            capability_name,
            config,
            agent.agent_id,
            mode="direct",
            step_id=step_id,
        )

        stored_as = None
        if store_as and result.status in {"succeeded", "denied", "requires_approval"}:
            agent_ctx = self._scoped_agents.require_agent(agent.agent_id)
            agent_ctx.step_context[store_as] = result.output
            self._scoped_agents.save_agent(agent_ctx)
            stored_as = store_as

        return self._persist_step(
            agent,
            action=action,
            status_after=action["next_status"],
            message=f"capability {capability_name}: {result.status}",
            capability_result={
                "status": result.status,
                "output": result.output,
                "error": result.error,
                "duration_ms": result.duration_ms,
                "capability": result.capability,
                "decision": dict(result.decision),
                "approval_request_id": result.approval_request_id,
                "audit_record_path": result.audit_record_path,
            },
            stored_as=stored_as,
            prompt_artifact_path=prompt_artifact_path,
        )

    def _execute_create_workflow(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        compile_context = self._build_create_workflow_context(agent, action)
        client = (
            self._workflow_compiler_client or
            getattr(self._workflow_authoring, "_workflow_compiler_client", None)
        )
        if client is not None:
            compiled = client.compile(
                action["workflow_request"],
                compile_context,
                agent.agent_id,
                agent.agent_id,
                "preview_only",
            )
            preview = compiled.envelope.preview
            spec = compiled.envelope.spec
            assumptions = compiled.envelope.assumptions
            risks = compiled.envelope.risks
            needs_confirmation = compiled.envelope.needs_confirmation
        else:
            authored = self._workflow_authoring.author_request(
                action["workflow_request"],
                caller_agent_id=agent.agent_id,
                owner_agent_id=agent.agent_id,
            )
            preview = authored.preview_text
            spec = authored.spec
            assumptions = list(authored.assumptions)
            risks = list(authored.risks)
            needs_confirmation = bool(authored.needs_confirmation)

        # When confirmation is explicitly required at the run-policy level,
        # always stop for review. When that policy is disabled (for example
        # after the parent already reviewed the plan), submit immediately
        # even if the compiler envelope would normally ask for confirmation.
        should_require_confirmation = bool(self._require_confirmation_for_workflows)

        if not should_require_confirmation:
            submission = self._workflow_authoring.submit(
                spec,
                created_by=Provenance(type="agent", id=agent.agent_id),
                caller_agent_id=agent.agent_id,
                owner_agent_id=agent.agent_id,
                workflow_metadata=(
                    {
                        "compiled_with_memory": compiled.compiled_with_memory,
                        "memory_refs_used": list(compiled.envelope.memory_refs_used),
                        "memory_tools_used": list(compiled.memory_tools_used or []),
                        "memory_context_summary": compiled.memory_context_summary,
                    }
                    if client is not None else None
                ),
            )
            created_workflow = self._workflow_store.load_workflow(submission.workflow_id)
            return self._persist_step(
                agent,
                action=action,
                status_after=action["next_status"],
                message=submission.message,
                workflow_id=submission.workflow_id,
                created_workflow_id=submission.workflow_id,
                created_workflow_status=(
                    created_workflow.status.value if created_workflow is not None else None
                ),
                workflow_submitted=True,
                prompt_artifact_path=prompt_artifact_path,
            )

        report_path = self._scoped_agents.write_report(
            agent.agent_id,
            self._build_confirmation_report(
                agent,
                action["reason"],
                preview,
                assumptions,
                risks,
            ),
        )
        parent_message = None
        if agent.parent_agent_id:
            parent_message = self._send_agent_message(
                agent,
                kind="report",
                subject=f"Workflow confirmation needed from {agent.title}",
                body="\n".join([
                    "Workflow creation requires confirmation.",
                    f"Agent: {agent.agent_id}",
                    f"Reason: {action['reason']}",
                    f"Report path: {report_path}",
                ]),
                to_agent_id=agent.parent_agent_id,
            )
        result = self._persist_step(
            agent,
            action=action,
            status_after="reporting",
            message="workflow creation requires confirmation",
            report_path=str(report_path),
            message_id=parent_message.message_id if parent_message is not None else None,
            created_parent_message_id=(
                parent_message.message_id if parent_message is not None else None
            ),
            message_to_agent_id=parent_message.to_agent_id if parent_message is not None else None,
            confirmation_required=True,
            workflow_submitted=False,
            prompt_artifact_path=prompt_artifact_path,
        )
        self._emit_mrn_reported(agent, result)
        return result

    def _build_create_workflow_context(self, agent: PersistentAgent, action: dict[str, Any]) -> str:
        parts = [
            f"Agent: {agent.agent_id} ({agent.title})",
            f"Mission: {_compact(agent.mission, limit=400)}",
            f"Scoped workflows visible: {len(self._workflow_summaries(agent))}",
        ]
        if isinstance(action.get("workflow_context"), str) and action["workflow_context"].strip():
            parts.extend([
                "Extra workflow context:",
                action["workflow_context"].strip(),
            ])
        return "\n\n".join(parts)

    def _build_confirmation_report(
        self,
        agent: PersistentAgent,
        reason: str,
        preview: str,
        assumptions: list[str],
        risks: list[str],
    ) -> str:
        lines = [
            f"# MRn Workflow Preview for {agent.title}",
            "",
            f"- agent_id: {agent.agent_id}",
            f"- mission: {_compact(agent.mission, limit=240)}",
            f"- iteration: {agent.current_iteration + 1}",
            f"- reason: {reason}",
            "",
            "## Preview",
            preview.strip() or "-",
            "",
            "## Assumptions",
        ]
        if assumptions:
            lines.extend(f"- {item}" for item in assumptions)
        else:
            lines.append("- none")
        lines.extend(["", "## Risks"])
        if risks:
            lines.extend(f"- {item}" for item in risks)
        else:
            lines.append("- none")
        return "\n".join(lines)

    def _execute_inspect_workflow(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        workflow = self._workflow_store.load_workflow(action["workflow_id"])
        if workflow is None:
            raise ValueError(f"workflow not found: {action['workflow_id']}")
        workflow = self._scoped_agents.normalize_workflow_ownership(workflow)
        if not self._scoped_agents.can_agent_access_workflow(agent.agent_id, workflow):
            raise AgentScopeError("access denied: workflow not in agent scope")
        summary = self._summarize_workflow(workflow)
        return self._persist_step(
            agent,
            action=action,
            status_after=action["next_status"],
            message=f"inspected workflow {workflow.workflow_id}",
            workflow_id=workflow.workflow_id,
            workflow_summary=summary,
            prompt_artifact_path=prompt_artifact_path,
        )

    def _summarize_workflow(self, workflow: Workflow) -> dict[str, Any]:
        tasks = []
        for task in workflow.tasks.values():
            output = self._workflow_store.load_task_output(workflow.workflow_id, task.task_id)
            tasks.append({
                "task_id": task.task_id,
                "label": task.label,
                "status": task.status.value,
                "summary": task.result_summary,
                "output_summary": output.summary if output is not None else None,
                "output_text": _compact(output.text, limit=180) if output is not None else None,
            })
        events = [
            {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "task_id": event.task_id,
                "message": _compact(event.message, limit=120),
            }
            for event in self._workflow_store.load_events(workflow.workflow_id, limit=5)
        ]
        return {
            "workflow_id": workflow.workflow_id,
            "title": workflow.title,
            "status": workflow.status.value,
            "owner_agent_id": workflow.owner_agent_id,
            "tasks": tasks,
            "recent_events": events,
        }

    def _execute_write_report(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        content = "\n".join([
            f"# MRn Report for {agent.title}",
            "",
            f"- mission: {_compact(agent.mission, limit=240)}",
            f"- iteration: {agent.current_iteration + 1}",
            f"- reason: {action['reason']}",
            "",
            action["report"].rstrip(),
        ])
        report_path = self._scoped_agents.write_report(agent.agent_id, content)
        message = None
        if agent.parent_agent_id:
            message = self._send_agent_message(
                agent,
                kind="report",
                subject=f"Report from {agent.title}",
                body="\n".join([content, "", f"Report path: {report_path}"]),
            )
        result = self._persist_step(
            agent,
            action=action,
            status_after=action["next_status"],
            message="report written",
            report_path=str(report_path),
            message_id=message.message_id if message is not None else None,
            created_parent_message_id=message.message_id if message is not None else None,
            message_to_agent_id=message.to_agent_id if message is not None else None,
            prompt_artifact_path=prompt_artifact_path,
        )
        self._emit_mrn_reported(agent, result)
        return result

    def _execute_send_message(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        message = self._send_agent_message(
            agent,
            kind=action["message_kind"],
            subject=action["message_subject"],
            body=action["message_body"],
            workflow_id=action.get("workflow_id"),
            to_agent_id=action.get("to_agent_id"),
        )
        return self._persist_step(
            agent,
            action=action,
            status_after=action["next_status"],
            message=f"message sent to {message.to_agent_id}",
            workflow_id=action.get("workflow_id"),
            message_id=message.message_id,
            created_parent_message_id=(
                message.message_id
                if message.to_agent_id == agent.parent_agent_id
                else None
            ),
            message_to_agent_id=message.to_agent_id,
            prompt_artifact_path=prompt_artifact_path,
        )

    def _execute_ask_parent(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        message = self._send_agent_message(
            agent,
            kind="question",
            subject=f"Parent request from {agent.title}",
            body=action["parent_request"],
            to_agent_id=agent.parent_agent_id,
        )
        return self._persist_step(
            agent,
            action=action,
            status_after="waiting",
            message="parent clarification requested",
            message_id=message.message_id,
            parent_request=action["parent_request"],
            created_parent_message_id=message.message_id,
            message_to_agent_id=message.to_agent_id,
            prompt_artifact_path=prompt_artifact_path,
        )

    def _execute_idle(
        self,
        agent: PersistentAgent,
        action: dict[str, Any],
        *,
        prompt_artifact_path: Optional[str] = None,
    ) -> MRnStepResult:
        return self._persist_step(
            agent,
            action=action,
            status_after="idle",
            message="agent remains idle",
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
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
