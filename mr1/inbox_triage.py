"""
Bounded inbox triage for the root MR1 inbox.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from mr1.agents import AgentRuntimeError, parse_agent_json_envelope
from mr1.brain_tools import governed_brain_tools
from mr1.core import Dispatcher, PermissionDenied
from mr1.worker_runner import MockRunner
from mr1.messages import MessageStore, PersistentMessage
from mr1.mrn_loop import MRnStepRunner
from mr1.mrn_run import MRnRunPolicy, MRnRunRunner
from mr1.scheduler import Scheduler
from mr1.scoped_agents import AgentStore
from mr1.workflow_authoring import PendingWorkflowDraft, WorkflowAuthoringService
from mr1.workflow_compiler import WorkflowCompilerClient
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


_PKG_ROOT = Path(__file__).resolve().parent
_MR1_CONFIG_PATH = _PKG_ROOT / "agents" / "mr1.yml"
_DEFAULT_TIMEOUT_S = 300
_ALLOWED_TRIAGE_ACTIONS = frozenset({
    "mark_read",
    "archive",
    "reply_message",
    "agent_run",
    "agent_step",
    "assign_mission",
    "create_workflow",
    "ask_user",
    "idle",
})
_TRIAGE_SYSTEM_PROMPT = """\
You are MR1 inbox triage for a bounded local coordination pass.

Return JSON only with this exact top-level shape:
{
  "actions": [
    {
      "type": "mark_read" | "archive" | "reply_message" | "agent_run" | "agent_step" | "assign_mission" | "create_workflow" | "ask_user" | "idle",
      "message_id": "optional",
      "agent_id": "optional",
      "reply_body": "optional",
      "mission": "optional",
      "workflow_request": "optional",
      "reason": "required short reason"
    }
  ],
  "summary": "short triage summary"
}

Rules:
- Return exactly one JSON object and nothing else.
- Do not use markdown fences.
- Choose at most the policy max_actions actions.
- Every action must include a non-empty reason.
- Only use the allowed action types listed in the prompt.
- Never propose loops, daemons, background jobs, or external messaging.
- Child-agent questions may be summarized, surfaced, marked read, archived, assigned, or converted into ask_user/workflow suggestions.
- Do not send substantive parent replies to child-agent questions unless reply_message is explicitly enabled by policy.
- If user clarification is needed, use ask_user.
- If nothing should happen, use idle.
"""

ReasonerFn = Callable[[str, str], str]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _compact(text: Any, *, limit: int = 1000) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.strip()
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _load_mr1_config(path: Path = _MR1_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_inbox_triage_reasoner(system_prompt: str, prompt: str) -> str:
    config = _load_mr1_config()
    allowed_tools = governed_brain_tools(config.get("allowed_tools") or [])
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    if config.get("model"):
        cmd.extend(["--model", str(config["model"])])
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])
    cmd.extend(["--append-system-prompt", system_prompt])
    dispatcher = Dispatcher()
    cli_flags = [token for token in cmd[1:] if token.startswith("-")]
    try:
        dispatcher.validate_full_spawn("mr1", cli_flags, allowed_tools)
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
        raise ValueError(f"inbox triage timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise ValueError(f"failed to run inbox triage reasoner: {exc}") from exc
    raw_output = proc.stdout or ""
    if proc.returncode != 0:
        detail = (proc.stderr or raw_output).strip() or f"exit {proc.returncode}"
        raise ValueError(f"inbox triage reasoner failed: {detail}")
    try:
        parsed = parse_agent_json_envelope(raw_output)
    except AgentRuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if parsed["is_error"]:
        raise ValueError(parsed["text"] or "inbox triage reasoner returned an error")
    return parsed["text"]


@dataclass(frozen=True)
class InboxTriagePolicy:
    max_messages: int = 10
    max_actions: int = 3
    allowed_actions: Optional[list[str]] = None
    allow_reply_messages: bool = False
    auto_mark_read: bool = True
    auto_archive: bool = False
    include_archived: bool = False

    def validate(self) -> "InboxTriagePolicy":
        if self.max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        if self.max_actions < 1:
            raise ValueError("max_actions must be >= 1")
        if self.allowed_actions is not None:
            unknown = sorted(set(self.allowed_actions) - _ALLOWED_TRIAGE_ACTIONS)
            if unknown:
                raise ValueError(
                    "unknown actions rejected deterministically: "
                    + ", ".join(unknown)
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_messages": self.max_messages,
            "max_actions": self.max_actions,
            "allowed_actions": list(self.allowed_actions) if self.allowed_actions is not None else None,
            "allow_reply_messages": self.allow_reply_messages,
            "auto_mark_read": self.auto_mark_read,
            "auto_archive": self.auto_archive,
            "include_archived": self.include_archived,
        }


@dataclass(frozen=True)
class InboxTriageResult:
    processed_messages: list[str]
    actions_executed: list[dict[str, Any]]
    summary: str
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_messages": list(self.processed_messages),
            "actions_executed": list(self.actions_executed),
            "summary": self.summary,
            "counts": dict(self.counts),
        }


class InboxTriageRunner:
    def __init__(
        self,
        *,
        workflow_store: Optional[WorkflowStore] = None,
        scoped_agent_store: Optional[AgentStore] = None,
        message_store: Optional[MessageStore] = None,
        reasoner: Optional[ReasonerFn] = None,
        mrn_step_runner: Optional[MRnStepRunner] = None,
        mrn_run_runner: Optional[MRnRunRunner] = None,
        workflow_compiler_client: Optional[WorkflowCompilerClient] = None,
        workflow_authoring_service: Optional[WorkflowAuthoringService] = None,
        pending_workflow_state: Optional[Any] = None,
    ):
        self._workflow_store = workflow_store or WorkflowStore()
        self._scoped_agents = scoped_agent_store or AgentStore(
            root=self._workflow_store.root.parent / "agents"
        )
        self._message_store = message_store or MessageStore(
            root=self._workflow_store.root.parent / "messages",
            scoped_agent_store=self._scoped_agents,
        )
        self._reasoner = reasoner or run_inbox_triage_reasoner
        self._workflow_compiler_client = (
            workflow_compiler_client or WorkflowCompilerClient(
                scoped_agent_store=self._scoped_agents,
            )
        )
        self._workflow_authoring = (
            workflow_authoring_service or self._build_default_workflow_authoring_service()
        )
        self._mrn_step_runner = mrn_step_runner or MRnStepRunner(
            workflow_store=self._workflow_store,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
            workflow_authoring_service=self._workflow_authoring,
        )
        self._mrn_run_runner = mrn_run_runner or MRnRunRunner(
            workflow_store=self._workflow_store,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
            workflow_compiler=self._workflow_compiler_client._compiler,
        )
        self._pending_workflow_state = pending_workflow_state

    def _build_default_workflow_authoring_service(self) -> WorkflowAuthoringService:
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
            authoring_backend="compiler_agent",
            workflow_compiler_client=self._workflow_compiler_client,
        )

    def run(
        self,
        policy: InboxTriagePolicy,
        caller_agent_id: Optional[str] = None,
    ) -> InboxTriageResult:
        validated_policy = policy.validate()
        root_agent_id = self._scoped_agents.root_agent_id
        resolved_caller = caller_agent_id or root_agent_id
        self._scoped_agents.require_agent(root_agent_id)
        self._scoped_agents.require_agent(resolved_caller)
        selected_messages = self._select_messages(root_agent_id, validated_policy)
        if not selected_messages:
            return InboxTriageResult(
                processed_messages=[],
                actions_executed=[],
                summary="No unread inbox messages.",
                counts=self._new_counts(),
            )

        selected_message_ids = {message.message_id for message in selected_messages}
        raw_plan = self._reasoner(
            _TRIAGE_SYSTEM_PROMPT,
            self._build_prompt(validated_policy, resolved_caller, selected_messages),
        )
        try:
            triage_plan = self._parse_and_validate_plan(
                raw_plan,
                policy=validated_policy,
                selected_message_ids=selected_message_ids,
                caller_agent_id=resolved_caller,
            )
        except ValueError as exc:
            corrected_raw_plan = self._reasoner(
                _TRIAGE_SYSTEM_PROMPT,
                self._build_correction_prompt(
                    validated_policy,
                    resolved_caller,
                    selected_messages,
                    raw_plan,
                    str(exc),
                ),
            )
            try:
                triage_plan = self._parse_and_validate_plan(
                    corrected_raw_plan,
                    policy=validated_policy,
                    selected_message_ids=selected_message_ids,
                    caller_agent_id=resolved_caller,
                )
            except ValueError as second_exc:
                raise ValueError(f"inbox triage failed: {second_exc}") from second_exc

        counts = self._new_counts()
        action_records: list[dict[str, Any]] = []
        summary = triage_plan["summary"]

        for action in triage_plan["actions"][:validated_policy.max_actions]:
            if (
                validated_policy.allowed_actions is not None
                and action["type"] not in set(validated_policy.allowed_actions)
            ):
                action_records.append({
                    **self._action_record_base(action),
                    "status": "stopped",
                })
                summary = f"{summary} Stopped on disallowed action: {action['type']}."
                break
            action_records.append(
                self._execute_action(
                    action,
                    caller_agent_id=resolved_caller,
                    root_agent_id=root_agent_id,
                    selected_messages=selected_messages,
                    counts=counts,
                )
            )

        self._post_process_messages(
            selected_messages,
            validated_policy,
            counts,
        )
        return InboxTriageResult(
            processed_messages=[message.message_id for message in selected_messages],
            actions_executed=action_records,
            summary=summary,
            counts=counts,
        )

    def _select_messages(
        self,
        root_agent_id: str,
        policy: InboxTriagePolicy,
    ) -> list[PersistentMessage]:
        messages = self._message_store.list_inbox(
            root_agent_id,
            include_archived=policy.include_archived,
        )
        unread = [message for message in messages if message.status == "unread"]
        return unread[:policy.max_messages]

    def _build_prompt(
        self,
        policy: InboxTriagePolicy,
        caller_agent_id: str,
        messages: list[PersistentMessage],
    ) -> str:
        payload = {
            "policy": policy.to_dict(),
            "allowed_action_types": sorted(_ALLOWED_TRIAGE_ACTIONS),
            "caller_agent_id": caller_agent_id,
            "root_agent_id": self._scoped_agents.root_agent_id,
            "messages": [self._message_payload(message) for message in messages],
            "visible_agents": [self._agent_payload(agent) for agent in self._scoped_agents.list_visible_agents(caller_agent_id)],
        }
        return "\n\n".join([
            "Perform one bounded inbox triage pass.",
            "Return JSON only.",
            _json_dumps(payload),
        ])

    def _build_correction_prompt(
        self,
        policy: InboxTriagePolicy,
        caller_agent_id: str,
        messages: list[PersistentMessage],
        invalid_output: str,
        error: str,
    ) -> str:
        return "\n\n".join([
            "Return corrected JSON only.",
            "Preserve the original triage intent where possible.",
            f"Validation error:\n{error}",
            "Original triage context:",
            self._build_prompt(policy, caller_agent_id, messages),
            "Invalid output:",
            invalid_output,
        ])

    def _message_payload(self, message: PersistentMessage) -> dict[str, Any]:
        return {
            "message_id": message.message_id,
            "from_agent_id": message.from_agent_id,
            "to_agent_id": message.to_agent_id,
            "kind": message.kind,
            "subject": message.subject,
            "body": _compact(message.body),
            "workflow_id": message.workflow_id,
            "task_id": message.task_id,
            "created_at": message.created_at,
            "status": message.status,
        }

    def _agent_payload(self, agent: Any) -> dict[str, Any]:
        return {
            "agent_id": agent.agent_id,
            "title": agent.title,
            "mr_level": agent.mr_level,
            "status": agent.status,
            "mission": _compact(agent.mission, limit=240),
            "run_status": agent.run_status,
            "parent_agent_id": agent.parent_agent_id,
        }

    def _parse_and_validate_plan(
        self,
        raw: str,
        *,
        policy: InboxTriagePolicy,
        selected_message_ids: set[str],
        caller_agent_id: str,
    ) -> dict[str, Any]:
        payload = raw.strip()
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
        summary = data.get("summary")
        actions = data.get("actions")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be a non-empty string")
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")
        if len(actions) > policy.max_actions:
            raise ValueError("actions must not exceed policy.max_actions")
        validated_actions: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            validated_actions.append(
                self._validate_action(
                    action,
                    index=index,
                    policy=policy,
                    selected_message_ids=selected_message_ids,
                    caller_agent_id=caller_agent_id,
                )
            )
        return {"summary": summary.strip(), "actions": validated_actions}

    def _validate_action(
        self,
        action: Any,
        *,
        index: int,
        policy: InboxTriagePolicy,
        selected_message_ids: set[str],
        caller_agent_id: str,
    ) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise ValueError(f"action {index} must be an object")
        action_type = action.get("type")
        reason = action.get("reason")
        if action_type not in _ALLOWED_TRIAGE_ACTIONS:
            raise ValueError(f"action {index} has unknown type: {action_type}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"action {index} must include a non-empty reason")
        if action.get("message_id") is not None:
            if not isinstance(action["message_id"], str) or action["message_id"] not in selected_message_ids:
                raise ValueError(f"action {index} references a message outside the selected set")
        if action.get("agent_id") is not None:
            if not isinstance(action["agent_id"], str):
                raise ValueError(f"action {index} agent_id must be a string")
            if not self._scoped_agents.can_manage_agent(caller_agent_id, action["agent_id"]):
                raise ValueError(f"action {index} references an agent outside caller scope")
        required_fields = {
            "mark_read": ("message_id",),
            "archive": ("message_id",),
            "reply_message": ("message_id", "reply_body"),
            "agent_run": ("agent_id",),
            "agent_step": ("agent_id",),
            "assign_mission": ("agent_id", "mission"),
            "create_workflow": ("workflow_request",),
            "ask_user": (),
            "idle": (),
        }
        for field_name in required_fields[action_type]:
            value = action.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"action {index} must include a non-empty {field_name}")
        if action_type == "reply_message" and not policy.allow_reply_messages:
            raise ValueError("reply_message is disabled by policy")
        return dict(action)

    def _execute_action(
        self,
        action: dict[str, Any],
        *,
        caller_agent_id: str,
        root_agent_id: str,
        selected_messages: list[PersistentMessage],
        counts: dict[str, int],
    ) -> dict[str, Any]:
        action_type = action["type"]
        if action_type == "mark_read":
            self._mark_read(action["message_id"], counts)
            return {
                **self._action_record_base(action),
                "status": "executed",
            }
        if action_type == "archive":
            self._archive(action["message_id"], counts)
            return {
                **self._action_record_base(action),
                "status": "executed",
            }
        if action_type == "reply_message":
            original = self._require_message(action["message_id"])
            created = self._message_store.create_message(
                from_agent_id=root_agent_id,
                to_agent_id=original.from_agent_id,
                kind=original.kind,
                subject=f"Re: {original.subject}",
                body=action["reply_body"].rstrip(),
                workflow_id=original.workflow_id,
                task_id=original.task_id,
            )
            counts["messages_sent"] += 1
            return {
                **self._action_record_base(action),
                "status": "executed",
                "created_message_id": created.message_id,
            }
        if action_type == "agent_run":
            result = self._mrn_run_runner.run(
                action["agent_id"],
                MRnRunPolicy(max_steps=2, max_workflows_created=1),
                caller_agent_id=caller_agent_id,
            )
            counts["agents_run"] += 1
            counts["workflows_created"] += result.workflows_created
            return {
                **self._action_record_base(action),
                "status": "executed",
                "run_id": result.run_id,
                "workflows_created": result.workflows_created,
            }
        if action_type == "agent_step":
            result = self._mrn_step_runner.step(
                action["agent_id"],
                caller_agent_id=caller_agent_id,
            )
            counts["agents_run"] += 1
            if result.created_workflow_id is not None:
                counts["workflows_created"] += 1
            return {
                **self._action_record_base(action),
                "status": "executed",
                "step_iteration": result.iteration,
                "created_workflow_id": result.created_workflow_id,
            }
        if action_type == "assign_mission":
            self._scoped_agents.assign_mission(
                caller_agent_id,
                action["agent_id"],
                action["mission"],
            )
            return {
                **self._action_record_base(action),
                "status": "executed",
            }
        if action_type == "create_workflow":
            result = self._workflow_compiler_client.compile(
                request=action["workflow_request"],
                context=_json_dumps({
                    "source": "inbox_triage",
                    "messages": [self._message_payload(message) for message in selected_messages],
                }),
                caller_agent_id=root_agent_id,
                owner_agent_id=root_agent_id,
                mode="preview_only",
            )
            if result.envelope.needs_confirmation:
                if self._pending_workflow_state is None:
                    raise ValueError("pending workflow state sink is not configured")
                draft = PendingWorkflowDraft(
                    original_request=action["workflow_request"],
                    mode="create",
                    spec=result.envelope.spec,
                    preview_text=result.envelope.preview,
                    assumptions=list(result.envelope.assumptions),
                    risks=list(result.envelope.risks),
                    needs_confirmation=result.envelope.needs_confirmation,
                    confidence=result.envelope.confidence,
                    complexity="complex",
                    compiled_with_memory=result.compiled_with_memory,
                    memory_refs_used=list(result.envelope.memory_refs_used),
                    memory_tools_used=list(result.memory_tools_used or []),
                    memory_context_summary=result.memory_context_summary,
                    memory_ref_warnings=list(result.memory_ref_warnings or []),
                )
                self._pending_workflow_state.set_pending_workflow(draft.to_dict())
                return {
                    **self._action_record_base(action),
                    "status": "confirmation_required",
                    "preview_text": result.envelope.preview,
                    "assumptions": list(result.envelope.assumptions),
                    "risks": list(result.envelope.risks),
                    "confidence": result.envelope.confidence,
                    "user_message": result.envelope.preview,
                }
            submission = self._workflow_authoring.submit(
                result.envelope.spec,
                created_by=Provenance(type="agent", id="MR1"),
                caller_agent_id=root_agent_id,
                owner_agent_id=root_agent_id,
                workflow_metadata={
                    "compiled_with_memory": result.compiled_with_memory,
                    "memory_refs_used": list(result.envelope.memory_refs_used),
                    "memory_tools_used": list(result.memory_tools_used or []),
                    "memory_context_summary": result.memory_context_summary,
                },
            )
            counts["workflows_created"] += 1
            return {
                **self._action_record_base(action),
                "status": "executed",
                "created_workflow_id": submission.workflow_id,
            }
        if action_type == "ask_user":
            return {
                **self._action_record_base(action),
                "status": "ask_user",
                "user_message": action["reason"],
            }
        return {
            **self._action_record_base(action),
            "status": "executed",
        }

    def _require_message(self, message_id: str) -> PersistentMessage:
        message = self._message_store.get_message(message_id)
        if message is None:
            raise ValueError(f"message not found: {message_id}")
        return message

    def _action_record_base(self, action: dict[str, Any]) -> dict[str, Any]:
        record = {
            "type": action["type"],
            "reason": action["reason"],
        }
        for key in (
            "message_id",
            "agent_id",
            "reply_body",
            "mission",
            "workflow_request",
        ):
            if key in action and action.get(key) is not None:
                record[key] = action[key]
        return record

    def _new_counts(self) -> dict[str, int]:
        return {
            "messages_read": 0,
            "messages_archived": 0,
            "messages_sent": 0,
            "agents_run": 0,
            "workflows_created": 0,
        }

    def _mark_read(self, message_id: str, counts: dict[str, int]) -> None:
        before = self._require_message(message_id)
        after = self._message_store.mark_read(message_id)
        if after is not None and after.status == "read" and before.status != "read":
            counts["messages_read"] += 1

    def _archive(self, message_id: str, counts: dict[str, int]) -> None:
        before = self._require_message(message_id)
        after = self._message_store.archive_message(message_id)
        if after is not None and after.status == "archived" and before.status != "archived":
            counts["messages_archived"] += 1

    def _post_process_messages(
        self,
        messages: list[PersistentMessage],
        policy: InboxTriagePolicy,
        counts: dict[str, int],
    ) -> None:
        for message in messages:
            current = self._message_store.get_message(message.message_id)
            if current is None or current.status == "archived":
                continue
            if policy.auto_mark_read:
                self._mark_read(message.message_id, counts)
            current = self._message_store.get_message(message.message_id)
            if current is None or current.status == "archived":
                continue
            if policy.auto_archive:
                self._archive(message.message_id, counts)
