"""
Workflow compiler client for agentic workflow authoring.

The workflow compiler converts a natural-language workflow request plus
scoped context into a validated envelope containing a human preview,
workflow JSON spec, assumptions, risks, and confirmation guidance.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mr1.agents import (
    AgentRuntimeError,
    build_agent_command,
    load_agent_runtime_config,
    parse_agent_json_envelope,
)
from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.core import Dispatcher, PermissionDenied
from mr1.scoped_agents import PersistentAgentStore
from mr1.scheduler import WorkflowSpecError, validate_spec
from mr1.tools import ToolRegistry, default_tool_registry
from mr1.watchers import WatcherRegistry, default_watcher_registry
from mr1.workflow_schema import (
    WorkflowSchemaRegistry,
    default_workflow_schema_registry,
)


CompilerFn = Callable[[str, str], str]
CompilerSubmitter = Callable[[dict[str, Any], str, str], tuple[str, str]]

_ALLOWED_CONFIDENCE = frozenset({"low", "medium", "high"})

_WORKFLOW_COMPILER_SYSTEM_PROMPT = """\
You are WorkflowCompiler for MR1.

You receive a JSON object with:
- request
- context
- caller_agent_id
- owner_agent_id
- owner_agent_title
- workflow_schema
- capabilities
- mode

Return JSON only with this exact top-level shape:
{
  "preview": "string",
  "spec": {...},
  "assumptions": ["string"],
  "risks": ["string"],
  "needs_confirmation": true,
  "confidence": "low" | "medium" | "high"
}

Rules:
- Return exactly one JSON object and nothing else.
- Do not use markdown fences.
- preview must let the caller validate intent without reading spec JSON.
- spec must be valid MR1 workflow JSON and must use the provided schema and capabilities.
- assumptions and risks must be short, actionable strings.
- Respect caller and owner information exactly as provided.
- Never mutate external state directly.
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _snippet(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) > limit:
        compact = compact[:limit] + "..."
    return compact


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload:
        raise WorkflowCompilerFailure(
            "workflow compiler returned empty output"
        )
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
        raise WorkflowCompilerFailure(
            f"workflow compiler returned invalid JSON: {exc}; raw={_snippet(payload)!r}"
        ) from exc
    if not isinstance(data, dict):
        raise WorkflowCompilerFailure(
            f"workflow compiler must return a JSON object; raw={_snippet(payload)!r}"
        )
    return data


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise WorkflowCompilerFailure(f"workflow compiler field '{field_name}' must be a list of strings")
    return list(value)


def run_workflow_compiler_agent(system_prompt: str, prompt: str) -> str:
    config = load_agent_runtime_config("workflow_compiler")
    cmd = build_agent_command(
        "workflow_compiler",
        prompt,
        config=config,
    )
    cmd.extend(["--append-system-prompt", system_prompt])
    cli_flags = [tok for tok in cmd[1:] if tok.startswith("-")]
    dispatcher = Dispatcher()
    try:
        dispatcher.validate_full_spawn(
            "workflow_compiler",
            cli_flags,
            list(config.get("allowed_tools", [])),
        )
    except PermissionDenied as exc:
        raise WorkflowCompilerFailure(str(exc)) from exc
    timeout_s = int(config.get("timeout_s") or 300)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkflowCompilerFailure(
            f"workflow compiler timed out after {timeout_s}s"
        ) from exc
    except OSError as exc:
        raise WorkflowCompilerFailure(f"failed to run workflow compiler: {exc}") from exc
    raw_output = proc.stdout or ""
    if proc.returncode != 0:
        detail = (proc.stderr or raw_output).strip() or f"exit {proc.returncode}"
        raise WorkflowCompilerFailure(f"workflow compiler failed: {detail}")
    try:
        parsed = parse_agent_json_envelope(raw_output)
    except AgentRuntimeError as exc:
        raise WorkflowCompilerFailure(str(exc)) from exc
    if parsed["is_error"]:
        detail = parsed["text"] or "workflow compiler agent returned an error"
        raise WorkflowCompilerFailure(detail)
    return parsed["text"]


@dataclass(frozen=True)
class WorkflowCompilerEnvelope:
    preview: str
    spec: dict[str, Any]
    assumptions: list[str]
    risks: list[str]
    needs_confirmation: bool
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview": self.preview,
            "spec": self.spec,
            "assumptions": list(self.assumptions),
            "risks": list(self.risks),
            "needs_confirmation": self.needs_confirmation,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class WorkflowCompilerResult:
    envelope: WorkflowCompilerEnvelope
    workflow_id: Optional[str] = None
    submission_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.envelope.to_dict()
        if self.workflow_id is not None:
            payload["workflow_id"] = self.workflow_id
        if self.submission_message is not None:
            payload["submission_message"] = self.submission_message
        return payload


class WorkflowCompilerFailure(ValueError):
    """Deterministic workflow compiler failure."""


class WorkflowCompilerClient:
    def __init__(
        self,
        *,
        compiler: Optional[CompilerFn] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        workflow_schema_registry: Optional[WorkflowSchemaRegistry] = None,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
        watcher_registry: Optional[WatcherRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        submitter: Optional[CompilerSubmitter] = None,
    ):
        self._compiler = compiler or run_workflow_compiler_agent
        self._capability_registry = capability_registry or default_capability_registry()
        self._workflow_schema_registry = (
            workflow_schema_registry or default_workflow_schema_registry()
        )
        self._scoped_agents = scoped_agent_store or PersistentAgentStore()
        self._watcher_registry = watcher_registry or default_watcher_registry()
        self._tool_registry = tool_registry or default_tool_registry()
        self._submitter = submitter

    def compile(
        self,
        request: str,
        context: str,
        caller_agent_id: str,
        owner_agent_id: str,
        mode: str,
    ) -> WorkflowCompilerResult:
        owner = self._scoped_agents.require_agent(owner_agent_id)
        self._scoped_agents.require_agent(caller_agent_id)
        payload = {
            "request": request,
            "context": context,
            "caller_agent_id": caller_agent_id,
            "owner_agent_id": owner.agent_id,
            "owner_agent_title": owner.title,
            "workflow_schema": self._workflow_schema_registry.describe_all(),
            "capabilities": self._capability_registry.describe_all(),
            "mode": mode,
        }
        raw = self._compiler(_WORKFLOW_COMPILER_SYSTEM_PROMPT, _json_dumps(payload))
        try:
            envelope = self._parse_and_validate_envelope(raw)
        except WorkflowCompilerFailure as exc:
            correction_prompt = self._build_correction_prompt(
                payload=payload,
                invalid_output=raw,
                error=str(exc),
            )
            corrected_raw = self._compiler(_WORKFLOW_COMPILER_SYSTEM_PROMPT, correction_prompt)
            try:
                envelope = self._parse_and_validate_envelope(corrected_raw)
            except WorkflowCompilerFailure as second_exc:
                raise WorkflowCompilerFailure(
                    f"workflow compilation failed: {second_exc}"
                ) from second_exc

        if mode == "submit_if_valid":
            if self._submitter is None:
                raise WorkflowCompilerFailure(
                    "workflow compiler submitter is not configured"
                )
            workflow_id, submission_message = self._submitter(
                envelope.spec,
                caller_agent_id,
                owner.agent_id,
            )
            return WorkflowCompilerResult(
                envelope=envelope,
                workflow_id=workflow_id,
                submission_message=submission_message,
            )
        return WorkflowCompilerResult(envelope=envelope)

    def _parse_and_validate_envelope(self, raw: str) -> WorkflowCompilerEnvelope:
        data = _extract_json_object(raw)
        preview = data.get("preview")
        if not isinstance(preview, str) or not preview.strip():
            raise WorkflowCompilerFailure(
                "workflow compiler field 'preview' must be a non-empty string"
            )
        spec = data.get("spec")
        if not isinstance(spec, dict):
            raise WorkflowCompilerFailure(
                "workflow compiler field 'spec' must be a JSON object"
            )
        assumptions = _string_list(data.get("assumptions"), field_name="assumptions")
        risks = _string_list(data.get("risks"), field_name="risks")
        needs_confirmation = data.get("needs_confirmation")
        if not isinstance(needs_confirmation, bool):
            raise WorkflowCompilerFailure(
                "workflow compiler field 'needs_confirmation' must be a boolean"
            )
        confidence = data.get("confidence")
        if confidence not in _ALLOWED_CONFIDENCE:
            raise WorkflowCompilerFailure(
                "workflow compiler field 'confidence' must be one of: low, medium, high"
            )
        try:
            validate_spec(
                spec,
                watcher_registry=self._watcher_registry,
                tool_registry=self._tool_registry,
            )
        except WorkflowSpecError as exc:
            raise WorkflowCompilerFailure(str(exc)) from exc
        return WorkflowCompilerEnvelope(
            preview=preview.strip(),
            spec=spec,
            assumptions=assumptions,
            risks=risks,
            needs_confirmation=needs_confirmation,
            confidence=confidence,
        )

    def _build_correction_prompt(
        self,
        *,
        payload: dict[str, Any],
        invalid_output: str,
        error: str,
    ) -> str:
        return "\n\n".join([
            "Return corrected JSON only.",
            "Preserve user intent.",
            f"Validation or envelope error:\n{error}",
            "Original compile payload:",
            _json_dumps(payload),
            "Invalid output:",
            invalid_output,
        ])
