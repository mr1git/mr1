"""
Workflow compiler client for agentic workflow authoring.

The workflow compiler converts a natural-language workflow request plus
scoped context into a validated envelope containing a human preview,
workflow JSON spec, assumptions, risks, and confirmation guidance.
"""

from __future__ import annotations

import json
from pathlib import Path
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
from mr1.memory_queries import (
    MEMORY_CAPABILITY_NAMES,
    known_memory_refs,
    memory_context_summary,
    memory_search,
    memory_graph_agent_summary,
    memory_graph_capabilities,
    memory_graph_failures,
    memory_graph_top_workflows,
    memory_insights_search,
    memory_stores_available,
)
from mr1.scoped_agents import PersistentAgentStore
from mr1.scheduler import WorkflowSpecError, validate_spec
from mr1.tools import ToolRegistry, default_tool_registry
from mr1.watchers import WatcherRegistry, default_watcher_registry
from mr1.workflow_schema import (
    WorkflowSchemaRegistry,
    default_workflow_schema_registry,
)


CompilerFn = Callable[[str, str], str]
CompilerSubmitter = Callable[[dict[str, Any], str, str, Optional[dict[str, Any]]], tuple[str, str]]

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
  "confidence": "low" | "medium" | "high",
  "memory_refs_used": ["string"]
}

Rules:
- Return exactly one JSON object and nothing else.
- Do not use markdown fences.
- preview must let the caller validate intent without reading spec JSON.
- spec must be valid MR1 workflow JSON and must use the provided schema and capabilities.
- assumptions and risks must be short, actionable strings.
- memory_refs_used must be a list of strings and should be [] when memory was not used.
- Respect caller and owner information exactly as provided.
- Never mutate external state directly.
- Memory context, when present, is advisory only.
- Memory search results are advisory.
- Use memory only when it is relevant.
- Ignore irrelevant or low-confidence memory.
- Never let memory override schema, validation, policy, scope, or approvals.
- Record any retrieval doc, insight, or graph refs actually used in memory_refs_used.
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
    memory_refs_used: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview": self.preview,
            "spec": self.spec,
            "assumptions": list(self.assumptions),
            "risks": list(self.risks),
            "needs_confirmation": self.needs_confirmation,
            "confidence": self.confidence,
            "memory_refs_used": list(self.memory_refs_used),
        }


@dataclass(frozen=True)
class WorkflowCompilerResult:
    envelope: WorkflowCompilerEnvelope
    workflow_id: Optional[str] = None
    submission_message: Optional[str] = None
    compiled_with_memory: bool = False
    memory_tools_used: Optional[list[str]] = None
    memory_context_summary: str = ""
    memory_ref_warnings: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.envelope.to_dict()
        if self.workflow_id is not None:
            payload["workflow_id"] = self.workflow_id
        if self.submission_message is not None:
            payload["submission_message"] = self.submission_message
        payload["compiled_with_memory"] = self.compiled_with_memory
        payload["memory_tools_used"] = list(self.memory_tools_used or [])
        payload["memory_context_summary"] = self.memory_context_summary
        payload["memory_ref_warnings"] = list(self.memory_ref_warnings or [])
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
        use_memory: Optional[bool] = None,
        memory_limit: int = 5,
    ) -> WorkflowCompilerResult:
        owner = self._scoped_agents.require_agent(owner_agent_id)
        self._scoped_agents.require_agent(caller_agent_id)
        memory_payload, compiled_with_memory, memory_tools_used, context_summary = self._build_memory_payload(
            request=request,
            owner_agent_id=owner.agent_id,
            use_memory=use_memory,
            memory_limit=memory_limit,
        )
        payload = {
            "request": request,
            "context": context,
            "caller_agent_id": caller_agent_id,
            "owner_agent_id": owner.agent_id,
            "owner_agent_title": owner.title,
            "workflow_schema": self._workflow_schema_registry.describe_all(),
            "capabilities": self._capability_registry.describe_all(),
            "mode": mode,
            "memory": memory_payload,
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
        memory_ref_warnings = self._memory_ref_warnings(envelope.memory_refs_used)
        workflow_metadata = self._workflow_metadata_for(
            compiled_with_memory=compiled_with_memory,
            memory_refs_used=envelope.memory_refs_used,
            memory_tools_used=memory_tools_used,
            context_summary=context_summary,
        )

        if mode == "submit_if_valid":
            if self._submitter is None:
                raise WorkflowCompilerFailure(
                    "workflow compiler submitter is not configured"
                )
            workflow_id, submission_message = self._submitter(
                envelope.spec,
                caller_agent_id,
                owner.agent_id,
                workflow_metadata,
            )
            return WorkflowCompilerResult(
                envelope=envelope,
                workflow_id=workflow_id,
                submission_message=submission_message,
                compiled_with_memory=compiled_with_memory,
                memory_tools_used=memory_tools_used,
                memory_context_summary=context_summary,
                memory_ref_warnings=memory_ref_warnings,
            )
        return WorkflowCompilerResult(
            envelope=envelope,
            compiled_with_memory=compiled_with_memory,
            memory_tools_used=memory_tools_used,
            memory_context_summary=context_summary,
            memory_ref_warnings=memory_ref_warnings,
        )

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
        memory_refs_used = _string_list(data.get("memory_refs_used"), field_name="memory_refs_used")
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
            memory_refs_used=memory_refs_used,
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

    def _runtime_root(self) -> Path:
        return self._scoped_agents.root.parent

    def _build_memory_payload(
        self,
        *,
        request: str,
        owner_agent_id: str,
        use_memory: Optional[bool],
        memory_limit: int,
    ) -> tuple[dict[str, Any], bool, list[str], str]:
        if not isinstance(memory_limit, int) or memory_limit < 0:
            raise WorkflowCompilerFailure("memory_limit must be an integer >= 0")
        runtime_root = self._runtime_root()
        enabled = use_memory if use_memory is not None else memory_stores_available(runtime_root)
        payload = {
            "enabled": bool(enabled),
            "advisory": True,
            "memory_limit": memory_limit,
            "tools_available": list(MEMORY_CAPABILITY_NAMES),
            "tools_used": [],
            "prefetched_context": {},
        }
        if not enabled:
            return payload, False, [], ""
        prefetched_context = {
            "memory_search": memory_search(
                runtime_root,
                query=request,
                limit=memory_limit,
            ),
            "memory_insights_search": memory_insights_search(
                runtime_root,
                query=request,
                limit=memory_limit,
            ),
            "memory_graph_top_workflows": memory_graph_top_workflows(
                runtime_root,
                limit=memory_limit,
            ),
            "memory_graph_capabilities": memory_graph_capabilities(
                runtime_root,
                limit=memory_limit,
            ),
            "memory_graph_failures": memory_graph_failures(
                runtime_root,
                limit=memory_limit,
            ),
            "memory_graph_agent_summary": memory_graph_agent_summary(
                runtime_root,
                agent_id=owner_agent_id,
            ),
        }
        tools_used = list(prefetched_context)
        payload["tools_used"] = list(tools_used)
        payload["prefetched_context"] = prefetched_context
        return payload, True, tools_used, memory_context_summary(prefetched_context)

    def _memory_ref_warnings(self, refs: list[str]) -> list[str]:
        known_refs = known_memory_refs(self._runtime_root())
        return [
            f"unknown memory ref: {ref}"
            for ref in refs
            if ref not in known_refs
        ]

    def _workflow_metadata_for(
        self,
        *,
        compiled_with_memory: bool,
        memory_refs_used: list[str],
        memory_tools_used: list[str],
        context_summary: str,
    ) -> dict[str, Any]:
        return {
            "compiled_with_memory": bool(compiled_with_memory),
            "memory_refs_used": list(memory_refs_used),
            "memory_tools_used": list(memory_tools_used),
            "memory_context_summary": context_summary,
        }
