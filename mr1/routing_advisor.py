from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


_WORKFLOW_ID_PATTERN = re.compile(r"\bwf-\d{8}T\d{6}-[0-9a-f]{6}\b")
_TASK_ID_PATTERN = re.compile(r"\btk-\d{8}T\d{6}-[0-9a-f]{6}\b")
_MESSAGE_ID_PATTERN = re.compile(r"\bmsg-\d{8}T\d{6,}-[0-9a-f]{6}\b")
_AGENT_ID_PATTERN = re.compile(r"\bag-\d{8}T\d{6}-[0-9a-f]{6}\b")
_APPROVAL_ID_PATTERN = re.compile(r"\bcap_approval_[A-Za-z0-9_]+\b")
_REPLY_INTENT_PATTERN = re.compile(
    r"\b(reply|respond|clarify)(?:\s+to)?(?:\s+this)?(?:\s+message)?\b",
    re.IGNORECASE,
)
_INSPECTION_PHRASES = (
    "check",
    "inspect",
    "show",
    "list",
    "display",
    "view",
    "status",
    "result",
    "results",
    "findings",
    "summarize findings",
    "what happened",
    "what's running",
    "whats running",
    "did it finish",
    "did the workflow finish",
    "finish running",
    "why failed",
    "why did",
)
_META_PREFIXES = (
    "how do",
    "how would",
    "how should",
    "when would",
    "when should",
    "why would",
    "why should",
    "describe",
    "explain",
    "compare",
    "what are",
    "what is",
)
_RUN_COMMAND_VERBS = (
    "approve",
    "deny",
    "reply",
    "respond",
    "clarify",
    "kill",
    "terminate",
    "resume",
    "pause",
    "rename",
    "delete",
    "remove",
    "wipe",
    "purge",
    "message",
    "send",
    "ask",
    "tell",
    "stop",
)
_PERSISTENT_MARKERS = (
    "persistent agent",
    "owner agent",
    "own self-evolution",
    "own this area",
    "own this domain",
    "long-term owner",
    "long term owner",
    "responsible for long-term",
    "responsible for long term",
    "give it a name",
    "self-evolution",
    "self-evolving",
    "not handled by mr1 directly",
)
_PERSISTENT_IMPERATIVE_PATTERNS = (
    re.compile(r"\bcreate\s+(?:a|an)\s+(?!workflow\b)(?:\w+\s+){0,3}(?:agent|child)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:spawn|make|add|start|set\s+up)\s+(?!workflow\b)(?:(?:a|an)\s+)?(?:\w+\s+){0,4}(?:agent|agents|child|children)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bcreate an owner agent\b", re.IGNORECASE),
    re.compile(r"\bcreate a persistent agent\b", re.IGNORECASE),
    re.compile(r"\bcreate (?:a|an) agent to own\b", re.IGNORECASE),
    re.compile(r"\bhave (?:an|a) agent own\b", re.IGNORECASE),
    re.compile(r"\bdelegate this (?:domain|area|responsibility)\b", re.IGNORECASE),
)
_WORKFLOW_MODIFY_TOKENS = (
    "modify",
    "update",
    "change",
    "edit",
    "append",
    "replace",
    "rerun",
    "insert",
    "remove",
    "cancel",
    "trigger",
)
_WORKFLOW_CREATE_MARKERS = (
    "create a workflow",
    "build a workflow",
    "run a workflow",
    "execute pipeline",
    "execute a pipeline",
    "build pipeline",
    "run pipeline",
)
_WORKFLOW_ACTION_WORDS = re.compile(
    r"\b(read|write|run|check|summarize|create|generate|inspect|list|save|wait|trigger|search)\b",
    re.IGNORECASE,
)
_RERUN_VERB_PATTERN = re.compile(r"\b(rerun|re-run|retry|restart)\b", re.IGNORECASE)
_AGENT_COMMAND_SHAPE_PATTERNS = (
    re.compile(
        r"\b(?:kill|terminate|resume|pause|stop)\s+(?:(?:that|this|the)\s+)?(?!that\b|this\b|it\b|everything\b|one\b)[a-z0-9][\w-]*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:message|send|ask|tell)\s+(?:(?:that|this|the)\s+)?(?!that\b|this\b|it\b|everything\b|one\b)[a-z0-9][\w-]*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brename\s+(?:(?:that|this|the)\s+)?(?!that\b|this\b|it\b|everything\b|one\b)[a-z0-9][\w-]*(?:\s+[a-z0-9][\w-]*){0,2}\s+to\s+[a-z0-9][\w-]*\b",
        re.IGNORECASE,
    ),
)

_WORKFLOW_CONFIRMATION_PATTERNS = (
    re.compile(r"^yes(?:,\s*submit(?:\s+it)?)?[.!]?$", re.IGNORECASE),
    re.compile(r"^(?:ok|okay|submit|do it|go ahead|proceed|confirm|run it|execute)[.!]?$", re.IGNORECASE),
)

_WORKFLOW_CANCELLATION_PATTERNS = (
    re.compile(r"^(?:no|cancel|stop|abort)[.!]?$", re.IGNORECASE),
    re.compile(r"^(?:nevermind|never mind)[.!]?$", re.IGNORECASE),
    re.compile(r"^actually\s+(?:nevermind|never mind),?\s+cancel(?:\s+that)?[.!]?$", re.IGNORECASE),
)

_CONFLICTING_AGENT_LIFECYCLE_PATTERNS = (
    re.compile(
        r"\b(?:create|spawn|make|add|start|set\s+up)\b.*\b(?:agent|agents|worker|workers|child|children)\b.*\b(?:and|then|immediately)\b.*\b(?:kill|terminate|delete|remove|stop)\b.*\b(?:all|them|it)\b",
        re.IGNORECASE,
    ),
)

_LOW_CONFIDENCE_THRESHOLD = 0.70


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _explicit_ids(pattern: re.Pattern[str], user_input: str) -> list[str]:
    return sorted(set(pattern.findall(user_input)))


def _contains_phrase(normalized: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, normalized) is not None


def _starts_with_meta_prefix(normalized: str) -> bool:
    return normalized.startswith(_META_PREFIXES)


def _is_meta_request(normalized: str) -> bool:
    if not normalized:
        return False
    if _starts_with_meta_prefix(normalized):
        return True
    if "capabilities" in normalized and any(
        token in normalized for token in ("what", "describe", "explain")
    ):
        return True
    if "tools vs workflows vs agents" in normalized:
        return True
    if "difference between" in normalized:
        return True
    if re.search(r"\b(?:why|how)\s+did\s+(?:you|u)\b", normalized):
        return True
    return False


def _has_explicit_operational_intent(normalized: str) -> bool:
    if not normalized or _is_meta_request(normalized):
        return False
    if normalized.endswith("?") and not any(
        normalized.startswith(prefix)
        for prefix in ("approve ", "deny ", "reply ", "respond ", "kill ", "terminate ", "resume ", "send ", "message ")
    ):
        return False
    return any(re.search(rf"\b{verb}\b", normalized) for verb in _RUN_COMMAND_VERBS)


def _has_agent_command_shape(normalized: str) -> bool:
    return any(pattern.search(normalized) for pattern in _AGENT_COMMAND_SHAPE_PATTERNS)


def _has_runtime_workflow_cue(runtime_grounding: Optional[dict[str, Any]]) -> bool:
    if not isinstance(runtime_grounding, dict):
        return False
    workflows = runtime_grounding.get("workflows")
    return isinstance(workflows, list) and bool(workflows)


def _has_runtime_agent_cue(runtime_grounding: Optional[dict[str, Any]]) -> bool:
    if not isinstance(runtime_grounding, dict):
        return False
    agents = runtime_grounding.get("agents")
    return isinstance(agents, list) and bool(agents)


def _has_runtime_approval_cue(runtime_grounding: Optional[dict[str, Any]]) -> bool:
    if not isinstance(runtime_grounding, dict):
        return False
    approvals = runtime_grounding.get("approvals")
    return isinstance(approvals, list) and bool(approvals)


def _looks_like_workflow_preview_confirmation(normalized: str) -> bool:
    return any(pattern.fullmatch(normalized) for pattern in _WORKFLOW_CONFIRMATION_PATTERNS)


def _looks_like_workflow_preview_cancellation(normalized: str) -> bool:
    return any(pattern.fullmatch(normalized) for pattern in _WORKFLOW_CANCELLATION_PATTERNS)


def _has_conflicting_agent_lifecycle_intent(normalized: str) -> bool:
    return any(pattern.search(normalized) for pattern in _CONFLICTING_AGENT_LIFECYCLE_PATTERNS)


@dataclass(frozen=True)
class RouteAdvice:
    route: str
    required_refs: list[str]
    side_effects_allowed: bool
    recommended_commands: list[str]
    confidence: float
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "required_refs": list(self.required_refs),
            "side_effects_allowed": self.side_effects_allowed,
            "recommended_commands": list(self.recommended_commands),
            "confidence": self.confidence,
            "reason": self.reason,
            "signals": dict(self.signals),
        }


def _advice(
    route: str,
    *,
    required_refs: Optional[list[str]] = None,
    side_effects_allowed: bool,
    recommended_commands: list[str],
    confidence: float,
    reason: str,
    signals: Optional[dict[str, Any]] = None,
) -> RouteAdvice:
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        return RouteAdvice(
            route="ask_clarification",
            required_refs=list(required_refs or []),
            side_effects_allowed=False,
            recommended_commands=["ask_clarification"],
            confidence=confidence,
            reason=reason,
            signals=dict(signals or {}),
        )
    return RouteAdvice(
        route=route,
        required_refs=list(required_refs or []),
        side_effects_allowed=side_effects_allowed,
        recommended_commands=list(recommended_commands),
        confidence=confidence,
        reason=reason,
        signals=dict(signals or {}),
    )


_PERSISTENT_IMPERATIVE_PATTERN_LABELS = (
    "create agent imperative",
    "spawn/make/add/start agent imperative",
    "create owner agent imperative",
    "create persistent agent imperative",
    "create agent to own imperative",
    "have an agent own imperative",
    "delegate domain ownership imperative",
)


def _matched_operational_verbs(normalized: str) -> list[str]:
    return [
        verb
        for verb in _RUN_COMMAND_VERBS
        if re.search(rf"\b{verb}\b", normalized)
    ]


def _matched_inspection_phrases(normalized: str) -> list[str]:
    matched = [
        phrase
        for phrase in _INSPECTION_PHRASES
        if _contains_phrase(normalized, phrase)
    ]
    if re.search(r"\bdid\b.*\bfinish(?:\s+running)?\b", normalized):
        matched.append("did ... finish")
    if re.search(r"\bwhy\b.*\bfail(?:ed)?\b", normalized):
        matched.append("why ... fail")
    if re.search(r"\b(?:show|list|display|view)\b", normalized):
        matched.append("show/list/display/view verb")
    if re.search(r"\bwhat(?:'s| is)\s+running\b", normalized):
        matched.append("what's running")
    if re.search(r"\bwhat\s+workflows?\s+are\s+running\b", normalized):
        matched.append("what workflows are running")
    return list(dict.fromkeys(matched))


def _matched_persistent_agent_patterns(user_input: str, normalized: str) -> list[str]:
    matched = [
        label
        for label, pattern in zip(_PERSISTENT_IMPERATIVE_PATTERN_LABELS, _PERSISTENT_IMPERATIVE_PATTERNS)
        if pattern.search(user_input)
    ]
    matched.extend(
        marker
        for marker in _PERSISTENT_MARKERS
        if marker in normalized
    )
    return list(dict.fromkeys(matched))


def _matched_workflow_patterns(
    user_input: str,
    normalized: str,
    *,
    pending_state: dict | None = None,
) -> list[str]:
    matched: list[str] = []
    pending_mode = str((pending_state or {}).get("mode") or "").strip().lower()
    if pending_mode:
        matched.append(f"pending workflow draft:{pending_mode}")
    if _looks_like_workflow_preview_confirmation(normalized):
        matched.append("workflow preview confirmation")
    if _looks_like_workflow_preview_cancellation(normalized):
        matched.append("workflow preview cancellation")
    matched.extend(
        f"workflow create marker:{marker}"
        for marker in _WORKFLOW_CREATE_MARKERS
        if marker in normalized
    )
    matched.extend(
        f"workflow modify token:{token}"
        for token in _WORKFLOW_MODIFY_TOKENS
        if re.search(rf"\b{re.escape(token)}\b", normalized)
    )
    if _RERUN_VERB_PATTERN.search(normalized):
        matched.append("workflow rerun verb")
    action_words = sorted(set(word.lower() for word in _WORKFLOW_ACTION_WORDS.findall(user_input)))
    if len(action_words) >= 2 and any(joiner in normalized for joiner in (" and ", ",", " then ")):
        matched.append("workflow multi-step action sequence")
    matched.extend(f"workflow action word:{word}" for word in action_words)
    return list(dict.fromkeys(matched))


def _build_route_signals(
    user_input: str,
    normalized: str,
    *,
    runtime_grounding: dict | None = None,
    pending_state: dict | None = None,
    required_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "matched_operational_verbs": _matched_operational_verbs(normalized),
        "matched_inspection_phrases": _matched_inspection_phrases(normalized),
        "matched_persistent_agent_patterns": _matched_persistent_agent_patterns(user_input, normalized),
        "matched_workflow_patterns": _matched_workflow_patterns(
            user_input,
            normalized,
            pending_state=pending_state,
        ),
        "explicit_references": list(required_refs or []),
        "runtime_cues": {
            "has_agents": _has_runtime_agent_cue(runtime_grounding),
            "has_workflows": _has_runtime_workflow_cue(runtime_grounding),
            "has_approvals": _has_runtime_approval_cue(runtime_grounding),
            "pending_workflow_mode": str((pending_state or {}).get("mode") or ""),
        },
    }


def build_route_advice(
    user_input: str,
    runtime_grounding: dict | None = None,
    pending_state: dict | None = None,
) -> RouteAdvice:
    normalized = _normalize_text(user_input)
    workflow_ids = _explicit_ids(_WORKFLOW_ID_PATTERN, user_input)
    task_ids = _explicit_ids(_TASK_ID_PATTERN, user_input)
    message_ids = _explicit_ids(_MESSAGE_ID_PATTERN, user_input)
    agent_ids = _explicit_ids(_AGENT_ID_PATTERN, user_input)
    approval_ids = _explicit_ids(_APPROVAL_ID_PATTERN, user_input)
    required_refs = workflow_ids + task_ids + message_ids + approval_ids + agent_ids
    signals = _build_route_signals(
        user_input,
        normalized,
        runtime_grounding=runtime_grounding,
        pending_state=pending_state,
        required_refs=required_refs,
    )
    persistent_request = (
        not _is_meta_request(normalized)
        and any(pattern.search(user_input) for pattern in _PERSISTENT_IMPERATIVE_PATTERNS)
    )
    persistent_request = persistent_request or (
        not _is_meta_request(normalized)
        and any(marker in normalized for marker in _PERSISTENT_MARKERS)
        and (
            "agent" in normalized
            or "child" in normalized
            or "own" in normalized
            or "responsible" in normalized
            or ("self-evolving" in normalized and "not handled by mr1 directly" in normalized)
            or ("self-evolution" in normalized and "not handled by mr1 directly" in normalized)
        )
    )

    if not normalized:
        return _advice(
            "ask_clarification",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["ask_clarification"],
            confidence=0.20,
            reason="Empty user input does not provide enough routing intent.",
            signals=signals,
        )

    if pending_state:
        pending_mode = str(pending_state.get("mode") or "").strip().lower()
        if pending_mode == "modify":
            return _advice(
                "modify_workflow",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["load_workflow", "author_workflow_modification", "submit_workflow"],
                confidence=0.88,
                reason="A pending workflow modification draft exists, so this turn stays in workflow-modification handling.",
                signals=signals,
            )
        if pending_mode == "create":
            return _advice(
                "create_workflow",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["author_workflow", "submit_workflow"],
                confidence=0.88,
                reason="A pending workflow draft exists, so this turn stays in workflow-authoring handling.",
                signals=signals,
            )

    if _has_conflicting_agent_lifecycle_intent(normalized):
        return _advice(
            "ask_clarification",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["ask_clarification"],
            confidence=0.91,
            reason="The turn combines conflicting agent lifecycle actions by creating agents and immediately destroying them.",
            signals=signals,
        )

    if _looks_like_workflow_preview_confirmation(normalized) or _looks_like_workflow_preview_cancellation(normalized):
        return _advice(
            "ask_clarification",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["ask_clarification"],
            confidence=0.74,
            reason="The turn looks like workflow confirmation or cancellation language, but no pending workflow draft exists.",
            signals=signals,
        )

    if _has_explicit_operational_intent(normalized):
        if message_ids and _REPLY_INTENT_PATTERN.search(user_input):
            return _advice(
                "run_commands",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["inspect_message", "send_message_to_agent"],
                confidence=0.98,
                reason="The turn references an explicit message id and asks for an immediate reply/clarification action.",
                signals=signals,
            )
        if approval_ids or (
            any(re.search(rf"\b{token}\b", normalized) for token in ("approve", "deny"))
            and ("approval" in normalized or "request" in normalized or _has_runtime_approval_cue(runtime_grounding))
        ):
            return _advice(
                "run_commands",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["inspect_approval", "apply_approval_decision"],
                confidence=0.97 if approval_ids else 0.82,
                reason="The turn requests an immediate approval decision rather than an explanation.",
                signals=signals,
            )
        if any(
            re.search(rf"\b{verb}\b", normalized)
            for verb in ("kill", "terminate", "resume", "pause", "rename", "message", "send", "ask", "tell", "stop")
        ) and (
            any(token in normalized for token in ("agent", "agents", "child", "children", "ag-"))
            or _has_agent_command_shape(normalized)
        ):
            return _advice(
                "run_commands",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["manage_agent", "send_message_to_agent"],
                confidence=0.84,
                reason="The turn requests an immediate operational agent command rather than a meta discussion.",
                signals=signals,
            )
        if any(re.search(rf"\b{verb}\b", normalized) for verb in ("delete", "remove", "wipe", "purge", "stop")):
            return _advice(
                "ask_clarification",
                required_refs=required_refs,
                side_effects_allowed=False,
                recommended_commands=["ask_clarification"],
                confidence=0.68,
                reason="The turn is operational and potentially destructive, but the exact runtime target or command form is underspecified.",
                signals=signals,
            )

    has_inspection_intent = any(_contains_phrase(normalized, phrase) for phrase in _INSPECTION_PHRASES)
    has_inspection_intent = has_inspection_intent or bool(
        re.search(r"\bdid\b.*\bfinish(?:\s+running)?\b", normalized)
    )
    has_inspection_intent = has_inspection_intent or bool(
        re.search(r"\bwhy\b.*\bfail(?:ed)?\b", normalized)
    )
    has_inspection_intent = has_inspection_intent or bool(
        re.search(r"\b(?:show|list|display|view)\b", normalized)
    )
    has_inspection_intent = has_inspection_intent or bool(
        re.search(r"\bwhat(?:'s| is)\s+running\b", normalized)
    )
    has_inspection_intent = has_inspection_intent or bool(
        re.search(r"\bwhat\s+workflows?\s+are\s+running\b", normalized)
    )
    if has_inspection_intent and not persistent_request and not _is_meta_request(normalized):
        explicit_state_refs = bool(workflow_ids or task_ids or agent_ids)
        workflow_language = any(
            token in normalized for token in ("workflow", "task", "findings", "results", "status", "running")
        )
        agent_language = any(
            token in normalized
            for token in (
                "agent",
                "child",
                "mr2",
                "blocked",
                "run_status",
                "inbox",
                "message",
            )
        )
        approval_language = "approval" in normalized or "approvals" in normalized
        pronoun_followup = (
            "that workflow" in normalized
            or "this workflow" in normalized
            or "the workflow" in normalized
            or re.search(r"\bof it\b", normalized) is not None
        )
        agent_followup = (
            "that agent" in normalized
            or "this agent" in normalized
            or "the agent" in normalized
            or "that child" in normalized
            or "this child" in normalized
            or "the child" in normalized
        )
        if (
            explicit_state_refs
            or workflow_language
            or (pronoun_followup and _has_runtime_workflow_cue(runtime_grounding))
            or agent_language
            or (agent_followup and _has_runtime_agent_cue(runtime_grounding))
            or approval_language
            or "pending" in normalized
            or "running" in normalized
        ):
            return _advice(
                "inspect_existing_state",
                required_refs=required_refs,
                side_effects_allowed=False,
                recommended_commands=["inspect_workflow", "inspect_task_results", "inspect_agent"],
                confidence=0.97 if explicit_state_refs else 0.80,
                reason="The turn asks for status, findings, or failure analysis of existing runtime state.",
                signals=signals,
            )

    if not _is_meta_request(normalized):
        if persistent_request and any(pattern.search(user_input) for pattern in _PERSISTENT_IMPERATIVE_PATTERNS):
            return _advice(
                "persistent_agent",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["list_agents", "create_persistent_agent", "send_message_to_agent"],
                confidence=0.94,
                reason="The turn requests long-term ownership/delegation to a persistent agent rather than one-shot execution.",
                signals=signals,
            )
        if persistent_request:
            return _advice(
                "persistent_agent",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["list_agents", "create_persistent_agent", "send_message_to_agent"],
                confidence=0.82,
                reason="The turn describes persistent ownership or self-evolution handling outside MR1's direct execution path.",
                signals=signals,
            )

    if workflow_ids and any(re.search(rf"\b{token}\b", normalized) for token in _WORKFLOW_MODIFY_TOKENS):
        return _advice(
            "modify_workflow",
            required_refs=required_refs,
            side_effects_allowed=True,
            recommended_commands=["load_workflow", "author_workflow_modification", "submit_workflow"],
            confidence=0.93,
            reason="The turn references an existing workflow and asks for modification or rerun behavior.",
            signals=signals,
        )

    if _RERUN_VERB_PATTERN.search(normalized):
        return _advice(
            "ask_clarification",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["ask_clarification"],
            confidence=0.68,
            reason="The turn requests a rerun/retry action but does not identify the workflow or task precisely enough to execute safely.",
            signals=signals,
        )

    if not _is_meta_request(normalized):
        if any(marker in normalized for marker in _WORKFLOW_CREATE_MARKERS):
            return _advice(
                "create_workflow",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["author_workflow", "submit_workflow"],
                confidence=0.92,
                reason="The turn explicitly asks for workflow/pipeline execution.",
                signals=signals,
            )
        action_words = _WORKFLOW_ACTION_WORDS.findall(user_input)
        if len(action_words) >= 2 and any(joiner in normalized for joiner in (" and ", ",", " then ")):
            return _advice(
                "create_workflow",
                required_refs=required_refs,
                side_effects_allowed=True,
                recommended_commands=["author_workflow", "submit_workflow"],
                confidence=0.78,
                reason="The turn requests a multi-step executable task that fits workflow authoring.",
                signals=signals,
            )

    if _is_meta_request(normalized):
        return _advice(
            "direct_response",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["answer_directly"],
            confidence=0.96,
            reason="The turn is explanatory or comparative, so MR1 should answer directly without execution.",
            signals=signals,
        )

    if normalized.startswith(("hi", "hello", "hey", "thanks", "thank you")):
        return _advice(
            "direct_response",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["answer_directly"],
            confidence=0.90,
            reason="The turn is conversational rather than an execution request.",
            signals=signals,
        )

    if normalized.endswith("?") and not _has_explicit_operational_intent(normalized):
        return _advice(
            "direct_response",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["answer_directly"],
            confidence=0.82,
            reason="The turn is a direct question without execute-now operational intent.",
            signals=signals,
        )

    if not any(
        token in normalized
        for token in (
            "workflow",
            "task",
            "agent",
            "child",
            "approve",
            "deny",
            "reply",
            "respond",
            "clarify",
            "kill",
            "terminate",
            "resume",
            "message",
            "send",
            "run",
            "execute",
            "pipeline",
            "create",
            "build",
            "modify",
            "update",
            "edit",
            "rerun",
        )
    ):
        return _advice(
            "direct_response",
            required_refs=required_refs,
            side_effects_allowed=False,
            recommended_commands=["answer_directly"],
            confidence=0.78,
            reason="The turn is descriptive or conversational and does not express an execute-now command.",
            signals=signals,
        )

    return _advice(
        "direct_response",
        required_refs=required_refs,
        side_effects_allowed=False,
        recommended_commands=["answer_directly"],
        confidence=0.75,
        reason="No clear operational route detected; defaulting to direct response.",
        signals=signals,
    )
