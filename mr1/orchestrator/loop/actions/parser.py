"""MRn action JSON parser and validator.

Owns the canonical `ALLOWED_MRN_ACTIONS` / `ALLOWED_STATUSES` /
`ACTION_DEFAULT_STATUS` constants and the strict per-action validation
that turns a raw model response into a normalized action dict.
"""

from __future__ import annotations

import json
from typing import Any

from mr1.messages import ALLOWED_MESSAGE_KINDS, normalize_message_kind


ALLOWED_MRN_ACTIONS = frozenset({
    "create_workflow",
    "inspect_workflow",
    "write_report",
    "send_message",
    "ask_parent",
    "idle",
    "call_capability",
})

ALLOWED_STATUSES = frozenset({
    "idle",
    "working",
    "waiting",
    "reporting",
    "blocked",
    "terminated",
})

ACTION_DEFAULT_STATUS = {
    "create_workflow": "working",
    "inspect_workflow": "working",
    "write_report": "reporting",
    "send_message": "waiting",
    "ask_parent": "waiting",
    "idle": "idle",
    "call_capability": "working",
}


def extract_json_object(text: str) -> dict[str, Any]:
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


def parse_and_validate_action(raw: str) -> dict[str, Any]:
    action = extract_json_object(raw)
    action_name = action.get("action")
    if action_name not in ALLOWED_MRN_ACTIONS:
        raise ValueError(
            "field 'action' must be one of: create_workflow, inspect_workflow, "
            "write_report, send_message, ask_parent, idle, call_capability"
        )
    reason = action.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("field 'reason' must be a non-empty string")
    next_status = action.get("next_status", ACTION_DEFAULT_STATUS[action_name])
    if next_status not in ALLOWED_STATUSES:
        raise ValueError(
            "field 'next_status' must be one of: idle, working, waiting, "
            "reporting, blocked, terminated"
        )

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
        normalized["message_kind"] = normalize_message_kind(normalized["message_kind"])
        if normalized["message_kind"] not in ALLOWED_MESSAGE_KINDS:
            allowed = ", ".join(sorted(ALLOWED_MESSAGE_KINDS))
            raise ValueError(f"send_message message_kind must be one of: {allowed}")
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
