from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from mr1.runtime_access import RuntimeAccess
from mr1.scoped_agents import AgentScopeError


_OBSERVE_PATTERN = re.compile(
    r"^\s*\[OBSERVE\]\s*(\{.*\})\s*\[/OBSERVE\]\s*$",
    re.DOTALL,
)

_OBSERVE_OPEN_MARKER = "[OBSERVE]"
_OBSERVE_CLOSE_MARKER = "[/OBSERVE]"

_ALLOWED_CALL_ARGS: dict[str, set[str]] = {
    "list_agents": {"limit", "pinned_agent_ids"},
    "read_agent": {"agent_id"},
    "list_messages": {
        "limit",
        "pinned_message_ids",
        "message_ids",
        "to_agent_id",
        "from_agent_id",
        "status",
        "include_archived",
    },
    "read_message": {"message_id"},
    "list_workflows": {"limit", "pinned_workflow_ids"},
    "read_workflow": {"workflow_id"},
    "list_pending_approvals": {"limit"},
    "list_recent_errors": {"limit"},
    "search_memory": {"query", "limit"},
}

_READ_CALL_NAMES = {"read_agent", "read_message", "read_workflow"}


@dataclass(frozen=True)
class ObservationCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ObservationRequest:
    calls: list[ObservationCall]


@dataclass(frozen=True)
class ObservationParseResult:
    status: str
    request: Optional[ObservationRequest] = None
    error: Optional[str] = None


def parse_observation_request(raw: str) -> ObservationParseResult:
    if _OBSERVE_OPEN_MARKER not in raw and _OBSERVE_CLOSE_MARKER not in raw:
        return ObservationParseResult(status="none")
    if raw.count(_OBSERVE_OPEN_MARKER) != 1 or raw.count(_OBSERVE_CLOSE_MARKER) != 1:
        return ObservationParseResult(
            status="invalid",
            error="observation request must contain exactly one OBSERVE block",
        )
    match = _OBSERVE_PATTERN.fullmatch(raw)
    if match is None:
        return ObservationParseResult(
            status="invalid",
            error="OBSERVE block must occupy the entire response",
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ObservationParseResult(
            status="invalid",
            error="invalid observation JSON",
        )
    request, error = _validate_observation_payload(payload)
    if error is not None:
        return ObservationParseResult(status="invalid", error=error)
    return ObservationParseResult(status="valid", request=request)


def execute_observation_request(
    request: ObservationRequest,
    runtime_access: RuntimeAccess,
    *,
    caller_agent_id: Optional[str] = None,
) -> dict[str, Any]:
    calls = []
    for index, call in enumerate(request.calls):
        payload: dict[str, Any] = {
            "index": index,
            "name": call.name,
            "args": dict(call.args),
        }
        try:
            result = _invoke_runtime_call(
                call,
                runtime_access,
                caller_agent_id=caller_agent_id,
            )
            payload["ok"] = True
            payload["result"] = _to_json_safe(result)
        except (ValueError, AgentScopeError) as exc:
            payload["ok"] = False
            payload["error"] = str(exc)
        except Exception:
            payload["ok"] = False
            payload["error"] = "internal observation error"
        calls.append(payload)
    return {
        "request_valid": True,
        "executed": True,
        "calls": calls,
    }


def invalid_observation_result(error: str) -> dict[str, Any]:
    return {
        "request_valid": False,
        "executed": False,
        "error": error,
        "calls": [],
    }


def format_observation_results_block(payload: dict[str, Any]) -> str:
    return "\n".join([
        "=== OBSERVATION RESULTS ===",
        json.dumps(payload, indent=2, sort_keys=True),
        "=== END OBSERVATION RESULTS ===",
    ])


def _validate_observation_payload(
    payload: Any,
) -> tuple[Optional[ObservationRequest], Optional[str]]:
    if not isinstance(payload, dict):
        return None, "observation request must be a JSON object"
    if set(payload.keys()) != {"calls"}:
        return None, 'observation request must contain only the "calls" field'
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return None, '"calls" must be a list'
    validated_calls: list[ObservationCall] = []
    for index, call in enumerate(calls):
        validated_call, error = _validate_observation_call(call, index=index)
        if error is not None:
            return None, error
        validated_calls.append(validated_call)
    return ObservationRequest(calls=validated_calls), None


def _validate_observation_call(
    payload: Any,
    *,
    index: int,
) -> tuple[Optional[ObservationCall], Optional[str]]:
    if not isinstance(payload, dict):
        return None, f"call {index} must be an object"
    if set(payload.keys()) != {"name", "args"}:
        return None, f'call {index} must contain only "name" and "args"'
    name = payload.get("name")
    args = payload.get("args")
    if not isinstance(name, str) or not name:
        return None, f"call {index} has invalid name"
    if name not in _ALLOWED_CALL_ARGS:
        return None, f"unknown observation call: {name}"
    if not isinstance(args, dict):
        return None, f"call {index} args must be an object"
    extra_args = set(args.keys()) - _ALLOWED_CALL_ARGS[name]
    if extra_args:
        extras = ", ".join(sorted(extra_args))
        return None, f"call {index} has unexpected args: {extras}"
    error = _validate_call_args(name, args, index=index)
    if error is not None:
        return None, error
    return ObservationCall(name=name, args=dict(args)), None


def _validate_call_args(name: str, args: dict[str, Any], *, index: int) -> Optional[str]:
    if name == "list_agents":
        return _validate_args_schema(
            args,
            index=index,
            int_fields={"limit"},
            list_str_fields={"pinned_agent_ids"},
        )
    if name == "read_agent":
        return _require_string_arg(args, "agent_id", index=index)
    if name == "list_messages":
        return _validate_args_schema(
            args,
            index=index,
            int_fields={"limit"},
            str_fields={"to_agent_id", "from_agent_id", "status"},
            bool_fields={"include_archived"},
            list_str_fields={"pinned_message_ids", "message_ids"},
        )
    if name == "read_message":
        return _require_string_arg(args, "message_id", index=index)
    if name == "list_workflows":
        return _validate_args_schema(
            args,
            index=index,
            int_fields={"limit"},
            list_str_fields={"pinned_workflow_ids"},
        )
    if name == "read_workflow":
        return _require_string_arg(args, "workflow_id", index=index)
    if name == "list_pending_approvals":
        return _validate_args_schema(args, index=index, int_fields={"limit"})
    if name == "list_recent_errors":
        return _validate_args_schema(args, index=index, int_fields={"limit"})
    if name == "search_memory":
        error = _require_string_arg(args, "query", index=index)
        if error is not None:
            return error
        if "limit" in args:
            if not isinstance(args["limit"], int):
                return f"call {index} arg limit must be an integer"
            if args["limit"] < 1 or args["limit"] > 10:
                return f"call {index} arg limit must be between 1 and 10"
        return None
    return f"unknown observation call: {name}"


def _validate_args_schema(
    args: dict[str, Any],
    *,
    index: int,
    str_fields: set[str] | None = None,
    int_fields: set[str] | None = None,
    bool_fields: set[str] | None = None,
    list_str_fields: set[str] | None = None,
) -> Optional[str]:
    for field in str_fields or set():
        if field in args and not isinstance(args[field], str):
            return f"call {index} arg {field} must be a string"
    for field in int_fields or set():
        if field in args and not isinstance(args[field], int):
            return f"call {index} arg {field} must be an integer"
    for field in bool_fields or set():
        if field in args and not isinstance(args[field], bool):
            return f"call {index} arg {field} must be a boolean"
    for field in list_str_fields or set():
        if field not in args:
            continue
        value = args[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return f"call {index} arg {field} must be a list of strings"
    return None


def _require_string_arg(args: dict[str, Any], key: str, *, index: int) -> Optional[str]:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        return f"call {index} arg {key} must be a non-empty string"
    return None


def _invoke_runtime_call(
    call: ObservationCall,
    runtime_access: RuntimeAccess,
    *,
    caller_agent_id: Optional[str],
) -> Any:
    method = getattr(runtime_access, call.name)
    kwargs = dict(call.args)
    kwargs["caller_agent_id"] = caller_agent_id
    if call.name in _READ_CALL_NAMES:
        kwargs["full"] = True
    return method(**kwargs)


def _to_json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value))
