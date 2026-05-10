"""MRn action parsing, validation, and dispatch.

The runner ingests a raw model response, validates it via
`parse_and_validate_action`, then routes it to the appropriate
executor via `dispatch_action`. Each action lives in its own grouped
module (workflow/messaging/report/capability).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .parser import (
    ACTION_DEFAULT_STATUS,
    ALLOWED_MRN_ACTIONS,
    ALLOWED_STATUSES,
    extract_json_object,
    parse_and_validate_action,
)
from .capability_action import execute_call_capability
from .messaging_actions import execute_ask_parent, execute_idle, execute_send_message
from .report_action import execute_write_report
from .workflow_actions import execute_create_workflow, execute_inspect_workflow

if TYPE_CHECKING:
    from mr1.mrn_loop import MRnStepResult, MRnStepRunner


__all__ = [
    "ACTION_DEFAULT_STATUS",
    "ALLOWED_MRN_ACTIONS",
    "ALLOWED_STATUSES",
    "dispatch_action",
    "execute_ask_parent",
    "execute_call_capability",
    "execute_create_workflow",
    "execute_idle",
    "execute_inspect_workflow",
    "execute_send_message",
    "execute_write_report",
    "extract_json_object",
    "parse_and_validate_action",
]


def dispatch_action(
    runner: "MRnStepRunner",
    agent,
    action: dict[str, Any],
    step_call_count: list[int],
    *,
    prompt_artifact_path: Optional[str] = None,
) -> "MRnStepResult":
    name = action["action"]
    if name == "call_capability":
        return execute_call_capability(
            runner, agent, action, step_call_count,
            prompt_artifact_path=prompt_artifact_path,
        )
    if name == "create_workflow":
        return execute_create_workflow(
            runner, agent, action,
            prompt_artifact_path=prompt_artifact_path,
        )
    if name == "inspect_workflow":
        return execute_inspect_workflow(
            runner, agent, action,
            prompt_artifact_path=prompt_artifact_path,
        )
    if name == "write_report":
        return execute_write_report(
            runner, agent, action,
            prompt_artifact_path=prompt_artifact_path,
        )
    if name == "send_message":
        return execute_send_message(
            runner, agent, action,
            prompt_artifact_path=prompt_artifact_path,
        )
    if name == "ask_parent":
        return execute_ask_parent(
            runner, agent, action,
            prompt_artifact_path=prompt_artifact_path,
        )
    return execute_idle(
        runner, agent, action,
        prompt_artifact_path=prompt_artifact_path,
    )
