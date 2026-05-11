"""Inspection / summary helpers for the MR1 root orchestrator.

Detects user intents to inspect workflows/tasks ("did it finish?", "show
findings"), and formats human-readable summary text for workflows,
tasks, and findings. Free functions take the `MR1` instance (`root`) as
their first parameter; `MR1` retains thin wrapper methods.

Imports module-level constants and helpers from `mr1.orchestrator.root`;
this works because the import line for this module is placed *after*
the constants block in `root.py`.
"""

import re
from typing import Any, Optional

from mr1.workflow_models import TaskStatus

from mr1.orchestrator.root import (
    _AGENT_REFERENCE_ALIASES,
    _FINDINGS_PREFERRED_LABEL_TOKENS,
    _WORKFLOW_AUTHORING_SUPPRESSION_PHRASES,
    _WORKFLOW_FINDINGS_INTENT_PHRASES,
    _WORKFLOW_INSPECTION_INTENT_PHRASES,
    _WORKFLOW_REFERENCE_ALIASES,
    _compact_text,
    _has_workflow_pronoun_reference,
    _normalize_routing_text,
)


def is_runtime_inspection_request(
    root,
    user_input: str,
    resolved_references: dict[str, Any],
) -> bool:
    normalized = _normalize_routing_text(user_input)
    if not normalized:
        return False
    has_intent = any(phrase in normalized for phrase in _WORKFLOW_INSPECTION_INTENT_PHRASES)
    has_intent = has_intent or bool(
        re.search(r"\bdid\b.*\bfinish(?:\s+running)?\b", normalized)
    )
    has_intent = has_intent or bool(
        re.search(r"\bwhy\b.*\bfail(?:ed)?\b", normalized)
    )
    has_intent = has_intent or bool(
        re.search(r"\bwhy\b.*\b(blocked|stuck|stop(?:ped)?)\b", normalized)
    )
    has_intent = has_intent or bool(
        re.search(r"\bwhy did\b", normalized)
    )
    suppresses_authoring = any(
        phrase in normalized for phrase in _WORKFLOW_AUTHORING_SUPPRESSION_PHRASES
    )
    if not has_intent and not suppresses_authoring:
        return False
    return bool(
        root._explicit_workflow_ids(user_input, resolved_references)
        or root._explicit_task_ids(user_input)
        or root._explicit_agent_ids(user_input, resolved_references)
        or any(alias in normalized for alias in _WORKFLOW_REFERENCE_ALIASES)
        or any(alias in normalized for alias in _AGENT_REFERENCE_ALIASES)
        or (_has_workflow_pronoun_reference(normalized) and root._state.last_referenced_workflow_id)
        or ("that agent" in normalized and (root._state.last_referenced_agent_id or root._state.last_created_agent_id))
    )

def is_workflow_findings_request(root, user_input: str) -> bool:
    normalized = _normalize_routing_text(user_input)
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _WORKFLOW_FINDINGS_INTENT_PHRASES)

def approval_request_id_from_result(root, payload: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    direct_value = payload.get("approval_request_id")
    if isinstance(direct_value, str) and direct_value:
        return direct_value
    data = payload.get("data")
    if isinstance(data, dict):
        nested_value = data.get("approval_request_id")
        if isinstance(nested_value, str) and nested_value:
            return nested_value
    return None

def task_summary_text(
    root,
    workflow_id: str,
    task_id: str,
) -> str:
    workflow = root._workflow_store.load_workflow(workflow_id)
    if workflow is None:
        return f"workflow not found: {workflow_id}"
    task = workflow.tasks.get(task_id)
    if task is None:
        return f"task not found: {task_id}"
    result_payload = root._workflow_store.read_result(workflow_id, task_id)
    output = root._workflow_store.load_task_output(workflow_id, task_id)
    approval_request_id = root._approval_request_id_from_result(result_payload)
    failure_type = None
    error_text = None
    summary_text = None
    if isinstance(result_payload, dict):
        failure_type = result_payload.get("failure_type") or result_payload.get("error_type")
        error_text = result_payload.get("error") or task.last_error
        summary_text = result_payload.get("summary")
    if output is not None:
        summary_text = output.summary or summary_text
    summary_text = summary_text or task.result_summary
    root._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)

    lines = [
        f"task_id:   {task.task_id}",
        f"label:     {task.label}",
        f"title:     {task.title}",
        f"workflow:  {workflow.workflow_id} ({workflow.title})",
        f"status:    {task.status.value}",
    ]
    if failure_type == "semantic_mismatch" or task.last_error_type == "semantic_mismatch":
        lines.append("semantic_validation: failed")
    if failure_type:
        lines.append(f"failure:   {failure_type}")
    if error_text:
        lines.append(f"error:     {_compact_text(error_text)}")
    if summary_text:
        lines.append(f"summary:   {_compact_text(summary_text)}")
    if approval_request_id:
        lines.append(f"approval:  {approval_request_id}")
    if task.log_stdout_path:
        lines.append(f"stdout:    {task.log_stdout_path}")
    if task.log_stderr_path:
        lines.append(f"stderr:    {task.log_stderr_path}")
    if task.result_path:
        lines.append(f"result:    {task.result_path}")
    if task.output_path:
        lines.append(f"output:    {task.output_path}")
    suggestions = []
    if task.status in {
        TaskStatus.FAILED,
        TaskStatus.TIMED_OUT,
        TaskStatus.CANCELLED,
        TaskStatus.BLOCKED,
    }:
        suggestions.append(f"/workflow rerun {workflow.workflow_id} {task.task_id}")
    if approval_request_id:
        suggestions.append(f"/approvals show {approval_request_id}")
    if suggestions:
        lines.extend(["", "suggested_next:"])
        lines.extend(f"- {item}" for item in suggestions)
    return "\n".join(lines)

def workflow_summary_text(root, workflow_id: str) -> str:
    workflow = root._workflow_store.load_workflow(workflow_id)
    if workflow is None:
        return f"workflow not found: {workflow_id}"
    root._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)
    task_lines: list[str] = []
    result_lines: list[str] = []
    failed_lines: list[str] = []
    approval_lines: list[str] = []
    semantic_lines: list[str] = []
    first_retry_task_id: Optional[str] = None
    first_approval_request_id: Optional[str] = None

    for label, task_id in workflow.label_to_task_id.items():
        task = workflow.tasks.get(task_id)
        if task is None:
            continue
        result_payload = root._workflow_store.read_result(workflow.workflow_id, task.task_id)
        output = root._workflow_store.load_task_output(workflow.workflow_id, task.task_id)
        summary_text = None
        if output is not None:
            summary_text = output.summary
        if summary_text is None and isinstance(result_payload, dict):
            summary_text = result_payload.get("summary")
        summary_text = summary_text or task.result_summary or task.last_error or task.blocked_reason
        task_line = f"- {label} ({task.task_id}): {task.status.value}"
        if summary_text:
            task_line += f" | {_compact_text(summary_text)}"
        task_lines.append(task_line)

        if task.is_terminal() and summary_text:
            result_lines.append(f"- {label}: {_compact_text(summary_text)}")

        if task.status in {
            TaskStatus.FAILED,
            TaskStatus.TIMED_OUT,
            TaskStatus.CANCELLED,
            TaskStatus.BLOCKED,
        }:
            failure_text = None
            if isinstance(result_payload, dict):
                failure_text = result_payload.get("failure_type") or result_payload.get("error_type")
            failure_text = failure_text or task.last_error_type or task.blocked_reason or task.last_error
            failed_lines.append(
                f"- {label} ({task.task_id}): {task.status.value}"
                + (f" | {_compact_text(failure_text)}" if failure_text else "")
            )
            if failure_text == "semantic_mismatch" or task.last_error_type == "semantic_mismatch":
                semantic_reason = task.last_error or summary_text or "task completed mechanically but did not satisfy intended postcondition"
                semantic_lines.append(
                    f"- {label} ({task.task_id}): semantic_mismatch | {_compact_text(semantic_reason)}"
                )
            if first_retry_task_id is None:
                first_retry_task_id = task.task_id

        approval_request_id = root._approval_request_id_from_result(result_payload)
        if approval_request_id:
            approval_reason = None
            if isinstance(result_payload, dict):
                decision_payload = result_payload.get("data", {}).get("decision")
                if isinstance(decision_payload, dict):
                    approval_reason = decision_payload.get("reason")
            approval_lines.append(
                f"- {label} ({task.task_id}): {approval_request_id}"
                + (f" | {_compact_text(approval_reason)}" if approval_reason else "")
            )
            if first_approval_request_id is None:
                first_approval_request_id = approval_request_id

    lines = [
        f"workflow_id: {workflow.workflow_id}",
        f"title:       {workflow.title}",
        f"status:      {workflow.status.value}",
        f"tasks:       {len(workflow.tasks)}",
    ]
    if workflow.finished_at:
        lines.append(f"finished:    {workflow.finished_at}")
    if semantic_lines:
        lines.extend(["", "semantic_validation: FAILED"])
        lines.extend(semantic_lines)
    lines.extend(["", "task_statuses:"])
    lines.extend(task_lines or ["- none"])
    if failed_lines:
        lines.extend(["", "failed_or_blocked:"])
        lines.extend(failed_lines)
    if result_lines:
        lines.extend(["", "result_summaries:"])
        lines.extend(result_lines[:8])
    if approval_lines:
        lines.extend(["", "approval_blockers:"])
        lines.extend(approval_lines)
    suggestions = [f"/workflow {workflow.workflow_id}", f"/events {workflow.workflow_id}"]
    if first_retry_task_id:
        suggestions.append(f"/workflow rerun {workflow.workflow_id} {first_retry_task_id}")
    if first_approval_request_id:
        suggestions.append(f"/approvals show {first_approval_request_id}")
    lines.extend(["", "suggested_next:"])
    lines.extend(f"- {item}" for item in suggestions)
    return "\n".join(lines)

def task_full_result_text(root, workflow_id: str, task_id: str) -> Optional[str]:
    output = root._workflow_store.load_task_output(workflow_id, task_id)
    if output is not None and output.text.strip():
        return output.text.strip()
    result_payload = root._workflow_store.read_result(workflow_id, task_id)
    if isinstance(result_payload, dict):
        text_value = result_payload.get("text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()
        body_value = result_payload.get("body")
        if isinstance(body_value, str) and body_value.strip():
            return body_value.strip()
    return None

def findings_candidate_sort_key(
    root,
    workflow_id: str,
    task,
) -> tuple[int, int, str]:
    label_text = f"{task.label} {task.title}".lower()
    priority = 1
    for index, token in enumerate(_FINDINGS_PREFERRED_LABEL_TOKENS):
        if token in label_text:
            priority = 0
            return (priority, index, task.task_id)
    has_full_text = 0 if root._task_full_result_text(workflow_id, task.task_id) else 1
    return (priority, has_full_text, task.task_id)

def select_workflow_findings_task(root, workflow) -> Optional[Any]:
    succeeded_terminal = [
        task
        for task in workflow.tasks.values()
        if task.status == TaskStatus.SUCCEEDED and task.is_terminal()
    ]
    if not succeeded_terminal:
        return None
    candidates = sorted(
        succeeded_terminal,
        key=lambda task: root._findings_candidate_sort_key(workflow.workflow_id, task),
    )
    return candidates[0] if candidates else None

def workflow_findings_text(root, workflow_id: str) -> str:
    workflow = root._workflow_store.load_workflow(workflow_id)
    if workflow is None:
        return f"workflow not found: {workflow_id}"
    root._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)
    semantic_failures = [
        task
        for task in workflow.tasks.values()
        if task.last_error_type == "semantic_mismatch"
    ]
    if semantic_failures:
        lines = [
            "semantic_validation: FAILED",
            "This workflow completed mechanically but did not satisfy its intended postcondition.",
            "",
            f"workflow_id: {workflow.workflow_id}",
            f"title:       {workflow.title}",
            f"status:      {workflow.status.value}",
        ]
        if workflow.finished_at:
            lines.append(f"finished:    {workflow.finished_at}")
        lines.extend(["", "semantic_mismatches:"])
        for task in semantic_failures:
            lines.append(
                f"- {task.label} ({task.task_id}): {_compact_text(task.last_error or task.result_summary or 'semantic_mismatch')}"
            )
        return "\n".join(lines)
    findings_task = root._select_workflow_findings_task(workflow)
    findings_text = None
    findings_summary = None
    findings_label = None
    if findings_task is not None:
        findings_text = root._task_full_result_text(workflow.workflow_id, findings_task.task_id)
        findings_summary = findings_task.result_summary
        findings_label = findings_task.label

    lines: list[str] = []
    if findings_text:
        lines.extend([
            "findings:",
            findings_text,
        ])
    elif findings_summary:
        lines.extend([
            "findings:",
            findings_summary,
        ])
    else:
        lines.append("findings: No final findings text is available for this workflow yet.")

    status_lines = [
        "",
        f"workflow_id: {workflow.workflow_id}",
        f"title:       {workflow.title}",
        f"status:      {workflow.status.value}",
    ]
    if findings_task is not None:
        status_lines.append(
            f"findings_task: {findings_label} ({findings_task.task_id})"
        )
    if workflow.finished_at:
        status_lines.append(f"finished:    {workflow.finished_at}")
    lines.extend(status_lines)
    return "\n".join(lines)

