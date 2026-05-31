from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Optional

from mr1 import workflow_events as ev
from mr1.capability_policy import CapabilityApprovalStore
from mr1.scoped_agents import PersistentAgentStore, is_agent_terminal
from mr1.workflow_models import Task, TaskStatus, Workflow


_SEMANTIC_AGENT_ID_PATTERN = re.compile(r"\bag-\d{8}T\d{6}-[0-9a-f]{6}\b")
_SEMANTIC_APPROVAL_ID_PATTERN = re.compile(r"\bcap_approval_[A-Za-z0-9]+\b")
_SEMANTIC_WORKFLOW_ID_PATTERN = re.compile(r"\bwf-\d{8}T\d{6}-[0-9a-f]{6}\b")
_SEMANTIC_TASK_ID_PATTERN = re.compile(r"\btk-\d{8}T\d{6}-[0-9a-f]{6}\b")
_SEMANTIC_MISMATCH_SUMMARY = "task completed mechanically but did not satisfy intended postcondition"
_SEMANTIC_CLARIFICATION_MARKERS = (
    "clarify",
    "clarification",
    "which agent",
    "which approval",
    "which task",
    "which workflow",
    "need more info",
    "need more information",
    "please confirm",
    "can you clarify",
)


def normalize_semantic_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def extract_semantic_expectation(
    raw: dict[str, Any],
    *,
    workflow_title: str,
) -> Optional[dict[str, Any]]:
    del workflow_title
    text_parts = [
        str(raw.get("label", "")),
        str(raw.get("title", "")),
        str(raw.get("prompt", "")),
    ]
    raw_text = " ".join(part for part in text_parts if part)
    normalized = normalize_semantic_text(raw_text)
    tool_type = raw.get("tool_type")
    path_value = None
    tool_config = raw.get("tool_config", {})
    tool_config_text = json.dumps(tool_config, sort_keys=True) if isinstance(tool_config, dict) else ""
    target_text = " ".join(part for part in (raw_text, tool_config_text) if part)
    agent_ids = sorted(set(_SEMANTIC_AGENT_ID_PATTERN.findall(target_text)))
    approval_ids = sorted(set(_SEMANTIC_APPROVAL_ID_PATTERN.findall(target_text)))
    workflow_ids = sorted(set(_SEMANTIC_WORKFLOW_ID_PATTERN.findall(target_text)))
    task_ids = sorted(set(_SEMANTIC_TASK_ID_PATTERN.findall(target_text)))
    if isinstance(tool_config, dict):
        candidate_path = tool_config.get("path")
        if isinstance(candidate_path, str) and candidate_path.strip():
            path_value = candidate_path
    command_text = ""
    if isinstance(tool_config, dict):
        argv = tool_config.get("argv")
        if isinstance(argv, list):
            command_text = " ".join(str(item) for item in argv)
        elif isinstance(tool_config.get("command"), str):
            command_text = str(tool_config["command"])

    if "kill" in normalized and "agent" in normalized:
        return {
            "kind": "kill_agents",
            "target_agent_ids": agent_ids,
        }
    if "create" in normalized and "agent" in normalized:
        return {
            "kind": "create_agents",
            "target_agent_ids": agent_ids,
        }
    if "approve" in normalized and ("approval" in normalized or "request" in normalized):
        return {
            "kind": "approve_request",
            "approval_request_ids": approval_ids,
        }
    if "rerun" in normalized and ("task" in normalized or "workflow" in normalized):
        return {
            "kind": "rerun_task",
            "workflow_ids": workflow_ids,
            "task_ids": task_ids,
        }
    if tool_type == "write_file" or (path_value and ("write file" in normalized or "create file" in normalized)):
        return {
            "kind": "write_file",
            "target_path": path_value,
        }
    if tool_type == "file_exists" or (path_value and "verify" in normalized and "file" in normalized):
        return {
            "kind": "verify_file",
            "target_path": path_value,
        }
    if "verify" in normalized and ("completion" in normalized or "complete" in normalized):
        return {
            "kind": "verify_completion",
            "target_path": path_value,
        }
    if tool_type == "shell_command":
        normalized_command = normalize_semantic_text(command_text)
        if any(token in normalized_command for token in ("pytest", "python -m pytest", "npm test", "unittest")):
            return {
                "kind": "run_tests",
                "command": command_text,
            }
    if any(phrase in normalized for phrase in ("run tests", "run pytest", "execute tests")):
        return {
            "kind": "run_tests",
            "command": command_text or raw_text,
        }
    return None


class SemanticValidator:
    def __init__(
        self,
        *,
        workspace_root: Path,
        approval_store: CapabilityApprovalStore,
        scoped_agents: PersistentAgentStore,
    ) -> None:
        self._workspace_root = workspace_root
        self._approval_store = approval_store
        self._scoped_agents = scoped_agents

    def expectation_for_task(
        self,
        workflow: Workflow,
        task: Task,
    ) -> Optional[dict[str, Any]]:
        metadata = workflow.metadata if isinstance(workflow.metadata, dict) else {}
        expectations = metadata.get("semantic_expectations")
        if not isinstance(expectations, dict):
            return None
        expectation = expectations.get(task.task_id)
        if isinstance(expectation, dict):
            return dict(expectation)
        expectation = expectations.get(task.label)
        if isinstance(expectation, dict):
            return dict(expectation)
        return None

    def validation_text(
        self,
        *,
        summary: Optional[str],
        text: Optional[str],
        data: Optional[dict[str, Any]],
    ) -> str:
        parts = []
        if summary:
            parts.append(summary)
        if text:
            parts.append(text)
        if data:
            parts.append(json.dumps(data, sort_keys=True))
        return normalize_semantic_text(" ".join(parts))

    def semantic_target_path(self, raw_path: Optional[str]) -> Optional[Path]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self._workspace_root / path
        return path.resolve(strict=False)

    def mismatch_reason(
        self,
        workflow: Workflow,
        task: Task,
        expectation: dict[str, Any],
        *,
        summary: Optional[str],
        text: Optional[str],
        data: Optional[dict[str, Any]],
    ) -> Optional[str]:
        del workflow, task
        raw_text = " ".join(
            part for part in (
                summary or "",
                text or "",
                json.dumps(data or {}, sort_keys=True),
            )
            if part
        )
        normalized = self.validation_text(summary=summary, text=text, data=data)
        if any(marker in normalized for marker in _SEMANTIC_CLARIFICATION_MARKERS):
            return "task output asked a question or requested clarification instead of completing the intended action"

        kind = expectation.get("kind")
        if kind == "kill_agents":
            target_agent_ids = [
                item for item in expectation.get("target_agent_ids", [])
                if isinstance(item, str) and item
            ]
            if not target_agent_ids:
                target_agent_ids = sorted(set(_SEMANTIC_AGENT_ID_PATTERN.findall(raw_text)))
            if target_agent_ids:
                remaining = [
                    agent_id
                    for agent_id in target_agent_ids
                    if (agent := self._scoped_agents.load_agent(agent_id)) is None or not is_agent_terminal(agent)
                ]
                if remaining:
                    return "target agents not terminated: " + ", ".join(remaining)
            if not any(token in normalized for token in ("kill", "killed", "terminate", "terminated", "skipped")):
                return "task output lacked agent termination evidence"
            return None

        if kind == "create_agents":
            mentioned_agent_ids = sorted(set(_SEMANTIC_AGENT_ID_PATTERN.findall(raw_text)))
            if mentioned_agent_ids:
                missing = [
                    agent_id
                    for agent_id in mentioned_agent_ids
                    if self._scoped_agents.load_agent(agent_id) is None
                ]
                if missing:
                    return "task output referenced agents that do not exist: " + ", ".join(missing)
                return None
            if "created" not in normalized or "agent" not in normalized:
                return "task output lacked created-agent evidence"
            return None

        if kind == "approve_request":
            approval_ids = [
                item for item in expectation.get("approval_request_ids", [])
                if isinstance(item, str) and item
            ]
            if not approval_ids:
                approval_ids = sorted(set(_SEMANTIC_APPROVAL_ID_PATTERN.findall(raw_text)))
            if not approval_ids:
                return "task output lacked approval id evidence"
            not_approved = [
                approval_id
                for approval_id in approval_ids
                if (approval := self._approval_store.load(approval_id)) is None or approval.status != "approved"
            ]
            if not_approved:
                return "approval status did not change to approved: " + ", ".join(not_approved)
            return None

        if kind == "rerun_task":
            if not any(token in normalized for token in ("rerun", "retry", "rescheduled", "scheduled", "running")):
                return "task output lacked rerun evidence"
            return None

        if kind == "write_file":
            target_path = self.semantic_target_path(expectation.get("target_path"))
            if target_path is None or not target_path.exists():
                return f"target file was not created: {expectation.get('target_path') or '-'}"
            return None

        if kind == "verify_file":
            target_path = self.semantic_target_path(expectation.get("target_path"))
            if target_path is None or not target_path.exists():
                return f"verification target does not exist: {expectation.get('target_path') or '-'}"
            return None

        if kind == "verify_completion":
            target_path = self.semantic_target_path(expectation.get("target_path"))
            if target_path is not None and not target_path.exists():
                return f"verification target does not exist: {expectation.get('target_path') or '-'}"
            if target_path is None and not any(token in normalized for token in ("verified", "complete", "completed")):
                return "task output lacked completion verification evidence"
            return None

        if kind == "run_tests":
            command_text = normalize_semantic_text(str(expectation.get("command", "")))
            has_test_context = any(
                token in normalized or token in command_text
                for token in ("pytest", "test", "tests", "collected")
            )
            has_test_outcome = any(
                token in normalized
                for token in ("passed", "failed", "error", "errors", "ok", "success")
            )
            if not has_test_context or not has_test_outcome:
                return "task output lacked test execution/pass-fail evidence"
            return None

        return None

    def apply(
        self,
        workflow: Workflow,
        task: Task,
        *,
        target_status: TaskStatus,
        event_type: str,
        summary: Optional[str],
        text: Optional[str],
        data: Optional[dict[str, Any]],
        result_payload: dict[str, Any],
    ) -> tuple[TaskStatus, str, Optional[str], Optional[str], Optional[str], dict[str, Any]]:
        if target_status is not TaskStatus.SUCCEEDED:
            return target_status, event_type, summary, None, None, result_payload
        expectation = self.expectation_for_task(workflow, task)
        if expectation is None:
            return target_status, event_type, summary, None, None, result_payload
        mismatch_reason = self.mismatch_reason(
            workflow,
            task,
            expectation,
            summary=summary,
            text=text,
            data=data,
        )
        if mismatch_reason is None:
            return target_status, event_type, summary, None, None, result_payload
        payload = dict(result_payload)
        payload["status"] = TaskStatus.FAILED.value
        payload["summary"] = _SEMANTIC_MISMATCH_SUMMARY
        payload["error"] = mismatch_reason
        payload["error_type"] = "semantic_mismatch"
        payload["failure_type"] = "semantic_mismatch"
        payload["semantic_validation"] = {
            "status": "failed",
            "kind": expectation.get("kind"),
            "reason": mismatch_reason,
        }
        return (
            TaskStatus.FAILED,
            ev.TASK_FAILED,
            _SEMANTIC_MISMATCH_SUMMARY,
            mismatch_reason,
            "semantic_mismatch",
            payload,
        )
