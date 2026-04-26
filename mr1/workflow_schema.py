"""
Deterministic workflow schema descriptions for authoring and inspection.

This module is metadata-only. It does not validate, execute, or compile
workflows. It only describes how workflow JSON must be written.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional


def _default_sections() -> dict[str, dict[str, Any]]:
    return {
        "workflow": {
            "summary": "Top-level workflow JSON object.",
            "type": "object",
            "required": ["title", "tasks"],
            "shape": {
                "title": "string",
                "tasks": ["Task"],
            },
            "rules": [
                "title is required and must be a non-empty string",
                "tasks is required and must be a non-empty list",
                "task labels must be unique within the workflow",
            ],
        },
        "task": {
            "summary": "Common task object fields shared by agent, tool, and watcher tasks.",
            "type": "object",
            "required": ["label", "task_kind"],
            "fields": {
                "label": {
                    "type": "string",
                    "required": True,
                    "description": "Unique task label within the workflow.",
                },
                "title": {
                    "type": "string",
                    "required": False,
                    "description": "Optional human-readable task title.",
                },
                "task_kind": {
                    "type": "enum",
                    "required": True,
                    "values": ["agent", "tool", "watcher"],
                },
                "depends_on": {
                    "type": "list[string]",
                    "required": False,
                    "description": "Optional list of upstream task labels that must complete first.",
                },
                "dependency_policy": {
                    "type": "enum",
                    "required": False,
                    "values": ["all_succeeded", "any_succeeded"],
                    "default": "all_succeeded",
                    "description": "Controls how dependency outcomes are interpreted at joins.",
                },
                "run_if": {
                    "type": "Condition",
                    "required": False,
                    "description": "Optional deterministic condition evaluated after the dependency gate passes.",
                },
                "inputs": {
                    "type": "list[InputSpec]",
                    "required": False,
                    "description": "Optional dataflow bindings that pass upstream outputs into this task.",
                },
            },
            "dataflow_rules": [
                "depends_on controls execution order",
                "dependency_policy controls how dependency outcomes unlock joins",
                "inputs controls data passing",
                "do not inline upstream outputs into prompts",
                "pass upstream outputs using inputs references",
                "downstream tasks may receive upstream status, condition_result, and skip_reason through inputs references",
            ],
            "status_values": [
                "created",
                "waiting",
                "ready",
                "running",
                "succeeded",
                "skipped",
                "failed",
                "timed_out",
                "cancelled",
                "blocked",
            ],
        },
        "inputs": {
            "summary": "InputSpec list used to pass upstream task outputs into downstream tasks.",
            "type": "list[InputSpec]",
            "item_type": "object",
            "item_shape": {
                "name": "string",
                "from": "<label>.<reference>",
            },
            "rules": [
                "inputs must be a list",
                "each input must be a JSON object",
                "each input object must contain non-empty 'name'",
                "each input object must contain non-empty 'from'",
                "inputs must NEVER be strings",
                "inputs must reference upstream dependencies or ancestors",
                "branch-aware workflows may inject status, condition_result, and skip_reason as normal input objects",
            ],
            "valid_examples": [
                [
                    {
                        "name": "notes",
                        "from": "read_notes.result.text",
                    }
                ],
                [
                    {
                        "name": "success_path_status",
                        "from": "success_path.status",
                    },
                    {
                        "name": "success_path_condition",
                        "from": "success_path.condition_result",
                    },
                    {
                        "name": "success_path_skip_reason",
                        "from": "success_path.skip_reason",
                    },
                ]
            ],
            "invalid_examples": [
                [
                    "read_notes.result.text",
                ]
            ],
        },
        "refs": {
            "summary": "Supported upstream reference syntax for InputSpec.from.",
            "supported_patterns": [
                "<label>.result",
                "<label>.result.summary",
                "<label>.result.text",
                "<label>.result.data",
                "<label>.result.data.<key>[.<nested_key>...]",
                "<label>.result.metrics",
                "<label>.result.metrics.<key>[.<nested_key>...]",
                "<label>.stdout",
                "<label>.stderr",
                "<label>.artifact.<artifact_name>",
                "<label>.status",
                "<label>.condition_result",
                "<label>.skip_reason",
            ],
            "branch_context_fields": [
                {
                    "reference": "<label>.status",
                    "type": "string",
                    "description": "Terminal or live upstream task status such as succeeded, skipped, or failed.",
                },
                {
                    "reference": "<label>.condition_result",
                    "type": "object",
                    "description": "Deterministic run_if evaluation metadata for conditional branch tasks.",
                },
                {
                    "reference": "<label>.skip_reason",
                    "type": "string",
                    "description": "Skip explanation for an upstream branch task when it did not run.",
                },
            ],
        },
        "conditions": {
            "summary": "Deterministic task conditions used by run_if.",
            "type": "object",
            "required": ["ref", "op"],
            "fields": {
                "ref": "<label>.<reference>",
                "op": [
                    "eq",
                    "ne",
                    "contains",
                    "exists",
                    "missing",
                    "gt",
                    "gte",
                    "lt",
                    "lte",
                    "truthy",
                    "falsy",
                ],
                "value": "required for eq/ne/contains/gt/gte/lt/lte",
            },
            "rules": [
                "run_if must be a JSON object",
                "run_if.ref must use workflow reference syntax",
                "run_if source label must be an upstream dependency or ancestor",
                "exists/missing/truthy/falsy do not require value",
            ],
            "examples": [
                {
                    "label": "success_path",
                    "depends_on": ["check"],
                    "run_if": {
                        "ref": "check.result.data.exit_code",
                        "op": "eq",
                        "value": 0,
                    },
                },
                {
                    "label": "final",
                    "depends_on": ["success_path", "failure_path"],
                    "dependency_policy": "any_succeeded",
                },
            ],
        },
        "task-kinds": {
            "summary": "Task-kind specific requirements and unused fields.",
            "agent": {
                "task_kind": "agent",
                "required_fields": {
                    "agent_type": "kazi",
                    "prompt": "string",
                },
                "optional_fields": {
                    "inputs": "list[InputSpec]",
                    "depends_on": "list[string]",
                    "dependency_policy": "\"all_succeeded\" | \"any_succeeded\"",
                    "run_if": "Condition",
                    "title": "string",
                },
                "unused_fields": ["tool_type", "tool_config", "watcher_type", "watch_config"],
            },
            "tool": {
                "task_kind": "tool",
                "required_fields": {
                    "tool_type": "string",
                    "tool_config": "object",
                },
                "optional_fields": {
                    "inputs": "list[InputSpec]",
                    "depends_on": "list[string]",
                    "dependency_policy": "\"all_succeeded\" | \"any_succeeded\"",
                    "run_if": "Condition",
                    "title": "string",
                },
                "unused_fields": ["prompt", "agent_type"],
            },
            "watcher": {
                "task_kind": "watcher",
                "required_fields": {
                    "watcher_type": "string",
                    "watch_config": "object",
                },
                "optional_fields": {
                    "inputs": "list[InputSpec]",
                    "depends_on": "list[string]",
                    "dependency_policy": "\"all_succeeded\" | \"any_succeeded\"",
                    "run_if": "Condition",
                    "title": "string",
                },
                "unused_fields": ["prompt", "agent_type"],
            },
        },
    }


class WorkflowSchemaRegistry:
    def __init__(self, sections: Optional[dict[str, dict[str, Any]]] = None):
        self._sections = deepcopy(sections or _default_sections())

    def describe_workflow(self) -> dict[str, Any]:
        return deepcopy(self._sections["workflow"])

    def describe_task(self) -> dict[str, Any]:
        return deepcopy(self._sections["task"])

    def describe_inputs(self) -> dict[str, Any]:
        return deepcopy(self._sections["inputs"])

    def describe_references(self) -> dict[str, Any]:
        return deepcopy(self._sections["refs"])

    def describe_task_kinds(self) -> dict[str, Any]:
        return deepcopy(self._sections["task-kinds"])

    def describe_conditions(self) -> dict[str, Any]:
        return deepcopy(self._sections["conditions"])

    def describe_all(self) -> dict[str, dict[str, Any]]:
        return {
            "workflow": self.describe_workflow(),
            "task": self.describe_task(),
            "inputs": self.describe_inputs(),
            "refs": self.describe_references(),
            "conditions": self.describe_conditions(),
            "task-kinds": self.describe_task_kinds(),
        }


_DEFAULT_REGISTRY: Optional[WorkflowSchemaRegistry] = None


def default_workflow_schema_registry() -> WorkflowSchemaRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = WorkflowSchemaRegistry()
    return _DEFAULT_REGISTRY
