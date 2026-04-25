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
                "inputs": {
                    "type": "list[InputSpec]",
                    "required": False,
                    "description": "Optional dataflow bindings that pass upstream outputs into this task.",
                },
            },
            "dataflow_rules": [
                "depends_on controls execution order",
                "inputs controls data passing",
                "do not inline upstream outputs into prompts",
                "pass upstream outputs using inputs references",
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
            ],
            "valid_examples": [
                [
                    {
                        "name": "notes",
                        "from": "read_notes.result.text",
                    }
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

    def describe_all(self) -> dict[str, dict[str, Any]]:
        return {
            "workflow": self.describe_workflow(),
            "task": self.describe_task(),
            "inputs": self.describe_inputs(),
            "refs": self.describe_references(),
            "task-kinds": self.describe_task_kinds(),
        }


_DEFAULT_REGISTRY: Optional[WorkflowSchemaRegistry] = None


def default_workflow_schema_registry() -> WorkflowSchemaRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = WorkflowSchemaRegistry()
    return _DEFAULT_REGISTRY
