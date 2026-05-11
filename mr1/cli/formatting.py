"""Small shared formatting helpers for the CLI.

These are domain-agnostic: timestamps, text truncation, table rendering,
task-label lookup, and the description/schema-view helpers shared by
both capabilities and tools sub-commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

from mr1.tools import ToolRegistry
from mr1.workflow_models import Workflow
from mr1.workflow_schema import (
    WorkflowSchemaRegistry,
    default_workflow_schema_registry,
)


def _reject_invalid_flag_combination(args: argparse.Namespace) -> Optional[int]:
    if getattr(args, "example", False) and getattr(args, "brief", False):
        print("error: invalid flag combination", file=sys.stderr)
        return 2
    return None


def _short_ts(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    # "2026-04-20T14:30:05.123456+00:00" → "2026-04-20 14:30:05"
    return iso.replace("T", " ")[:19]


def _compact_text(text: Optional[str], *, limit: int = 120) -> str:
    if not text:
        return "-"
    normalized = " ".join(text.split())
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _render_table(rows: list[tuple[str, ...]], indent: str = "") -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for row in rows:
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        out.append(indent + "  ".join(cells).rstrip())
    return "\n".join(out)


def _label_for_task_id(wf: Workflow, task_id: str) -> Optional[str]:
    for label, tid in wf.label_to_task_id.items():
        if tid == task_id:
            return label
    return None


def _brief_description(description: dict[str, Any]) -> dict[str, str]:
    return {
        "name": description["name"],
        "type": description["type"],
        "description": description["description"],
    }


def _select_description_view(
    description: dict[str, Any],
    *,
    example_only: bool,
    brief: bool,
) -> dict[str, Any]:
    if example_only and brief:
        raise ValueError("invalid flag combination")
    if example_only:
        examples = list(description.get("examples") or [])
        return examples[0] if examples else {}
    if brief:
        return _brief_description(description)
    return description


def _format_description_text(description: dict[str, Any]) -> str:
    lines = [
        f"name:         {description['name']}",
        f"type:         {description['type']}",
        f"description:  {description['description']}",
        "inputs:",
        json.dumps(description.get("inputs", {}), indent=2, sort_keys=True),
        "outputs:",
        json.dumps(description.get("outputs", {}), indent=2, sort_keys=True),
        "config_schema:",
        json.dumps(description.get("config_schema", {}), indent=2, sort_keys=True),
    ]
    if "runtime" in description:
        lines.extend([
            "runtime:",
            json.dumps(description.get("runtime", {}), indent=2, sort_keys=True),
        ])
    if "workflow_task_allowed" in description:
        lines.append(
            f"workflow_task_allowed: {bool(description.get('workflow_task_allowed'))}"
        )
    if "risk_score" in description:
        lines.extend([
            f"risk_score:      {description.get('risk_score')}",
            f"direct_allowed:  {bool(description.get('direct_allowed'))}",
            f"workflow_allowed:{bool(description.get('workflow_allowed'))}",
            f"requires_scope:  {bool(description.get('requires_scope'))}",
            f"is_filesystem:   {bool(description.get('is_filesystem'))}",
            f"is_execution:    {bool(description.get('is_execution'))}",
            f"path_arg_fields: {', '.join(description.get('path_arg_fields') or []) or '-'}",
        ])
    lines.extend([
        "examples:",
        json.dumps(description.get("examples", []), indent=2, sort_keys=True),
    ])
    return "\n".join(lines)


def _describe_schema_section(
    section: Optional[str],
    registry: Optional[WorkflowSchemaRegistry] = None,
) -> dict[str, Any]:
    active_registry = registry or default_workflow_schema_registry()
    if section is None:
        return active_registry.describe_all()
    if section == "workflow":
        return active_registry.describe_workflow()
    if section == "task":
        return active_registry.describe_task()
    if section == "inputs":
        return active_registry.describe_inputs()
    if section == "refs":
        return active_registry.describe_references()
    if section == "task-kinds":
        return active_registry.describe_task_kinds()
    raise ValueError(f"schema section not found: {section}")


def _brief_schema_view(section: Optional[str], description: dict[str, Any]) -> dict[str, Any]:
    if section is None:
        return {
            "workflow": description["workflow"]["summary"],
            "task": description["task"]["summary"],
            "inputs": {
                "summary": description["inputs"]["summary"],
                "item_shape": description["inputs"]["item_shape"],
                "rules": [
                    "inputs must be a list of objects",
                    "each input object must include non-empty name and from",
                    "inputs must NEVER be strings",
                ],
            },
            "refs": description["refs"]["summary"],
            "task-kinds": description["task-kinds"]["summary"],
        }
    if section == "inputs":
        return {
            "summary": description["summary"],
            "item_shape": description["item_shape"],
            "rules": [
                "inputs must be a list of objects",
                "each input object must include non-empty name and from",
                "inputs must NEVER be strings",
                "inputs must reference upstream dependencies or ancestors",
            ],
        }
    keys = ("summary", "required", "shape", "fields", "supported_patterns", "agent", "tool", "watcher")
    return {key: description[key] for key in keys if key in description}


def _format_schema(
    section: Optional[str] = None,
    registry: Optional[WorkflowSchemaRegistry] = None,
    *,
    json_output: bool = False,
    brief: bool = False,
) -> str:
    description = _describe_schema_section(section, registry=registry)
    view = _brief_schema_view(section, description) if brief else description
    return json.dumps(view, indent=2, sort_keys=True)


def _config_shape_for_tool(registry: ToolRegistry, tool_type: str) -> str:
    for tool in registry.list_tools():
        if tool.tool_type == tool_type:
            return tool.config_shape
    return "-"
