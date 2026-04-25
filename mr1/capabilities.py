"""
Global capability descriptions for tools, watchers, and agents.

Capabilities describe what MR1 can do without changing the execution
model used by tools, watchers, or agent tasks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from mr1.tools import ToolRegistry, default_tool_registry
from mr1.watchers import WatcherRegistry, default_watcher_registry


def _default_agent_capabilities() -> list[dict[str, Any]]:
    return [
        {
            "name": "kazi_agent",
            "type": "agent",
            "description": "LLM-based reasoning and generation task.",
            "inputs": {
                "prompt": "string",
                "inputs": "dataflow inputs",
            },
            "outputs": {
                "result.text": "generated output",
            },
            "examples": [
                {
                    "label": "summarize",
                    "title": "Summarize",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Summarize the provided inputs.",
                }
            ],
            "config_schema": {
                "prompt": {"type": "string", "required": True},
                "inputs": {"type": "list[input_ref]", "required": False, "default": []},
            },
        }
    ]


class CapabilityRegistry:
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        watcher_registry: Optional[WatcherRegistry] = None,
        agent_capabilities: Optional[list[dict[str, Any]]] = None,
    ):
        self._tool_registry = tool_registry or default_tool_registry()
        self._watcher_registry = watcher_registry or default_watcher_registry()
        self._capabilities: dict[str, dict[str, Any]] = {}

        descriptions = []
        descriptions.extend(self._tool_registry.describe_all_tools())
        descriptions.extend(self._watcher_registry.describe_all_watchers())
        descriptions.extend(deepcopy(agent_capabilities or _default_agent_capabilities()))

        for capability in descriptions:
            self._register(capability)

    def _register(self, capability: dict[str, Any]) -> None:
        normalized = {
            "name": capability["name"],
            "type": capability["type"],
            "description": capability["description"],
            "inputs": deepcopy(dict(capability.get("inputs", {}))),
            "outputs": deepcopy(dict(capability.get("outputs", {}))),
            "examples": deepcopy(list(capability.get("examples", []))),
            "config_schema": deepcopy(dict(capability.get("config_schema", {}))),
        }
        name = normalized["name"]
        if name in self._capabilities:
            raise ValueError(f"duplicate capability name '{name}'")
        self._capabilities[name] = normalized

    def list_capabilities(self) -> list[str]:
        return sorted(self._capabilities)

    def describe_capability(self, name: str) -> dict[str, Any]:
        capability = self._capabilities.get(name)
        if capability is None:
            raise ValueError(f"capability not found: {name}")
        return deepcopy(capability)

    def describe_all(self) -> list[dict[str, Any]]:
        return [self.describe_capability(name) for name in self.list_capabilities()]


_DEFAULT_REGISTRY: Optional[CapabilityRegistry] = None


def default_capability_registry() -> CapabilityRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = CapabilityRegistry()
    return _DEFAULT_REGISTRY
