"""Read-only tool policy for Claude brain-plane subprocesses."""

from __future__ import annotations

from typing import Iterable


READ_ONLY_BRAIN_TOOLS = frozenset({
    "Read",
    "Glob",
    "Grep",
})


def governed_brain_tools(configured_tools: Iterable[str] | None) -> list[str]:
    """Return the explicit read-only brain tools allowed under MR1 governance."""
    filtered: list[str] = []
    for tool in configured_tools or []:
        if tool in READ_ONLY_BRAIN_TOOLS and tool not in filtered:
            filtered.append(tool)
    return filtered
