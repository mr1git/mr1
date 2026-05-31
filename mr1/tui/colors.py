from __future__ import annotations

from dataclasses import dataclass

from mr1.scoped_agents import derive_lifecycle_state

DEPTH_PALETTE = (
    "bright_magenta",
    "bright_red",
    "bright_yellow",
    "bright_green",
    "bright_cyan",
    "bright_blue",
)
ACTIVE_RUN_STATUSES = frozenset({"active", "waiting", "running", "working"})


@dataclass(frozen=True)
class ColorSpec:
    color: str
    dim: bool
    bold: bool

    @property
    def style(self) -> str:
        parts = []
        if self.dim:
            parts.append("dim")
        if self.bold:
            parts.append("bold")
        parts.append(self.color)
        return " ".join(parts)


def depth_color(depth: int) -> str:
    normalized = max(depth - 1, 0)
    return DEPTH_PALETTE[normalized % len(DEPTH_PALETTE)]


def is_terminal_status(status: str | None, run_status: str | None = None) -> bool:
    lifecycle = derive_lifecycle_state(status, run_status)
    return bool(lifecycle["is_terminal"])


def color_for_agent(
    *,
    depth: int,
    status: str | None,
    run_status: str | None,
    selected: bool = False,
) -> ColorSpec:
    lifecycle = derive_lifecycle_state(status, run_status)
    dim = bool(lifecycle["is_terminal"])
    normalized_run_status = (run_status or "").strip().lower()
    bold = selected or (normalized_run_status in ACTIVE_RUN_STATUSES and not dim)
    return ColorSpec(color=depth_color(depth), dim=dim, bold=bold)


def severity_style(severity: str | None) -> str:
    normalized = (severity or "").upper()
    if normalized == "CRITICAL":
        return "bold bright_red"
    if normalized == "ERROR":
        return "bold red"
    if normalized == "WARNING":
        return "bold yellow"
    return "bright_cyan"
