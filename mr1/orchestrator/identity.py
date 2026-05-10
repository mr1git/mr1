"""Orchestrator identity, paths, and timestamp/ID helpers.

Single source for `_PKG_ROOT`, `_AGENTS_DIR`, the per-agent YAML config
paths, the agent-config loader, and the `_now_iso` / `_generate_task_id`
/ `_compact_text` utilities used across MR1, MRn, and the action
handlers. Both `mr1.mr1` and `mr1.mrn_loop` import from here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


_PKG_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_DIR = _PKG_ROOT / "agents"
_MR1_CONFIG_PATH = _AGENTS_DIR / "mr1.yml"
_MRN_CONFIG_PATH = _AGENTS_DIR / "mrn.yml"
_KAZI_CONFIG_PATH = _AGENTS_DIR / "kazi.yml"


def _load_agent_config(path: Path) -> dict:
    """Load an agent YAML definition."""
    with open(path) as f:
        return yaml.safe_load(f)


def _generate_task_id() -> str:
    """Generate a unique, timestamp-prefixed task ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"task-{ts}-{short}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_text(value: Any, *, limit: int = 240) -> str:
    if not isinstance(value, str):
        return "-"
    normalized = " ".join(value.split())
    if not normalized:
        return "-"
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized
