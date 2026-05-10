"""Text formatting helpers shared by action handlers and the runner."""

from __future__ import annotations

import json
from typing import Any


_MESSAGE_BODY_LIMIT = 4096
_MESSAGE_BODY_TRUNCATION_SUFFIX = "...[truncated, use message_id for full]"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _compact(text: Any, *, limit: int = 240) -> str:
    if not isinstance(text, str):
        return "-"
    normalized = " ".join(text.split())
    if not normalized:
        return "-"
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _truncate_message_body(text: str, *, limit: int = _MESSAGE_BODY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_MESSAGE_BODY_TRUNCATION_SUFFIX))
    return text[:keep] + _MESSAGE_BODY_TRUNCATION_SUFFIX
