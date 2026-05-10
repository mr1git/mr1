"""
Background trace emitter for MR1 chat sessions.

Tails the event JSONL file and prints compact, readable trace lines into the
chat stream so testing doesn't feel silent while agents run autonomously.

Only events in TRACE_EVENT_TYPES are emitted; everything else is ignored.
Historical events present at startup are skipped — only new events are shown.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

_POLL_INTERVAL = 0.25  # seconds
_MAX_SUMMARY_LEN = 80

TRACE_EVENT_TYPES: frozenset[str] = frozenset([
    "mrn_step_started",
    "mrn_step_completed",
    "mrn_reported",
    "capability_invoked",
    "capability_completed",
    "capability_denied",
])


def _truncate(text: str, max_len: int = _MAX_SUMMARY_LEN) -> str:
    if len(text) > max_len:
        return text[:max_len] + "[...]"
    return text


def format_trace_line(event: dict) -> str:
    """Return a compact one-liner for display in the chat stream."""
    etype = event.get("event_type", "")
    actor = event.get("actor_id") or event.get("target_id") or "?"
    status = event.get("status", "")
    meta = event.get("metadata") or {}

    if etype == "mrn_step_started":
        iteration = meta.get("iteration", "?")
        return f"system: agent {actor} step {iteration} started"

    if etype == "mrn_step_completed":
        action = meta.get("action") or "?"
        return f"system: agent {actor} {action} -> {status}"

    if etype == "mrn_reported":
        return f"{actor}: report written"

    if etype.startswith("capability_"):
        summary = event.get("summary") or ""
        label = etype.replace("capability_", "")
        return f"system: capability {label} {actor} -> {_truncate(summary) if summary else status}"

    summary = event.get("summary") or ""
    return f"system: {etype} {actor} -> {_truncate(summary) if summary else status}"


class TraceEmitter:
    """Tails the event JSONL on a daemon thread and prints trace events."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self,
        event_log_path: Path,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        path = Path(event_log_path)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tail,
            args=(path, print_fn),
            daemon=True,
            name="trace-emitter",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _tail(self, path: Path, print_fn: Callable[[str], None]) -> None:
        # Seek to the current end of the file so we skip historical events.
        offset = path.stat().st_size if path.exists() else 0

        while not self._stop.is_set():
            try:
                if not path.exists():
                    self._stop.wait(_POLL_INTERVAL)
                    continue

                current_size = path.stat().st_size
                if current_size <= offset:
                    self._stop.wait(_POLL_INTERVAL)
                    continue

                with open(path, "r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for raw_line in handle:
                        stripped = raw_line.strip()
                        if not stripped:
                            continue
                        try:
                            event = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        if event.get("event_type") in TRACE_EVENT_TYPES:
                            try:
                                print_fn(format_trace_line(event))
                            except Exception:
                                pass
                    offset = handle.tell()

            except OSError:
                pass

            self._stop.wait(_POLL_INTERVAL)
