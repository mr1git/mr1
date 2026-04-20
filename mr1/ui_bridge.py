"""
Structured JSON bridge between the Ink UI and the MR1 orchestrator.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, TextIO

from mr1.mr1 import MR1


@dataclass
class BridgeMessage:
    type: str
    payload: dict[str, Any]


class JsonEmitter:
    def __init__(self, stream: Optional[TextIO] = None):
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()

    def emit(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(json.dumps(payload) + "\n")
            self._stream.flush()


class BridgeSession:
    def __init__(
        self,
        mr1_instance: Optional[MR1] = None,
        emitter: Optional[JsonEmitter] = None,
        snapshot_interval_s: float = 0.35,
    ):
        self._emitter = emitter or JsonEmitter()
        self._mr1 = mr1_instance or MR1(event_sink=self._handle_event)
        self._snapshot_interval_s = snapshot_interval_s
        self._stop = threading.Event()
        self._ticker: Optional[threading.Thread] = None
        self._history_cutoff = datetime.now(timezone.utc).isoformat()

    def _handle_event(self, event: dict[str, Any]) -> None:
        self._emitter.emit({"type": "event", "event": event})

    def emit_snapshot(self) -> None:
        snapshot = self._mr1.build_timeline_snapshot()
        def is_recent(value: Any) -> bool:
            return isinstance(value, str) and value >= self._history_cutoff

        snapshot["tasks"] = [
            task
            for task in snapshot.get("tasks", [])
            if task.get("status") == "running"
            or is_recent(task.get("started_at"))
            or is_recent(task.get("finished_at"))
            or is_recent(task.get("updated_at"))
        ]
        snapshot["conversation"] = [
            entry
            for entry in snapshot.get("conversation", [])
            if is_recent(entry.get("timestamp"))
        ]
        snapshot["events"] = [
            entry
            for entry in snapshot.get("events", [])
            if is_recent(entry.get("timestamp"))
        ]
        self._emitter.emit({"type": "snapshot", "snapshot": snapshot})

    def start(self) -> None:
        try:
            self._mr1.start()
        except Exception as e:  # pragma: no cover - defensive
            self._emitter.emit(
                {"type": "error", "message": f"Failed to start MR1 bridge: {e}", "fatal": True}
            )
            raise

        self._emitter.emit(
            {
                "type": "ready",
                "session_id": self._mr1._state.session_id,
            }
        )
        self.emit_snapshot()

        self._ticker = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._ticker.start()

    def _snapshot_loop(self) -> None:
        while not self._stop.wait(self._snapshot_interval_s):
            try:
                self.emit_snapshot()
            except Exception as e:  # pragma: no cover - defensive
                self._emitter.emit({"type": "error", "message": str(e), "fatal": False})

    def close(self) -> None:
        self._stop.set()
        if self._ticker and self._ticker.is_alive():
            self._ticker.join(timeout=1)
        killed = self._mr1.shutdown("bridge_shutdown")
        self._emitter.emit({"type": "shutdown", "killed": killed})

    def handle_input(self, text: str) -> None:
        builtin_result = self._mr1._handle_builtin(text)
        if builtin_result is not None:
            self._emitter.emit(
                {"type": "command_result", "command": text, "output": builtin_result}
            )
            self.emit_snapshot()
            return

        try:
            response = self._mr1.step(text)
        except Exception as e:  # pragma: no cover - defensive
            self._emitter.emit({"type": "error", "message": str(e), "fatal": False})
            return

        self._emitter.emit({"type": "response", "text": response})
        self.emit_snapshot()

    def process_message(self, payload: dict[str, Any]) -> bool:
        msg_type = payload.get("type")

        if msg_type == "input":
            text = str(payload.get("text", "")).strip()
            if text:
                self.handle_input(text)
            return True

        if msg_type == "snapshot":
            self.emit_snapshot()
            return True

        if msg_type == "shutdown":
            self.close()
            return False

        self._emitter.emit(
            {"type": "error", "message": f"Unknown bridge message type: {msg_type}", "fatal": False}
        )
        return True


def main() -> None:
    interval = float(os.getenv("MR1_UI_SNAPSHOT_INTERVAL_S", "0.35"))
    session = BridgeSession(snapshot_interval_s=interval)
    session.start()

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                session._emitter.emit(
                    {"type": "error", "message": "Malformed bridge JSON input", "fatal": False}
                )
                continue

            keep_going = session.process_message(payload)
            if not keep_going:
                return
    finally:
        if not session._stop.is_set():
            session.close()


if __name__ == "__main__":
    main()
