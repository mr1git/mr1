"""Tests for the Ink UI bridge."""

import json
from io import StringIO
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

from mr1.ui_bridge import BridgeSession, JsonEmitter


class FakeMR1:
    def __init__(self):
        self._state = SimpleNamespace(session_id="sess-123")
        self.shutdown_reason = None

    def start(self):
        return None

    def build_timeline_snapshot(self):
        old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        new = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
        return {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "session": {"session_id": "sess-123", "started_at": None, "status": "running"},
            "summary": {"task_count": 0, "running_count": 0, "decision_count": 0},
            "root": {"id": "mr1", "name": "MR1", "status": "running"},
            "tasks": [],
            "events": [
                {"timestamp": old, "summary": "stale event", "lane": "system"},
                {"timestamp": new, "summary": "fresh event", "lane": "system"},
            ],
            "recent_decisions": [],
            "conversation": [
                {"timestamp": old, "role": "user", "text": "old", "kind": "message", "lane": "conversation"},
                {"timestamp": new, "role": "user", "text": "new", "kind": "message", "lane": "conversation"},
            ],
        }

    def _handle_builtin(self, text: str):
        if text == "/status":
            return "status block"
        return None

    def step(self, text: str):
        return f"echo: {text}"

    def shutdown(self, reason: str = "user"):
        self.shutdown_reason = reason
        return 2


def _read_messages(buffer: StringIO) -> list[dict]:
    buffer.seek(0)
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_bridge_start_emits_ready_and_snapshot():
    output = StringIO()
    session = BridgeSession(
        mr1_instance=FakeMR1(),
        emitter=JsonEmitter(output),
        snapshot_interval_s=999,
    )

    session.start()
    session.close()

    messages = _read_messages(output)
    assert messages[0]["type"] == "ready"
    assert messages[1]["type"] == "snapshot"
    assert messages[1]["snapshot"]["conversation"] == [
        {"timestamp": messages[1]["snapshot"]["conversation"][0]["timestamp"], "role": "user", "text": "new", "kind": "message", "lane": "conversation"}
    ]
    assert messages[1]["snapshot"]["events"] == [
        {"timestamp": messages[1]["snapshot"]["events"][0]["timestamp"], "summary": "fresh event", "lane": "system"}
    ]
    assert messages[-1]["type"] == "shutdown"


def test_bridge_processes_input_and_builtin_results():
    output = StringIO()
    session = BridgeSession(
        mr1_instance=FakeMR1(),
        emitter=JsonEmitter(output),
        snapshot_interval_s=999,
    )
    session.start()
    output.truncate(0)
    output.seek(0)

    session.process_message({"type": "input", "text": "/status"})
    session.process_message({"type": "input", "text": "hello"})
    session.close()

    messages = _read_messages(output)
    assert any(msg["type"] == "command_result" and msg["output"] == "status block" for msg in messages)
    assert any(msg["type"] == "response" and msg["text"] == "echo: hello" for msg in messages)


def test_bridge_forwards_runtime_events():
    output = StringIO()
    session = BridgeSession(
        mr1_instance=FakeMR1(),
        emitter=JsonEmitter(output),
        snapshot_interval_s=999,
    )

    session._handle_event({"type": "task_spawned", "task_id": "task-1"})
    messages = _read_messages(output)
    assert messages == [{"type": "event", "event": {"type": "task_spawned", "task_id": "task-1"}}]
