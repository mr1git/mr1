"""
Tests for mr1.chat_input and mr1.trace_emitter.

Covers:
- KNOWN_COMMANDS contains expected commands
- ChatCompleter slash-command prefix filtering
- ChatCompleter argument completion via id_fetchers
- ChatInput.summarize_for_display multiline paste handling
- Key bindings include both Shift+Enter and Ctrl+N
- format_trace_line produces compact readable lines
- TraceEmitter starts and stops cleanly
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mr1.chat_input import (
    KNOWN_COMMANDS,
    ChatCompleter,
    ChatInput,
    make_key_bindings,
    PROMPT_TOOLKIT_AVAILABLE,
)
from mr1.trace_emitter import TraceEmitter, format_trace_line


# ---------------------------------------------------------------------------
# KNOWN_COMMANDS
# ---------------------------------------------------------------------------

def test_known_commands_contains_core_commands():
    for cmd in ("/agent", "/workflow", "/status", "/help", "/stop", "/kill", "/clear"):
        assert cmd in KNOWN_COMMANDS, f"{cmd} missing from KNOWN_COMMANDS"


def test_known_commands_are_sorted():
    assert KNOWN_COMMANDS == sorted(KNOWN_COMMANDS)


# ---------------------------------------------------------------------------
# ChatInput.summarize_for_display
# ---------------------------------------------------------------------------

def test_summarize_single_line():
    assert ChatInput.summarize_for_display("hello world") == "hello world"


def test_summarize_multiline():
    text = "line1\nline2\nline3"
    assert ChatInput.summarize_for_display(text) == "[Pasted 3 lines]"


def test_summarize_two_lines():
    assert ChatInput.summarize_for_display("a\nb") == "[Pasted 2 lines]"


def test_summarize_preserves_single_line_commands():
    assert ChatInput.summarize_for_display("/agent ag-123") == "/agent ag-123"


# ---------------------------------------------------------------------------
# ChatCompleter (requires prompt_toolkit)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PROMPT_TOOLKIT_AVAILABLE, reason="prompt_toolkit not installed")
class TestChatCompleter:
    def _completions(self, completer: ChatCompleter, text: str) -> list[str]:
        from prompt_toolkit.document import Document
        doc = Document(text, cursor_position=len(text))
        return [c.display for c in completer.get_completions(doc, None)]

    def _completion_texts(self, completer: ChatCompleter, text: str) -> list[str]:
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import to_plain_text
        doc = Document(text, cursor_position=len(text))
        results = []
        for c in completer.get_completions(doc, None):
            display = c.display
            if hasattr(display, "__iter__") and not isinstance(display, str):
                results.append(to_plain_text(display))
            else:
                results.append(str(display))
        return results

    def test_slash_prefix_returns_matching_commands(self):
        completer = ChatCompleter({})
        results = self._completion_texts(completer, "/a")
        assert "/agent" in results
        assert "/agents" in results
        assert "/approvals" in results
        # Should not include /status
        assert "/status" not in results

    def test_slash_prefix_wor_returns_workflow_only(self):
        completer = ChatCompleter({})
        results = self._completion_texts(completer, "/wor")
        assert all("workflow" in r for r in results), f"unexpected results: {results}"
        assert "/workflow" in results
        assert "/workflows" in results

    def test_slash_prefix_status_returns_only_status(self):
        completer = ChatCompleter({})
        results = self._completion_texts(completer, "/stat")
        assert "/status" in results
        assert len(results) == 1

    def test_no_completions_without_slash(self):
        completer = ChatCompleter({})
        results = self._completion_texts(completer, "hello")
        assert results == []

    def test_argument_completion_calls_fetcher(self):
        fetched = ["ag-001", "ag-002", "ag-003"]
        fetcher = MagicMock(return_value=fetched)
        completer = ChatCompleter({"agent": fetcher})
        results = self._completion_texts(completer, "/agent ")
        fetcher.assert_called_once()
        for aid in fetched:
            assert aid in results

    def test_argument_completion_filters_by_prefix(self):
        fetched = ["ag-001", "ag-002", "wf-999"]
        fetcher = MagicMock(return_value=fetched)
        completer = ChatCompleter({"agent": fetcher})
        results = self._completion_texts(completer, "/agent ag-")
        assert "ag-001" in results
        assert "ag-002" in results
        assert "wf-999" not in results

    def test_workflow_argument_completion(self):
        fetched = ["wf-abc", "wf-def"]
        fetcher = MagicMock(return_value=fetched)
        completer = ChatCompleter({"workflow": fetcher})
        results = self._completion_texts(completer, "/workflow ")
        fetcher.assert_called_once()
        assert "wf-abc" in results
        assert "wf-def" in results

    def test_fetcher_exception_returns_empty(self):
        def bad_fetcher():
            raise RuntimeError("db error")
        completer = ChatCompleter({"agent": bad_fetcher})
        results = self._completion_texts(completer, "/agent ")
        assert results == []

    def test_unknown_command_arg_returns_empty(self):
        completer = ChatCompleter({})
        results = self._completion_texts(completer, "/status ")
        assert results == []


# ---------------------------------------------------------------------------
# Key bindings
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PROMPT_TOOLKIT_AVAILABLE, reason="prompt_toolkit not installed")
def test_key_bindings_include_newline_inserts():
    kb = make_key_bindings()
    assert kb is not None
    bound_keys = [str(b.keys) for b in kb.bindings]
    # Ctrl+N — reliable cross-terminal newline insert
    ctrl_n = any("c-n" in k.lower() for k in bound_keys)
    # escape+enter — Alt+Enter fallback; prompt_toolkit normalises "enter" -> "c-m"
    alt_enter = any("escape" in k.lower() and "c-m" in k.lower() for k in bound_keys)
    assert ctrl_n, f"Ctrl+N not found in bindings: {bound_keys}"
    assert alt_enter, f"Alt+Enter (escape+enter) not found in bindings: {bound_keys}"


# ---------------------------------------------------------------------------
# format_trace_line
# ---------------------------------------------------------------------------

def _make_event(event_type, actor_id="ag-test", status="ok", **extra):
    ev = {
        "event_type": event_type,
        "actor_id": actor_id,
        "status": status,
        "summary": extra.pop("summary", ""),
        "metadata": extra.pop("metadata", {}),
    }
    ev.update(extra)
    return ev


def test_format_mrn_step_started():
    ev = _make_event("mrn_step_started", actor_id="ag-abc", metadata={"iteration": 3})
    line = format_trace_line(ev)
    assert "ag-abc" in line
    assert "3" in line
    assert "started" in line


def test_format_mrn_step_completed():
    ev = _make_event(
        "mrn_step_completed",
        actor_id="ag-xyz",
        status="waiting_on_parent",
        metadata={"action": "ask_parent", "iteration": 2},
    )
    line = format_trace_line(ev)
    assert "ag-xyz" in line
    assert "ask_parent" in line
    assert "waiting_on_parent" in line


def test_format_mrn_reported():
    ev = _make_event("mrn_reported", actor_id="Sentinel")
    line = format_trace_line(ev)
    assert "Sentinel" in line
    assert "report" in line


def test_format_capability_invoked():
    ev = _make_event("capability_invoked", actor_id="ag-001", status="running", summary="run bash")
    line = format_trace_line(ev)
    assert "capability" in line
    assert "ag-001" in line


def test_format_long_summary_truncated():
    long_summary = "x" * 100
    ev = _make_event("capability_completed", actor_id="ag-001", summary=long_summary)
    line = format_trace_line(ev)
    assert "[...]" in line
    assert len(line) < 200


# ---------------------------------------------------------------------------
# TraceEmitter
# ---------------------------------------------------------------------------

def test_trace_emitter_starts_and_stops_on_missing_file(tmp_path):
    path = tmp_path / "nonexistent_events.jsonl"
    emitter = TraceEmitter()
    emitter.start(path)
    time.sleep(0.1)
    emitter.stop()


def test_trace_emitter_emits_new_events(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("")  # create empty file

    received: list[str] = []

    def capture(line: str):
        received.append(line)

    emitter = TraceEmitter()
    emitter.start(path, print_fn=capture)
    time.sleep(0.1)

    event = {
        "event_type": "mrn_step_started",
        "actor_id": "ag-emit-test",
        "status": "started",
        "summary": "test",
        "metadata": {"iteration": 1},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()

    # Give the tail thread time to pick it up.
    deadline = time.time() + 2.0
    while not received and time.time() < deadline:
        time.sleep(0.05)

    emitter.stop()
    assert any("ag-emit-test" in line for line in received), f"No trace line found: {received}"


def test_trace_emitter_skips_historical_events(tmp_path):
    path = tmp_path / "events.jsonl"
    old_event = {
        "event_type": "mrn_step_started",
        "actor_id": "ag-old",
        "status": "started",
        "summary": "",
        "metadata": {"iteration": 0},
    }
    path.write_text(json.dumps(old_event) + "\n")

    received: list[str] = []
    emitter = TraceEmitter()
    emitter.start(path, print_fn=received.append)
    time.sleep(0.3)
    emitter.stop()

    assert not any("ag-old" in line for line in received), (
        f"Historical event should have been skipped: {received}"
    )


def test_trace_emitter_ignores_non_trace_events(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("")

    received: list[str] = []
    emitter = TraceEmitter()
    emitter.start(path, print_fn=received.append)
    time.sleep(0.1)

    noise = {"event_type": "snapshot_saved", "actor_id": "mr1", "status": "ok", "summary": ""}
    with open(path, "a") as f:
        f.write(json.dumps(noise) + "\n")
        f.flush()

    time.sleep(0.4)
    emitter.stop()
    assert received == [], f"Non-trace events should be ignored: {received}"
