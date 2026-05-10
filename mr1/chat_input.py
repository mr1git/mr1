"""
Enhanced chat input for MR1 manual testing sessions.

Uses prompt_toolkit for:
- Slash command completion with Up/Down navigation
- Argument completion (agent/message/workflow ids) via Tab
- Multiline input: Enter submits, Shift+Enter or Ctrl+N inserts newline
- Persistent history across sessions (FileHistory)

Falls back gracefully when prompt_toolkit is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Generator, Iterable

_HISTORY_PATH = Path(__file__).resolve().parent / "memory" / "active" / "chat_history.txt"

# All built-in slash commands.  Keep this in sync with _handle_builtin in mr1.py.
KNOWN_COMMANDS: list[str] = sorted([
    "/agent",
    "/agents",
    "/approval",
    "/approvals",
    "/artifacts",
    "/capabilities",
    "/capability",
    "/clear",
    "/events",
    "/exit",
    "/help",
    "/history",
    "/inbox",
    "/inputs",
    "/jobs",
    "/kill",
    "/memdltr",
    "/message",
    "/outbox",
    "/result",
    "/schema",
    "/status",
    "/stop",
    "/tasks",
    "/test kill agents",
    "/test spawn agents",
    "/tool",
    "/tools",
    "/vizualize",
    "/visualize",
    "/visualize-web",
    "/watchers",
    "/workflow",
    "/workflows",
])

# Maps command word (after the leading /) to the id_fetcher key.
_ARGUMENT_COMMANDS: dict[str, str] = {
    "agent": "agent",
    "agents": "agent",
    "message": "message",
    "inbox": "message",
    "outbox": "message",
    "workflow": "workflow",
    "approval": "approval",
    "approvals": "approval",
}


def _matches(text: str, candidates: Iterable[str]) -> list[str]:
    """Return candidates that contain text as a prefix or substring."""
    text_lower = text.lower()
    return [c for c in candidates if text_lower in c.lower()]


try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    class ChatCompleter(Completer):
        def __init__(self, id_fetchers: dict[str, Callable[[], list[str]]]) -> None:
            self._id_fetchers = id_fetchers

        def get_completions(
            self, document, complete_event
        ) -> Generator[Completion, None, None]:
            text = document.text_before_cursor

            if not text.startswith("/"):
                return

            parts = text.split(" ", 1)
            cmd_word = parts[0]

            if len(parts) == 1:
                # Still typing the command name — complete against KNOWN_COMMANDS
                for cmd in KNOWN_COMMANDS:
                    if cmd.lower().startswith(cmd_word.lower()):
                        yield Completion(cmd[len(cmd_word):], display=cmd)
                return

            # Command already typed; complete the argument
            cmd_key = cmd_word.lstrip("/").lower()
            fetcher_key = _ARGUMENT_COMMANDS.get(cmd_key)
            if fetcher_key is None:
                return

            fetcher = self._id_fetchers.get(fetcher_key)
            if fetcher is None:
                return

            try:
                ids = fetcher()
            except Exception:
                return

            arg_so_far = parts[1]
            for id_val in ids:
                if arg_so_far.lower() in id_val.lower():
                    yield Completion(
                        id_val[len(arg_so_far):],
                        display=id_val,
                    )

    def make_key_bindings() -> KeyBindings:
        kb = KeyBindings()

        # Ctrl+N — reliable cross-terminal newline insert.
        @kb.add("c-n")
        def _ctrl_n(event):
            event.current_buffer.insert_text("\n")

        # Escape+Enter — Alt+Enter in most terminal emulators.
        @kb.add("escape", "enter")
        def _alt_enter(event):
            event.current_buffer.insert_text("\n")

        return kb

    class ChatInput:
        def __init__(
            self, id_fetchers: dict[str, Callable[[], list[str]]] | None = None
        ) -> None:
            _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._session: PromptSession = PromptSession(
                history=FileHistory(str(_HISTORY_PATH)),
                completer=ChatCompleter(id_fetchers or {}),
                complete_while_typing=True,
                key_bindings=make_key_bindings(),
                multiline=False,
            )

        def get_input(self) -> str:
            return self._session.prompt("\nyou > ")

        @staticmethod
        def summarize_for_display(text: str) -> str:
            lines = text.splitlines()
            if len(lines) > 1:
                return f"[Pasted {len(lines)} lines]"
            return text

    PROMPT_TOOLKIT_AVAILABLE = True

except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

    class ChatCompleter:  # type: ignore[no-redef]
        pass

    class ChatInput:  # type: ignore[no-redef]
        def __init__(self, id_fetchers=None) -> None:
            pass

        def get_input(self) -> str:
            return input("\nyou > ")

        @staticmethod
        def summarize_for_display(text: str) -> str:
            lines = text.splitlines()
            if len(lines) > 1:
                return f"[Pasted {len(lines)} lines]"
            return text

    def make_key_bindings():
        return None
