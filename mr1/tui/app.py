from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.application.current import get_app_or_none

from mr1.tui.data import RuntimeDataSource, RuntimeSnapshot, live_focus_agent_id
from mr1.tui.layout import render_screen
from mr1.tui.navigation import (
    coerce_selected_agent_id,
    coerce_selected_event_id,
    first_child_agent_id,
    newer_event_id,
    next_sibling_agent_id,
    older_event_id,
    parent_agent_id,
    previous_sibling_agent_id,
)
from mr1.tui.state import (
    DETAIL_KIND_AGENT,
    DETAIL_KIND_EVENT,
    TUI_MODE_DETAIL,
    TUI_MODE_TIMELINE,
    TUI_MODE_TREE,
    TuiState,
)


class MR1TUIApplication:
    def __init__(
        self,
        *,
        store_root: Optional[Path] = None,
        poll_interval_s: float = 0.5,
        event_limit: int = 200,
    ):
        self._data_source = RuntimeDataSource(store_root=store_root)
        self._poll_interval_s = poll_interval_s
        self._event_limit = event_limit
        self._state = TuiState()
        self._snapshot = self._data_source.load_snapshot(event_limit=self._event_limit)
        self._detail_payload: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._refresh_lock = threading.RLock()
        self._init_selection()
        self._application = Application(
            layout=Layout(
                Window(
                    content=FormattedTextControl(self._render_screen),
                    wrap_lines=False,
                    always_hide_cursor=True,
                )
            ),
            full_screen=True,
            key_bindings=self._build_key_bindings(),
            mouse_support=False,
        )

    def _init_selection(self) -> None:
        self._state.selected_agent_id = live_focus_agent_id(self._snapshot.tree, show_dead=True)
        self._state.selected_event_id = coerce_selected_event_id(self._snapshot.events, None)
        self._refresh_detail_payload()

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("q")
        def _quit(event) -> None:
            self._stop.set()
            event.app.exit()

        @kb.add("?")
        def _toggle_help(event) -> None:
            self._state.show_help = not self._state.show_help
            event.app.invalidate()

        @kb.add("r")
        def _refresh(event) -> None:
            self.refresh()
            event.app.invalidate()

        @kb.add("d")
        def _toggle_dead(event) -> None:
            if self._state.base_mode() != TUI_MODE_TREE:
                return
            self._state.show_dead = not self._state.show_dead
            self._state.selected_agent_id = coerce_selected_agent_id(
                self._snapshot.tree,
                self._state.selected_agent_id,
                show_dead=self._state.show_dead,
            )
            self._state.status_message = "show dead" if self._state.show_dead else "hide dead"
            self._refresh_detail_payload()
            event.app.invalidate()

        @kb.add("n")
        def _jump_now(event) -> None:
            if self._state.base_mode() == TUI_MODE_TIMELINE:
                self._state.selected_event_id = coerce_selected_event_id(self._snapshot.events, None)
                self._state.status_message = "newest event"
            else:
                self._state.live_follow = True
                self._state.selected_agent_id = live_focus_agent_id(
                    self._snapshot.tree,
                    show_dead=self._state.show_dead,
                )
                self._state.status_message = "live follow"
            self._refresh_detail_payload()
            event.app.invalidate()

        @kb.add("f")
        def _focus_mode(event) -> None:
            if self._state.mode == TUI_MODE_TIMELINE:
                self._state.mode = TUI_MODE_TREE
                self._state.previous_mode = TUI_MODE_TREE
                self._state.status_message = "tree"
            elif self._state.mode == TUI_MODE_TREE:
                self._state.previous_mode = TUI_MODE_TREE
                self._state.mode = TUI_MODE_TIMELINE
                self._state.live_follow = False
                self._state.selected_event_id = coerce_selected_event_id(self._snapshot.events, self._state.selected_event_id)
                self._state.status_message = "timeline"
            event.app.invalidate()

        @kb.add("escape")
        def _escape(event) -> None:
            if self._state.mode == TUI_MODE_DETAIL:
                self._state.mode = self._state.previous_mode
                self._state.detail_kind = None
                self._refresh_detail_payload()
            elif self._state.mode == TUI_MODE_TIMELINE:
                self._state.mode = TUI_MODE_TREE
                self._state.previous_mode = TUI_MODE_TREE
                self._state.detail_kind = None
                self._refresh_detail_payload()
            event.app.invalidate()

        @kb.add("enter")
        def _detail(event) -> None:
            if self._state.mode == TUI_MODE_TREE and self._state.selected_agent_id:
                self._state.previous_mode = TUI_MODE_TREE
                self._state.mode = TUI_MODE_DETAIL
                self._state.detail_kind = DETAIL_KIND_AGENT
            elif self._state.mode == TUI_MODE_TIMELINE and self._state.selected_event_id:
                self._state.previous_mode = TUI_MODE_TIMELINE
                self._state.mode = TUI_MODE_DETAIL
                self._state.detail_kind = DETAIL_KIND_EVENT
            self._refresh_detail_payload()
            event.app.invalidate()

        @kb.add("left")
        def _left(event) -> None:
            if self._state.mode != TUI_MODE_TREE or not self._state.selected_agent_id:
                return
            target = previous_sibling_agent_id(
                self._snapshot.tree,
                self._state.selected_agent_id,
                show_dead=self._state.show_dead,
            )
            if target:
                self._state.selected_agent_id = target
                self._state.live_follow = False
                self._refresh_detail_payload()
                event.app.invalidate()

        @kb.add("right")
        def _right(event) -> None:
            if self._state.mode != TUI_MODE_TREE or not self._state.selected_agent_id:
                return
            target = next_sibling_agent_id(
                self._snapshot.tree,
                self._state.selected_agent_id,
                show_dead=self._state.show_dead,
            )
            if target:
                self._state.selected_agent_id = target
                self._state.live_follow = False
                self._refresh_detail_payload()
                event.app.invalidate()

        @kb.add("up")
        def _up(event) -> None:
            if self._state.mode == TUI_MODE_TIMELINE:
                self._state.selected_event_id = older_event_id(self._snapshot.events, self._state.selected_event_id)
                self._refresh_detail_payload()
                event.app.invalidate()
                return
            if self._state.mode != TUI_MODE_TREE or not self._state.selected_agent_id:
                return
            target = parent_agent_id(
                self._snapshot.tree,
                self._state.selected_agent_id,
                show_dead=self._state.show_dead,
            )
            if target:
                self._state.selected_agent_id = target
                self._state.live_follow = False
                self._refresh_detail_payload()
                event.app.invalidate()

        @kb.add("down")
        def _down(event) -> None:
            if self._state.mode == TUI_MODE_TIMELINE:
                self._state.selected_event_id = newer_event_id(self._snapshot.events, self._state.selected_event_id)
                self._refresh_detail_payload()
                event.app.invalidate()
                return
            if self._state.mode != TUI_MODE_TREE or not self._state.selected_agent_id:
                return
            target = first_child_agent_id(
                self._snapshot.tree,
                self._state.selected_agent_id,
                show_dead=self._state.show_dead,
            )
            if target:
                self._state.selected_agent_id = target
                self._state.live_follow = False
                self._refresh_detail_payload()
                event.app.invalidate()

        return kb

    def _refresh_detail_payload(self) -> None:
        if self._state.show_help:
            self._detail_payload = None
            return
        try:
            if self._state.mode == TUI_MODE_DETAIL and self._state.detail_kind == DETAIL_KIND_EVENT and self._state.selected_event_id:
                self._detail_payload = self._data_source.event_detail(self._state.selected_event_id)
                return
            if self._state.selected_agent_id:
                self._detail_payload = self._data_source.agent_detail(self._state.selected_agent_id)
                return
        except Exception as exc:
            self._state.status_message = str(exc)
        self._detail_payload = None

    def refresh(self) -> None:
        with self._refresh_lock:
            try:
                self._snapshot = self._data_source.load_snapshot(event_limit=self._event_limit)
                self._state.selected_event_id = coerce_selected_event_id(
                    self._snapshot.events,
                    self._state.selected_event_id,
                )
                if self._state.mode == TUI_MODE_TREE and self._state.live_follow:
                    self._state.selected_agent_id = live_focus_agent_id(
                        self._snapshot.tree,
                        show_dead=self._state.show_dead,
                    )
                else:
                    self._state.selected_agent_id = coerce_selected_agent_id(
                        self._snapshot.tree,
                        self._state.selected_agent_id,
                        show_dead=self._state.show_dead,
                    )
                self._refresh_detail_payload()
            except Exception as exc:
                self._state.status_message = f"refresh failed: {exc}"

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll_interval_s):
            self.refresh()
            if self._application.is_running:
                self._application.invalidate()

    def _start_polling(self) -> None:
        if self._poll_thread is not None:
            return
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _render_screen(self):
        app = get_app_or_none()
        width = 120
        height = 40
        if app is not None:
            size = app.output.get_size()
            width = size.columns
            height = size.rows
        return ANSI(
            render_screen(
                self._snapshot,
                self._state,
                width=width,
                height=height,
                detail_payload=self._detail_payload,
            )
        )

    def run(self) -> int:
        self._start_polling()
        try:
            self._application.run()
            return 0
        finally:
            self._stop.set()
            if self._poll_thread is not None:
                self._poll_thread.join(timeout=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MR1 read-only runtime TUI")
    parser.add_argument(
        "--store-root",
        type=Path,
        default=None,
        help="Override the workflow store root (defaults to mr1/memory/workflows).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds for runtime refresh.",
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=200,
        help="Maximum number of recent events to load into timeline mode.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = MR1TUIApplication(
        store_root=args.store_root,
        poll_interval_s=max(args.poll_interval, 0.1),
        event_limit=max(args.event_limit, 1),
    )
    return app.run()
