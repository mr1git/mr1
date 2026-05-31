from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TUI_MODE_TREE = "tree"
TUI_MODE_TIMELINE = "timeline"
TUI_MODE_DETAIL = "detail"

DETAIL_KIND_AGENT = "agent"
DETAIL_KIND_EVENT = "event"


@dataclass
class TuiState:
    mode: str = TUI_MODE_TREE
    previous_mode: str = TUI_MODE_TREE
    detail_kind: Optional[str] = None
    selected_agent_id: Optional[str] = None
    selected_event_id: Optional[str] = None
    show_dead: bool = True
    live_follow: bool = True
    show_help: bool = False
    status_message: Optional[str] = None

    def base_mode(self) -> str:
        if self.mode == TUI_MODE_DETAIL:
            return self.previous_mode
        return self.mode
