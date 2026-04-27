"""
Persistent local agent messaging and scoped inbox helpers.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mr1.scoped_agents import PersistentAgentStore


_DEFAULT_ROOT = Path(__file__).resolve().parent / "memory" / "messages"
_ALLOWED_MESSAGE_KINDS = frozenset({
    "report",
    "question",
    "alert",
    "status",
    "request",
})
_ALLOWED_MESSAGE_STATUSES = frozenset({
    "unread",
    "read",
    "archived",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_message_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"msg-{timestamp}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class PersistentMessage:
    message_id: str
    from_agent_id: str
    to_agent_id: str
    kind: str
    subject: str
    body: str
    workflow_id: Optional[str]
    task_id: Optional[str]
    created_at: str
    read_at: Optional[str]
    archived_at: Optional[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "from_agent_id": self.from_agent_id,
            "to_agent_id": self.to_agent_id,
            "kind": self.kind,
            "subject": self.subject,
            "body": self.body,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "read_at": self.read_at,
            "archived_at": self.archived_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PersistentMessage":
        message = cls(
            message_id=str(data["message_id"]),
            from_agent_id=str(data["from_agent_id"]),
            to_agent_id=str(data["to_agent_id"]),
            kind=str(data["kind"]),
            subject=str(data["subject"]),
            body=str(data["body"]),
            workflow_id=str(data["workflow_id"]) if data.get("workflow_id") else None,
            task_id=str(data["task_id"]) if data.get("task_id") else None,
            created_at=str(data["created_at"]),
            read_at=str(data["read_at"]) if data.get("read_at") else None,
            archived_at=str(data["archived_at"]) if data.get("archived_at") else None,
            status=str(data["status"]),
        )
        _validate_message(message)
        return message


def _validate_message(message: PersistentMessage) -> None:
    if message.kind not in _ALLOWED_MESSAGE_KINDS:
        raise ValueError(f"invalid message kind: {message.kind}")
    if message.status not in _ALLOWED_MESSAGE_STATUSES:
        raise ValueError(f"invalid message status: {message.status}")
    if not message.subject.strip():
        raise ValueError("message subject must be non-empty")
    if not message.body.strip():
        raise ValueError("message body must be non-empty")


class MessageStore:
    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
    ):
        self._root = Path(root) if root else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._scoped_agents = scoped_agent_store or PersistentAgentStore(
            root=self._root.parent / "agents"
        )
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    def message_path(self, message_id: str) -> Path:
        return self._root / f"{message_id}.json"

    def can_agent_access_message(
        self,
        agent_id: str,
        message: PersistentMessage,
    ) -> bool:
        if self._scoped_agents.is_root_agent(agent_id):
            return True
        return (
            message.to_agent_id == agent_id
            or message.from_agent_id == agent_id
        )

    def can_agent_send_message(self, from_agent_id: str, to_agent_id: str) -> bool:
        sender = self._scoped_agents.load_agent(from_agent_id)
        recipient = self._scoped_agents.load_agent(to_agent_id)
        if sender is None or recipient is None:
            return False
        if self._scoped_agents.is_root_agent(from_agent_id):
            return True
        return (
            recipient.agent_id == sender.parent_agent_id
            or recipient.agent_id in self._scoped_agents.descendant_ids(from_agent_id)
        )

    def create_message(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        kind: str,
        subject: str,
        body: str,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> PersistentMessage:
        kind = kind.strip()
        if kind not in _ALLOWED_MESSAGE_KINDS:
            raise ValueError(f"invalid message kind: {kind}")
        message = PersistentMessage(
            message_id=new_message_id(),
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            kind=kind,
            subject=subject.strip(),
            body=body.rstrip(),
            workflow_id=workflow_id,
            task_id=task_id,
            created_at=_now_iso(),
            read_at=None,
            archived_at=None,
            status="unread",
        )
        _validate_message(message)
        with self._lock:
            self._write_message(message)
        return message

    def get_message(self, message_id: str) -> Optional[PersistentMessage]:
        with self._lock:
            path = self.message_path(message_id)
            if not path.exists():
                return None
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return PersistentMessage.from_dict(json.load(handle))
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                return None

    def list_inbox(
        self,
        agent_id: str,
        include_archived: bool = False,
    ) -> list[PersistentMessage]:
        messages = [
            message
            for message in self._list_messages()
            if message.to_agent_id == agent_id
            and (include_archived or message.status != "archived")
        ]
        return sorted(messages, key=lambda item: (item.created_at, item.message_id), reverse=True)

    def list_outbox(
        self,
        agent_id: str,
        include_archived: bool = False,
    ) -> list[PersistentMessage]:
        messages = [
            message
            for message in self._list_messages()
            if message.from_agent_id == agent_id
            and (include_archived or message.status != "archived")
        ]
        return sorted(messages, key=lambda item: (item.created_at, item.message_id), reverse=True)

    def mark_read(self, message_id: str) -> Optional[PersistentMessage]:
        with self._lock:
            message = self.get_message(message_id)
            if message is None:
                return None
            if message.status == "archived":
                return message
            if message.status == "read" and message.read_at:
                return message
            updated = PersistentMessage(
                message_id=message.message_id,
                from_agent_id=message.from_agent_id,
                to_agent_id=message.to_agent_id,
                kind=message.kind,
                subject=message.subject,
                body=message.body,
                workflow_id=message.workflow_id,
                task_id=message.task_id,
                created_at=message.created_at,
                read_at=_now_iso(),
                archived_at=message.archived_at,
                status="read",
            )
            self._write_message(updated)
            return updated

    def archive_message(self, message_id: str) -> Optional[PersistentMessage]:
        with self._lock:
            message = self.get_message(message_id)
            if message is None:
                return None
            if message.status == "archived" and message.archived_at:
                return message
            updated = PersistentMessage(
                message_id=message.message_id,
                from_agent_id=message.from_agent_id,
                to_agent_id=message.to_agent_id,
                kind=message.kind,
                subject=message.subject,
                body=message.body,
                workflow_id=message.workflow_id,
                task_id=message.task_id,
                created_at=message.created_at,
                read_at=message.read_at,
                archived_at=_now_iso(),
                status="archived",
            )
            self._write_message(updated)
            return updated

    def _list_messages(self) -> list[PersistentMessage]:
        with self._lock:
            messages: list[PersistentMessage] = []
            for path in sorted(self._root.glob("msg-*.json")):
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        messages.append(PersistentMessage.from_dict(json.load(handle)))
                except (OSError, json.JSONDecodeError, KeyError, ValueError):
                    continue
            return messages

    def _write_message(self, message: PersistentMessage) -> None:
        target = self.message_path(message.message_id)
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(message.to_dict(), handle, indent=2)
        tmp.replace(target)
