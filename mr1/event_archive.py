"""
Event archive — sealed, queryable segments of timeline history.

`events.jsonl` is append-only and grows forever. Rotation moves sealed history
out of the live file and into numbered segments under `events/archive/`, with
`events/segments.json` as the manifest.

The manifest is the authority, not the directory listing. Rotation writes the
segment first, fsyncs it, then updates the manifest, then rewrites the live
file. Crash anywhere in that sequence is safe:

  * crash after the segment, before the manifest → the segment is orphaned and
    ignored; the live file still holds every event; the next rotation reuses
    the same segment number and overwrites it.
  * crash after the manifest, before the live rewrite → the events exist in
    both places; every reader dedupes by `event_id`, so they appear once.

What is never safe is losing an event, so nothing is ever removed from the live
file until its segment is durably on disk and named in the manifest.

Segments keep the same JSONL shape as the live file — one event per line, in
index order — so history stays greppable with or without MR1. Compression is
per-segment and transparent to readers.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


MANIFEST_NAME = "segments.json"
ARCHIVE_DIR_NAME = "archive"
MANIFEST_VERSION = 1
SEGMENT_STEM = "events"


@dataclass(frozen=True)
class ArchiveSegment:
    """One sealed run of history. Immutable once written."""

    name: str
    first_index: int
    last_index: int
    first_timestamp: str
    last_timestamp: str
    count: int
    sealed_at: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "first_index": self.first_index,
            "last_index": self.last_index,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "count": self.count,
            "sealed_at": self.sealed_at,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchiveSegment":
        return cls(
            name=str(payload["name"]),
            first_index=int(payload["first_index"]),
            last_index=int(payload["last_index"]),
            first_timestamp=str(payload.get("first_timestamp") or ""),
            last_timestamp=str(payload.get("last_timestamp") or ""),
            count=int(payload.get("count", 0)),
            sealed_at=str(payload.get("sealed_at") or ""),
            size_bytes=int(payload.get("size_bytes", 0)),
        )


class EventArchive:
    """Sealed segments plus the manifest that indexes them."""

    def __init__(self, events_dir: Path, *, compress: bool = True):
        self._events_dir = Path(events_dir)
        self._compress = compress

    @property
    def archive_dir(self) -> Path:
        return self._events_dir / ARCHIVE_DIR_NAME

    @property
    def manifest_path(self) -> Path:
        return self._events_dir / MANIFEST_NAME

    # -- manifest ------------------------------------------------------

    def read_manifest(self) -> dict[str, Any]:
        path = self.manifest_path
        if not path.exists():
            return {"version": MANIFEST_VERSION, "segments": [], "last_index": 0}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            # A corrupt manifest must not be read as "no history" — that would
            # let rotation restart indices from 1 and silently fork the log.
            raise EventArchiveError(f"event archive manifest is unreadable: {path}")
        if not isinstance(payload, dict):
            raise EventArchiveError(f"event archive manifest is malformed: {path}")
        payload.setdefault("segments", [])
        payload.setdefault("last_index", 0)
        return payload

    def segments(self) -> list[ArchiveSegment]:
        manifest = self.read_manifest()
        items = [
            ArchiveSegment.from_dict(entry)
            for entry in manifest.get("segments", [])
            if isinstance(entry, dict)
        ]
        return sorted(items, key=lambda item: item.first_index)

    def archived_last_index(self) -> int:
        """The highest event index that has been sealed. 0 when empty."""
        return int(self.read_manifest().get("last_index", 0) or 0)

    def archived_count(self) -> int:
        return sum(segment.count for segment in self.segments())

    def total_bytes(self) -> int:
        return sum(segment.size_bytes for segment in self.segments())

    def is_empty(self) -> bool:
        return not self.segments()

    # -- reading -------------------------------------------------------

    def iter_raw_events(self) -> Iterator[dict[str, Any]]:
        """Every archived event, oldest first, across all segments."""
        for segment in self.segments():
            path = self.archive_dir / segment.name
            if not path.exists():
                # Named in the manifest but missing on disk: that is data loss,
                # not an empty result. Say so rather than under-reporting history.
                raise EventArchiveError(
                    f"archive segment named in the manifest is missing: {path}"
                )
            with self._open_segment(path) as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        continue

    def _open_segment(self, path: Path):
        if path.name.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    # -- writing -------------------------------------------------------

    def seal(
        self,
        payloads: list[dict[str, Any]],
        *,
        sealed_at: str,
    ) -> Optional[ArchiveSegment]:
        """
        Write `payloads` as a new sealed segment and name it in the manifest.

        Returns None for an empty input. The caller only removes these events
        from the live log *after* this returns.
        """
        if not payloads:
            return None
        manifest = self.read_manifest()
        existing = manifest.get("segments", [])
        number = len(existing) + 1

        indices = [int(item["event_index"]) for item in payloads]
        suffix = ".jsonl.gz" if self._compress else ".jsonl"
        name = f"{SEGMENT_STEM}-{number:06d}{suffix}"

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        path = self.archive_dir / name
        tmp = path.with_name(path.name + ".tmp")
        self._write_segment(tmp, payloads)
        tmp.replace(path)
        _fsync_directory(self.archive_dir)

        segment = ArchiveSegment(
            name=name,
            first_index=min(indices),
            last_index=max(indices),
            first_timestamp=str(payloads[0].get("timestamp") or ""),
            last_timestamp=str(payloads[-1].get("timestamp") or ""),
            count=len(payloads),
            sealed_at=sealed_at,
            size_bytes=path.stat().st_size,
        )
        manifest["version"] = MANIFEST_VERSION
        manifest["segments"] = list(existing) + [segment.to_dict()]
        manifest["last_index"] = max(
            int(manifest.get("last_index", 0) or 0),
            segment.last_index,
        )
        self._write_manifest(manifest)
        return segment

    def _write_segment(self, path: Path, payloads: list[dict[str, Any]]) -> None:
        opener = (
            (lambda: gzip.open(path, "wt", encoding="utf-8"))
            if self._compress else
            (lambda: open(path, "w", encoding="utf-8"))
        )
        with opener() as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except (AttributeError, OSError):  # pragma: no cover - gzip wrapper
                pass

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        path = self.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        _fsync_directory(path.parent)


class EventArchiveError(RuntimeError):
    """The archive is inconsistent. Never swallowed — history must not lie."""


def _fsync_directory(path: Path) -> None:
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
