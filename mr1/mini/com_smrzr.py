"""
com_smrzr — Communication Summarizer
======================================
Reads tasks/{task_id}/comms/, summarizes the contents into a single
markdown file at tasks/{task_id}/summary.md, writes compressed highlights
to memory/rag/ via mem_rtvr.ingest_chunks(), and signals that the comms
folder can be deleted.

No LLM calls. Deterministic extraction of key lines and structure.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.core import Logger
from mr1.mini import mem_rtvr


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent
_TASKS_DIR = _PKG_ROOT / "tasks"

# Maximum characters per highlights chunk for RAG.
_HIGHLIGHT_CHUNK_SIZE = 400


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class SmrzrResult:
    """Result of a summarization run."""

    def __init__(self):
        self.summary_file: Optional[str] = None
        self.comms_read: int = 0
        self.highlights_ingested: int = 0
        self.comms_clearable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_file": self.summary_file,
            "comms_read": self.comms_read,
            "highlights_ingested": self.highlights_ingested,
            "comms_clearable": self.comms_clearable,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize(
    task_id: str,
    logger: Optional[Logger] = None,
) -> SmrzrResult:
    """
    Summarize all communications for a task.

    1. Read every file in tasks/{task_id}/comms/
    2. Extract structure: who said what, key decisions, outcomes
    3. Write tasks/{task_id}/summary.md
    4. Write compressed highlights to RAG via mem_rtvr.ingest_chunks
    5. Signal that comms/ can be deleted

    Returns SmrzrResult with metadata.
    """
    result = SmrzrResult()
    if logger is None:
        logger = Logger()

    comms_dir = _TASKS_DIR / task_id / "comms"
    task_dir = _TASKS_DIR / task_id

    if not comms_dir.exists():
        logger.log(task_id, "com_smrzr", "summarize", "ok",
                    metadata={"note": "no comms directory"})
        return result

    # --- Read all comm files ---
    entries = _read_comms(comms_dir)
    result.comms_read = len(entries)

    if not entries:
        logger.log(task_id, "com_smrzr", "summarize", "ok",
                    metadata={"note": "comms directory empty"})
        return result

    # --- Build summary ---
    summary_md = _build_summary(task_id, entries)

    # --- Write summary file ---
    task_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    result.summary_file = str(summary_path)

    logger.log(task_id, "com_smrzr", "write_summary", "ok",
               metadata={"file": str(summary_path), "size": len(summary_md)})

    # --- Extract highlights and ingest into RAG ---
    highlights = _extract_highlights(task_id, entries)
    if highlights:
        ingested = mem_rtvr.ingest_chunks(highlights, logger=logger)
        result.highlights_ingested = ingested

    # --- Signal comms clearable ---
    result.comms_clearable = True

    logger.log(task_id, "com_smrzr", "summarize", "ok",
               metadata=result.to_dict())

    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _read_comms(comms_dir: Path) -> list[dict[str, Any]]:
    """
    Read all files in a comms directory, sorted by modification time.

    Each file is treated as one communication entry. Supports .json
    (structured) and .txt/.md (plain text).
    """
    entries = []

    files = sorted(comms_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    for path in files:
        if path.is_dir() or path.name.startswith("."):
            continue

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        entry = {
            "filename": path.name,
            "mtime": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }

        if path.suffix == ".json":
            try:
                entry["data"] = json.loads(raw)
                entry["text"] = _flatten_json_entry(entry["data"])
            except json.JSONDecodeError:
                entry["text"] = raw
        else:
            entry["text"] = raw

        entries.append(entry)

    return entries


def _flatten_json_entry(data: dict) -> str:
    """Extract readable text from a structured comm entry."""
    parts = []
    for key in ("agent_type", "action", "result", "output", "message"):
        if key in data:
            parts.append(f"{key}: {data[key]}")
    if "metadata" in data and isinstance(data["metadata"], dict):
        for k, v in data["metadata"].items():
            parts.append(f"  {k}: {v}")
    return "\n".join(parts) if parts else json.dumps(data, indent=2)


def _build_summary(task_id: str, entries: list[dict]) -> str:
    """
    Build a markdown summary from communication entries.

    Structure:
      # Task {task_id} Summary
      ## Timeline
      - {timestamp} {agent}: {action/message snippet}
      ## Key Outputs
      - Truncated outputs from each entry
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Task {task_id} — Summary",
        f"Generated: {ts}",
        f"Communications: {len(entries)}",
        "",
        "## Timeline",
        "",
    ]

    for entry in entries:
        time_str = entry.get("mtime", "?")[:19]
        filename = entry["filename"]
        # First line of the text as a snippet.
        text = entry.get("text", "")
        snippet = text.split("\n")[0][:120]
        lines.append(f"- **{time_str}** `{filename}`: {snippet}")

    lines.extend(["", "## Details", ""])

    for entry in entries:
        filename = entry["filename"]
        text = entry.get("text", "")
        # Truncate long entries.
        if len(text) > 800:
            text = text[:800] + "\n...(truncated)"
        lines.append(f"### {filename}")
        lines.append(f"```\n{text}\n```")
        lines.append("")

    return "\n".join(lines)


def _extract_highlights(
    task_id: str,
    entries: list[dict],
) -> list[dict[str, str]]:
    """
    Extract compressed highlights from comms for RAG ingestion.

    Takes the first meaningful line from each entry and bundles them
    into chunks suitable for mem_rtvr.ingest_chunks().
    """
    highlights = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Collect snippets from all entries.
    snippets = []
    for entry in entries:
        text = entry.get("text", "").strip()
        if not text:
            continue
        # Take first 2 non-empty lines as the highlight.
        meaningful = [
            line.strip() for line in text.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ][:2]
        if meaningful:
            snippets.append(" | ".join(meaningful))

    # Bundle snippets into chunks.
    current = f"Task {task_id} highlights:\n"
    chunk_idx = 0

    for snippet in snippets:
        candidate = f"{current}- {snippet}\n"
        if len(candidate) > _HIGHLIGHT_CHUNK_SIZE:
            highlights.append({
                "id": f"smrzr-{task_id}-{chunk_idx}",
                "text": current.strip(),
                "source": f"summary:{task_id}",
            })
            chunk_idx += 1
            current = f"Task {task_id} highlights (cont):\n- {snippet}\n"
        else:
            current = candidate

    # Don't forget the last chunk.
    if current.strip() and current.strip() != f"Task {task_id} highlights:":
        highlights.append({
            "id": f"smrzr-{task_id}-{chunk_idx}",
            "text": current.strip(),
            "source": f"summary:{task_id}",
        })

    return highlights
