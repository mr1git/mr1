"""
mem_dltr — Memory Deletion / Distillation Agent
=================================================
Called on command. Reads the current MR1 conversation state from
memory/active/, decides what to:
  - forget (delete from active)
  - dump as plain text to memory/dumps/
  - chunk for RAG indexing to memory/rag/ (via mem_rtvr.ingest_chunks)

Writes a dated memory dump file, then clears what it has processed
from active memory.

No LLM calls. Deterministic triage based on age and size heuristics.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.core import Logger
from mr1.mini import mem_rtvr


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent
_ACTIVE_DIR = _PKG_ROOT / "memory" / "active"
_DUMPS_DIR = _PKG_ROOT / "memory" / "dumps"
_RAG_DIR = _PKG_ROOT / "memory" / "rag"

# Chunks longer than this get split for RAG ingestion.
_RAG_CHUNK_SIZE = 500

# Decisions and tasks older than this many entries get distilled.
_MAX_ACTIVE_DECISIONS = 20
_MAX_ACTIVE_TASKS = 30


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class DltrResult:
    """Result of a memory distillation run."""

    def __init__(self):
        self.forgotten: int = 0        # Entries deleted outright.
        self.dumped: int = 0           # Entries written to dumps/.
        self.rag_chunks: int = 0       # Chunks sent to RAG store.
        self.dump_file: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "forgotten": self.forgotten,
            "dumped": self.dumped,
            "rag_chunks": self.rag_chunks,
            "dump_file": self.dump_file,
        }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = _RAG_CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks, breaking at paragraph boundaries when possible.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds chunk_size, split it by sentences.
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                buf = ""
                for sentence in sentences:
                    if len(buf) + len(sentence) + 1 <= chunk_size:
                        buf = f"{buf} {sentence}" if buf else sentence
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sentence
                if buf:
                    current = buf
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def distill(
    logger: Optional[Logger] = None,
) -> DltrResult:
    """
    Run one distillation pass over memory/active/.

    1. Read mr1_state.json
    2. Triage old decisions → dump + RAG
    3. Triage completed tasks → dump + RAG
    4. Write a dated dump file to memory/dumps/
    5. Prune the processed entries from active state
    6. Return stats

    Never raises — logs errors and continues.
    """
    result = DltrResult()
    if logger is None:
        logger = Logger()

    state_path = _ACTIVE_DIR / "mr1_state.json"
    if not state_path.exists():
        logger.log("mem_dltr", "mem_dltr", "distill", "ok",
                    metadata={"note": "no active state found"})
        return result

    # --- Load state ---
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.log("mem_dltr", "mem_dltr", "distill", "error",
                    metadata={"error": str(e)})
        return result

    decisions = state.get("decisions", [])
    tasks = state.get("tasks", {})

    # --- Identify what to distill ---

    # Decisions beyond the retention window.
    overflow_decisions = []
    if len(decisions) > _MAX_ACTIVE_DECISIONS:
        overflow_decisions = decisions[:-_MAX_ACTIVE_DECISIONS]
        state["decisions"] = decisions[-_MAX_ACTIVE_DECISIONS:]

    # Completed/failed/killed tasks.
    completed_tasks = {}
    remaining_tasks = {}
    for tid, task in tasks.items():
        if task.get("status") in ("completed", "failed", "killed"):
            completed_tasks[tid] = task
        else:
            remaining_tasks[tid] = task
    state["tasks"] = remaining_tasks

    # If nothing to distill, done.
    if not overflow_decisions and not completed_tasks:
        logger.log("mem_dltr", "mem_dltr", "distill", "ok",
                    metadata={"note": "nothing to distill"})
        return result

    # --- Build dump content ---

    dump_lines = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if overflow_decisions:
        dump_lines.append(f"# Decisions distilled at {ts}\n")
        for d in overflow_decisions:
            dump_lines.append(
                f"- [{d.get('timestamp', '?')[:19]}] {d.get('action', '?')}: "
                f"{d.get('input_summary', '?')}"
            )
            result.forgotten += 1
        dump_lines.append("")

    if completed_tasks:
        dump_lines.append(f"# Completed tasks distilled at {ts}\n")
        for tid, task in completed_tasks.items():
            dump_lines.append(
                f"## {tid} [{task.get('agent_type', '?')}] — {task.get('status', '?')}"
            )
            dump_lines.append(f"  {task.get('description', 'no description')}")
            started = task.get("started_at", "?")
            finished = task.get("finished_at", "?")
            dump_lines.append(f"  started={started[:19]}  finished={finished[:19]}")
            dump_lines.append("")
            result.dumped += 1

    dump_text = "\n".join(dump_lines)

    # --- Write dump file ---

    _DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    dump_filename = f"distill-{ts}.md"
    dump_path = _DUMPS_DIR / dump_filename
    dump_path.write_text(dump_text, encoding="utf-8")
    result.dump_file = dump_filename

    logger.log("mem_dltr", "mem_dltr", "dump", "ok",
               metadata={"file": dump_filename, "size": len(dump_text)})

    # --- Chunk and ingest into RAG ---

    chunks = _chunk_text(dump_text)
    if chunks:
        rag_chunks = [
            {
                "id": f"dltr-{ts}-{i}",
                "text": chunk,
                "source": dump_filename,
            }
            for i, chunk in enumerate(chunks)
        ]
        ingested = mem_rtvr.ingest_chunks(rag_chunks, logger=logger)
        result.rag_chunks = ingested

    # --- Write pruned state back ---

    try:
        tmp = state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.rename(state_path)
    except OSError as e:
        logger.log("mem_dltr", "mem_dltr", "prune", "error",
                    metadata={"error": str(e)})
        return result

    logger.log("mem_dltr", "mem_dltr", "distill", "ok",
               metadata=result.to_dict())

    return result
