"""
ctx_pkgr — Context Packager
=============================
Takes a task description, calls mem_rtvr to pull relevant memory,
then trims and formats the results into a tight context package dict
ready for a Kazi.

Ruthlessly minimal. Every token in the package must earn its place.
"""

from typing import Any, Optional

from mr1.core import Logger
from mr1.mini import mem_rtvr


# ---------------------------------------------------------------------------
# Limits — keep context packages small so Kazis have room to work.
# ---------------------------------------------------------------------------

# Maximum characters in the memory context section.
_MAX_MEMORY_CHARS = 2000

# Maximum number of memory chunks to include.
_MAX_MEMORY_CHUNKS = 3

# Maximum characters per individual chunk.
_MAX_CHUNK_CHARS = 600

# Minimum relevance score to include a chunk.
_MIN_RELEVANCE = 0.15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def package(
    task_id: str,
    task_description: str,
    allowed_tools: Optional[list[str]] = None,
    working_dir: Optional[str] = None,
    file_paths: Optional[list[str]] = None,
    extra_context: Optional[str] = None,
    logger: Optional[Logger] = None,
) -> dict[str, Any]:
    """
    Build a context package for a Kazi.

    1. Query mem_rtvr with the task description
    2. Filter and trim results
    3. Assemble into a context dict matching kazi.run()'s expected format

    Returns a dict ready to be passed directly to kazi.run().
    """
    if logger is None:
        logger = Logger()

    # --- Retrieve relevant memory ---
    memory_chunks = _retrieve_and_trim(task_description, logger)

    # --- Build the instructions ---
    # The instructions are the task description enriched with
    # any relevant memory context. Keep it surgical.
    instructions_parts = [task_description]

    if memory_chunks:
        memory_block = _format_memory_block(memory_chunks)
        instructions_parts.append(memory_block)

    if extra_context:
        # Trim extra context to prevent bloat.
        trimmed = extra_context[:_MAX_MEMORY_CHARS]
        instructions_parts.append(f"ADDITIONAL CONTEXT:\n{trimmed}")

    instructions = "\n\n".join(instructions_parts)

    # --- Assemble context package ---
    context = {
        "task_id": task_id,
        "instructions": instructions,
    }

    if allowed_tools:
        context["allowed_tools"] = allowed_tools
    if working_dir:
        context["working_dir"] = working_dir
    if file_paths:
        context["file_paths"] = file_paths

    logger.log(
        task_id, "ctx_pkgr", "package", "ok",
        metadata={
            "memory_chunks": len(memory_chunks),
            "instructions_len": len(instructions),
        },
    )

    return context


def _retrieve_and_trim(
    query: str,
    logger: Logger,
) -> list[dict[str, Any]]:
    """
    Call mem_rtvr and aggressively filter the results.

    Drops chunks below the relevance threshold and trims each chunk
    to _MAX_CHUNK_CHARS.
    """
    raw = mem_rtvr.retrieve(
        query=query,
        top_n=_MAX_MEMORY_CHUNKS * 2,  # Fetch extra, then filter.
        include_dumps=True,
        logger=logger,
    )

    trimmed = []
    total_chars = 0

    for chunk in raw:
        # Skip low-relevance noise.
        if chunk.get("score", 0) < _MIN_RELEVANCE:
            continue

        text = chunk["text"][:_MAX_CHUNK_CHARS]

        # Stop if we'd exceed the total memory budget.
        if total_chars + len(text) > _MAX_MEMORY_CHARS:
            break

        trimmed.append({
            "text": text,
            "source": chunk.get("source", "?"),
            "score": chunk.get("score", 0),
        })
        total_chars += len(text)

        if len(trimmed) >= _MAX_MEMORY_CHUNKS:
            break

    return trimmed


def _format_memory_block(chunks: list[dict[str, Any]]) -> str:
    """
    Format memory chunks into a compact text block for the Kazi prompt.

    No fluff. Just the content and its source.
    """
    lines = ["RELEVANT MEMORY:"]
    for chunk in chunks:
        source = chunk["source"]
        text = chunk["text"].strip()
        # Indent the chunk text under the source label.
        lines.append(f"[{source}]")
        lines.append(text)
        lines.append("")  # Blank line between chunks.

    return "\n".join(lines).rstrip()
