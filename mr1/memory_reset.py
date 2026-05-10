"""
Deterministic memory reset service for root MR1.

Resets only MR1 memory artifacts (active context, graph, insights, RAG/retrieval/dumps).
Never touches protected runtime records: agents, messages, workflows, approvals,
the event timeline, or snapshots.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ResetScope = Literal["active", "graph", "insights", "rag", "all"]

_VALID_SCOPES: frozenset[str] = frozenset({"active", "graph", "insights", "rag", "all"})


@dataclass
class MemoryResetResult:
    scope: str
    deleted_paths: list[str] = field(default_factory=list)
    recreated_paths: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "deleted_paths": self.deleted_paths,
            "recreated_paths": self.recreated_paths,
            "skipped_paths": self.skipped_paths,
            "warnings": self.warnings,
        }


def reset_memory(memory_root: Path, scope: ResetScope) -> MemoryResetResult:
    """Reset MR1 memory artifacts for the given scope.

    Raises ValueError for an unknown scope.
    All file-system errors are caught; missing paths are recorded in skipped_paths.
    """
    if scope not in _VALID_SCOPES:
        raise ValueError(f"unknown reset scope: {scope!r}; valid: {sorted(_VALID_SCOPES)}")

    result = MemoryResetResult(scope=scope)

    if scope == "all":
        for sub_scope in ("active", "graph", "insights", "rag"):
            sub = _dispatch(memory_root, sub_scope)  # type: ignore[arg-type]
            result.deleted_paths.extend(sub.deleted_paths)
            result.recreated_paths.extend(sub.recreated_paths)
            result.skipped_paths.extend(sub.skipped_paths)
            result.warnings.extend(sub.warnings)
        return result

    return _dispatch(memory_root, scope)


def _dispatch(memory_root: Path, scope: str) -> MemoryResetResult:
    result = MemoryResetResult(scope=scope)
    if scope == "active":
        _reset_active(memory_root, result)
    elif scope == "graph":
        _reset_graph(memory_root, result)
    elif scope == "insights":
        _reset_insights(memory_root, result)
    elif scope == "rag":
        _reset_rag(memory_root, result)
    return result


# ---------------------------------------------------------------------------
# Per-scope helpers
# ---------------------------------------------------------------------------

def _reset_active(memory_root: Path, result: MemoryResetResult) -> None:
    active_dir = memory_root / "active"

    # Delete mr1_state.json — MR1 reinitialises it with a fresh session_id on next boot
    state_path = active_dir / "mr1_state.json"
    if state_path.exists():
        _safe_delete_file(state_path, result)
    else:
        result.skipped_paths.append(str(state_path))

    # Delete mr1_context.md if present
    context_path = active_dir / "mr1_context.md"
    if context_path.exists():
        _safe_delete_file(context_path, result)
    else:
        result.skipped_paths.append(str(context_path))

    # Clear mr1_turns/ contents (keep the directory)
    turns_dir = active_dir / "mr1_turns"
    if turns_dir.is_dir():
        _safe_clear_dir(turns_dir, result)
    else:
        result.skipped_paths.append(str(turns_dir))


def _reset_graph(memory_root: Path, result: MemoryResetResult) -> None:
    graph_dir = memory_root / "graph"
    for filename in ("graph.json", "cursor.json"):
        path = graph_dir / filename
        if path.exists():
            _safe_delete_file(path, result)
        else:
            result.skipped_paths.append(str(path))


def _reset_insights(memory_root: Path, result: MemoryResetResult) -> None:
    insights_dir = memory_root / "insights"
    for filename in (
        "insights.json",
        "cursor.json",
        "curation_runs.jsonl",
        "feedback.jsonl",
        "feedback_cursor.json",
    ):
        path = insights_dir / filename
        if path.exists():
            _safe_delete_file(path, result)
        else:
            result.skipped_paths.append(str(path))


def _reset_rag(memory_root: Path, result: MemoryResetResult) -> None:
    # ChromaDB vector store
    chroma_dir = memory_root / "rag" / ".chroma"
    if chroma_dir.exists():
        _safe_rmtree(chroma_dir, result)
    else:
        result.skipped_paths.append(str(chroma_dir))

    # Distilled dump files
    dumps_dir = memory_root / "dumps"
    if dumps_dir.is_dir():
        count = 0
        for path in list(dumps_dir.iterdir()):
            if path.suffix in {".md", ".txt"} and path.is_file():
                _safe_delete_file(path, result)
                count += 1
        if count == 0:
            result.skipped_paths.append(str(dumps_dir / "*.md / *.txt"))
    else:
        result.skipped_paths.append(str(dumps_dir))

    # Lexical retrieval index
    retrieval_dir = memory_root / "retrieval"
    for filename in ("documents.jsonl", "manifest.json"):
        path = retrieval_dir / filename
        if path.exists():
            _safe_delete_file(path, result)
        else:
            result.skipped_paths.append(str(path))


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_delete_file(path: Path, result: MemoryResetResult) -> None:
    try:
        path.unlink()
        result.deleted_paths.append(str(path))
    except OSError as exc:
        result.warnings.append(f"could not delete {path}: {exc}")


def _safe_rmtree(path: Path, result: MemoryResetResult) -> None:
    try:
        shutil.rmtree(path)
        result.deleted_paths.append(str(path))
    except OSError as exc:
        result.warnings.append(f"could not remove {path}: {exc}")


def _safe_clear_dir(dir_path: Path, result: MemoryResetResult) -> None:
    for child in list(dir_path.iterdir()):
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            result.deleted_paths.append(str(child))
        except OSError as exc:
            result.warnings.append(f"could not remove {child}: {exc}")


