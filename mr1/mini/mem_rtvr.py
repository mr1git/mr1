"""
mem_rtvr — Memory Retrieval Agent
==================================
Simple wrapper: takes a query string, searches memory/rag/ using chromadb
for vector similarity, and also checks memory/dumps/ for recent plain text.
Returns top N relevant chunks.

No LLM calls. No subprocess spawning. Pure retrieval.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chromadb

from mr1.core import Logger


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _PKG_ROOT / "memory" / "rag"
_DUMPS_DIR = _PKG_ROOT / "memory" / "dumps"
_CHROMA_DIR = _RAG_DIR / ".chroma"

# Default number of results to return.
_DEFAULT_TOP_N = 5

# Maximum number of recent dump files to scan.
_MAX_RECENT_DUMPS = 10


# ---------------------------------------------------------------------------
# ChromaDB singleton — lazily initialised, reused across calls.
# ---------------------------------------------------------------------------
_chroma_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    """Get or create the persistent chromadb collection."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    _collection = _chroma_client.get_or_create_collection(
        name="mr1_memory",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ---------------------------------------------------------------------------
# Ingestion — called by mem_dltr to add chunks to the RAG store.
# ---------------------------------------------------------------------------

def ingest_chunks(
    chunks: list[dict[str, str]],
    logger: Optional[Logger] = None,
) -> int:
    """
    Add text chunks to the RAG vector store.

    Each chunk dict must have:
        id       (str) — unique identifier
        text     (str) — the content to embed and index
        source   (str) — origin label (e.g. "dump-20260401", "task-xyz")

    Returns the number of chunks successfully added.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c.get("source", "unknown"),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        for c in chunks
    ]

    # Upsert so re-ingesting the same chunk ID just updates it.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    if logger:
        logger.log(
            "mem_rtvr", "mem_rtvr", "ingest", "ok",
            metadata={"count": len(ids)},
        )

    return len(ids)


# ---------------------------------------------------------------------------
# Retrieval — the main public API.
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_n: int = _DEFAULT_TOP_N,
    include_dumps: bool = True,
    logger: Optional[Logger] = None,
) -> list[dict[str, Any]]:
    """
    Search memory for chunks relevant to the query.

    Returns a list of result dicts sorted by relevance:
        [{"text": "...", "source": "...", "score": 0.87, "type": "rag"|"dump"}, ...]

    Searches:
      1. memory/rag/ via chromadb vector similarity
      2. memory/dumps/ via simple substring matching on recent files
    """
    results: list[dict[str, Any]] = []

    # --- RAG search ---
    results.extend(_search_rag(query, top_n))

    # --- Dumps search ---
    if include_dumps:
        results.extend(_search_dumps(query, top_n))

    # Sort by score descending, take top_n overall.
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    results = results[:top_n]

    if logger:
        logger.log(
            "mem_rtvr", "mem_rtvr", "retrieve", "ok",
            metadata={"query": query[:100], "results": len(results)},
        )

    return results


def _search_rag(query: str, top_n: int) -> list[dict[str, Any]]:
    """Search the chromadb vector store."""
    collection = _get_collection()

    # If the collection is empty, query would fail.
    if collection.count() == 0:
        return []

    # Clamp n_results to collection size.
    n = min(top_n, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Chromadb cosine distance: 0 = identical, 2 = opposite.
        # Convert to a 0-1 similarity score.
        score = max(0.0, 1.0 - (dist / 2.0))
        hits.append({
            "text": doc,
            "source": meta.get("source", "rag"),
            "score": round(score, 4),
            "type": "rag",
        })

    return hits


def _search_dumps(query: str, top_n: int) -> list[dict[str, Any]]:
    """
    Search recent plain-text dump files for query terms.

    Uses simple case-insensitive substring matching. Not fancy, but fast
    and sufficient for recent memories that haven't been RAG-indexed yet.
    """
    if not _DUMPS_DIR.exists():
        return []

    # Get dump files sorted newest first.
    dump_files = sorted(
        _DUMPS_DIR.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:_MAX_RECENT_DUMPS]

    # Also check .txt files.
    txt_files = sorted(
        _DUMPS_DIR.glob("*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:_MAX_RECENT_DUMPS]

    all_files = sorted(
        dump_files + txt_files,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:_MAX_RECENT_DUMPS]

    query_lower = query.lower()
    query_terms = query_lower.split()
    hits = []

    for path in all_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        content_lower = content.lower()

        # Score: fraction of query terms found in the file.
        matched = sum(1 for term in query_terms if term in content_lower)
        if matched == 0:
            continue

        score = matched / len(query_terms) if query_terms else 0.0

        # Extract the most relevant paragraph (first one containing a match).
        paragraphs = content.split("\n\n")
        best_para = content[:500]  # Fallback: first 500 chars.
        for para in paragraphs:
            if any(term in para.lower() for term in query_terms):
                best_para = para.strip()[:500]
                break

        hits.append({
            "text": best_para,
            "source": f"dump:{path.name}",
            "score": round(score * 0.7, 4),  # Discount vs RAG results.
            "type": "dump",
        })

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_n]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def collection_stats() -> dict[str, Any]:
    """Return basic stats about the RAG collection."""
    collection = _get_collection()
    return {
        "count": collection.count(),
        "name": collection.name,
        "chroma_dir": str(_CHROMA_DIR),
    }
