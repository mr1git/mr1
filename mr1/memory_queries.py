"""
Shared read-only memory query helpers for CLI, capabilities, and compiler prefetch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mr1.memory_retrieval import (
    LexicalMemoryRetriever,
    RETRIEVAL_DOC_TYPES,
    RetrievalStore,
    update_memory_retrieval,
)

DEFAULT_MEMORY_LIMIT = 5
DEFAULT_GRAPH_CAPABILITY_LIMIT = 10
DEFAULT_GRAPH_FAILURE_LIMIT = 10
MEMORY_CAPABILITY_NAMES = [
    "memory_search",
    "memory_insights_search",
    "memory_insight_show",
    "memory_graph_top_workflows",
    "memory_graph_capabilities",
    "memory_graph_failures",
    "memory_graph_agent_summary",
]


def _memory_root(runtime_root: Optional[Path]) -> Path:
    if runtime_root is None:
        return Path(__file__).resolve().parent / "memory"
    return Path(runtime_root)


def _insight_store(runtime_root: Optional[Path]) -> InsightStore:
    from mr1.memory_curator import InsightStore

    return InsightStore(_memory_root(runtime_root) / "insights")


def _graph_store(runtime_root: Optional[Path]) -> MemoryGraphStore:
    from mr1.memory_graph import MemoryGraphStore

    return MemoryGraphStore(_memory_root(runtime_root) / "graph")


def memory_stores_available(runtime_root: Optional[Path]) -> bool:
    root = _memory_root(runtime_root)
    return any((
        (root / "retrieval" / "documents.jsonl").exists(),
        (root / "retrieval" / "manifest.json").exists(),
        (root / "insights" / "insights.json").exists(),
        (root / "insights" / "feedback.jsonl").exists(),
        (root / "graph" / "graph.json").exists(),
        (root / "events" / "events.jsonl").exists(),
    ))


def _load_insights_or_empty(runtime_root: Optional[Path]) -> dict[str, MemoryInsight]:
    store = _insight_store(runtime_root)
    if not store.insights_path.exists():
        return {}
    return store.load_insights()


def _load_graph_or_empty(runtime_root: Optional[Path]) -> MemoryGraph:
    from mr1.memory_graph import MemoryGraph

    store = _graph_store(runtime_root)
    if not store.graph_path.exists():
        return MemoryGraph()
    return store.load_graph()


def _load_graph_cursor_or_zero(runtime_root: Optional[Path]) -> int:
    store = _graph_store(runtime_root)
    if not store.cursor_path.exists():
        return 0
    return store.load_cursor()


def _validate_limit(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("limit must be an integer >= 0")
    if value < 0:
        raise ValueError("limit must be an integer >= 0")
    return value


def _validate_memory_search_types(types: Any) -> Optional[list[str]]:
    if types is None:
        return None
    if not isinstance(types, list) or any(not isinstance(item, str) or not item.strip() for item in types):
        raise ValueError("types must be a list of non-empty strings")
    if not types:
        return None
    normalized = [_normalized_text(item) for item in types]
    invalid = sorted(item for item in normalized if item not in RETRIEVAL_DOC_TYPES)
    if invalid:
        raise ValueError(f"invalid retrieval document types: {', '.join(invalid)}")
    return normalized


def _validate_types(types: Any) -> Optional[list[str]]:
    from mr1.memory_curator import _ALLOWED_INSIGHT_TYPES

    if types is None:
        return None
    if not isinstance(types, list) or any(not isinstance(item, str) or not item.strip() for item in types):
        raise ValueError("types must be a list of non-empty strings")
    normalized = [item.strip() for item in types]
    invalid = sorted(item for item in normalized if item not in _ALLOWED_INSIGHT_TYPES)
    if invalid:
        raise ValueError(f"invalid insight types: {', '.join(invalid)}")
    return normalized


def _validate_status(status: Any, *, default: str = "active") -> str:
    from mr1.memory_curator import _ALLOWED_STATUSES

    if status is None:
        return default
    if not isinstance(status, str) or not status.strip():
        raise ValueError("status must be a non-empty string")
    normalized = status.strip()
    if normalized not in _ALLOWED_STATUSES:
        raise ValueError(f"invalid status: {normalized}")
    return normalized


def _validate_agent_id(agent_id: Any) -> str:
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("agent_id must be a non-empty string")
    return agent_id.strip()


def _validate_insight_id(insight_id: Any) -> str:
    if not isinstance(insight_id, str) or not insight_id.strip():
        raise ValueError("insight_id must be a non-empty string")
    return insight_id.strip()


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def memory_search(
    runtime_root: Optional[Path],
    *,
    query: Any,
    limit: Any = DEFAULT_MEMORY_LIMIT,
    types: Any = None,
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    validated_limit = _validate_limit(limit, default=DEFAULT_MEMORY_LIMIT)
    validated_types = _validate_memory_search_types(types)
    root = _memory_root(runtime_root)
    retrieval_store = RetrievalStore(root)
    updated_now = False
    update_errors: list[str] = []
    retrieval_ready = retrieval_store.exists()
    if retrieval_ready:
        try:
            retrieval_store.load_documents()
        except ValueError:
            retrieval_ready = False
    if not retrieval_ready:
        result = update_memory_retrieval(root)
        updated_now = True
        update_errors = list(result.errors)
        try:
            retrieval_store.load_documents()
            retrieval_ready = retrieval_store.exists()
        except ValueError:
            retrieval_ready = False
    if not retrieval_ready:
        return {
            "items": [],
            "count": 0,
            "query": query.strip(),
            "types": list(validated_types or []),
            "retrieval_ready": False,
            "updated_now": updated_now,
            "update_errors": update_errors,
        }
    retriever = LexicalMemoryRetriever(root)
    items = [
        item.to_dict()
        for item in retriever.search(
            query=query.strip(),
            limit=validated_limit,
            types=validated_types,
        )
    ]
    return {
        "items": items,
        "count": len(items),
        "query": query.strip(),
        "types": list(validated_types or []),
        "retrieval_ready": True,
        "updated_now": updated_now,
        "update_errors": update_errors,
    }


def _insight_text_fields(insight: MemoryInsight) -> list[str]:
    return [
        _normalized_text(insight.title),
        _normalized_text(insight.summary),
        _normalized_text(insight.recommended_action),
        _normalized_text(json.dumps(insight.metadata, sort_keys=True)),
    ]


def _compact_insight(insight: MemoryInsight) -> dict[str, Any]:
    return {
        "insight_id": insight.insight_id,
        "insight_type": insight.insight_type,
        "title": insight.title,
        "summary": insight.summary,
        "confidence": float(insight.confidence),
        "severity": insight.severity,
        "recommended_action": insight.recommended_action,
        "related_nodes": list(insight.related_nodes),
    }


def list_filtered_insights(
    runtime_root: Optional[Path],
    *,
    insight_types: Optional[set[str]] = None,
    include_statuses: Optional[set[str]] = None,
) -> list[MemoryInsight]:
    statuses = include_statuses or {"active"}
    items = [
        insight
        for insight in _load_insights_or_empty(runtime_root).values()
        if insight.status in statuses
        and (insight_types is None or insight.insight_type in insight_types)
    ]
    items.sort(key=lambda item: (item.updated_at, item.insight_id), reverse=True)
    return items


def memory_insights_search(
    runtime_root: Optional[Path],
    *,
    query: Any = None,
    types: Any = None,
    status: Any = "active",
    limit: Any = DEFAULT_MEMORY_LIMIT,
) -> dict[str, Any]:
    validated_query = ""
    if query is not None:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        validated_query = query.strip()
    validated_types = _validate_types(types)
    validated_status = _validate_status(status, default="active")
    validated_limit = _validate_limit(limit, default=DEFAULT_MEMORY_LIMIT)
    requested_types = set(validated_types or [])
    items = list_filtered_insights(
        runtime_root,
        insight_types=requested_types if validated_types is not None else None,
        include_statuses={validated_status},
    )
    normalized_query = _normalized_text(validated_query)
    matched: list[tuple[MemoryInsight, int, int, int]] = []
    for insight in items:
        fields = _insight_text_fields(insight)
        exact_match = 0
        substring_match = 0
        if normalized_query:
            exact_match = 1 if any(field == normalized_query for field in fields) else 0
            substring_match = 1 if any(normalized_query in field for field in fields) else 0
            if not exact_match and not substring_match:
                continue
        matched.append((
            insight,
            exact_match,
            substring_match,
            int(not requested_types or insight.insight_type in requested_types),
        ))
    matched.sort(key=lambda item: item[0].insight_id)
    matched.sort(key=lambda item: item[0].updated_at, reverse=True)
    matched.sort(key=lambda item: float(item[0].confidence), reverse=True)
    matched.sort(key=lambda item: item[3], reverse=True)
    matched.sort(key=lambda item: item[2], reverse=True)
    matched.sort(key=lambda item: item[1], reverse=True)
    selected = [_compact_insight(insight) for insight, _, _, _ in matched[:validated_limit]]
    return {
        "items": selected,
        "count": len(selected),
        "query": validated_query,
        "types": list(validated_types or []),
        "status": validated_status,
        "limit": validated_limit,
    }


def memory_insight_show(runtime_root: Optional[Path], *, insight_id: Any) -> dict[str, Any]:
    validated_insight_id = _validate_insight_id(insight_id)
    insight = _load_insights_or_empty(runtime_root).get(validated_insight_id)
    return {
        "found": insight is not None,
        "insight": insight.to_dict() if insight is not None else None,
    }


def memory_graph_top_workflows(runtime_root: Optional[Path], *, limit: Any = DEFAULT_MEMORY_LIMIT) -> dict[str, Any]:
    from mr1.memory_graph import top_workflow_templates

    validated_limit = _validate_limit(limit, default=DEFAULT_MEMORY_LIMIT)
    items = top_workflow_templates(_load_graph_or_empty(runtime_root), limit=validated_limit)
    return {"items": items, "count": len(items)}


def memory_graph_capabilities(runtime_root: Optional[Path], *, limit: Any = DEFAULT_GRAPH_CAPABILITY_LIMIT) -> dict[str, Any]:
    from mr1.memory_graph import capability_stats

    validated_limit = _validate_limit(limit, default=DEFAULT_GRAPH_CAPABILITY_LIMIT)
    items = capability_stats(_load_graph_or_empty(runtime_root), limit=validated_limit)
    return {"items": items, "count": len(items)}


def memory_graph_failures(runtime_root: Optional[Path], *, limit: Any = DEFAULT_GRAPH_FAILURE_LIMIT) -> dict[str, Any]:
    from mr1.memory_graph import failure_modes

    validated_limit = _validate_limit(limit, default=DEFAULT_GRAPH_FAILURE_LIMIT)
    items = failure_modes(_load_graph_or_empty(runtime_root), limit=validated_limit)
    return {"items": items, "count": len(items)}


def memory_graph_agent_summary(runtime_root: Optional[Path], *, agent_id: Any) -> dict[str, Any]:
    from mr1.memory_graph import agent_summary

    validated_agent_id = _validate_agent_id(agent_id)
    graph = _load_graph_or_empty(runtime_root)
    try:
        summary = agent_summary(graph, validated_agent_id)
    except ValueError:
        summary = None
    return {
        "found": summary is not None,
        "summary": summary,
    }


def known_memory_refs(runtime_root: Optional[Path]) -> set[str]:
    insights = set(_load_insights_or_empty(runtime_root))
    graph = _load_graph_or_empty(runtime_root)
    retrieval_docs: set[str] = set()
    try:
        retrieval_docs = {
            item.doc_id
            for item in RetrievalStore(_memory_root(runtime_root)).load_documents()
        }
    except ValueError:
        retrieval_docs = set()
    return insights | set(graph.nodes) | retrieval_docs


def memory_context_summary(prefetched_context: dict[str, Any]) -> str:
    retrieval = len(list(prefetched_context.get("memory_search", {}).get("items", [])))
    insights = len(list(prefetched_context.get("memory_insights_search", {}).get("items", [])))
    workflows = len(list(prefetched_context.get("memory_graph_top_workflows", {}).get("items", [])))
    capabilities = len(list(prefetched_context.get("memory_graph_capabilities", {}).get("items", [])))
    failures = len(list(prefetched_context.get("memory_graph_failures", {}).get("items", [])))
    agent_found = bool(prefetched_context.get("memory_graph_agent_summary", {}).get("found"))
    return (
        f"retrieval={retrieval}, "
        f"insights={insights}, "
        f"top_workflows={workflows}, "
        f"capabilities={capabilities}, "
        f"failures={failures}, "
        f"agent_summary={'present' if agent_found else 'missing'}"
    )


def memory_graph_last_processed_event_index(runtime_root: Optional[Path]) -> int:
    return _load_graph_cursor_or_zero(runtime_root)
