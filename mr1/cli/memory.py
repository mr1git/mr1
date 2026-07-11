"""Memory-related CLI: graph, insights, curation, feedback, retrieval,
snapshots, doctor. All `_cmd_memory_*`, `_cmd_doctor`, and `_cmd_snapshot_*`
handlers plus their formatters live here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.doctor import (
    create_snapshot,
    filter_doctor_report,
    inspect_snapshot,
    list_snapshots,
    repair_state_file,
    run_doctor,
)
from mr1.memory_curator import (
    InsightStore,
    MemoryCurationRun,
    MemoryInsight,
    build_memory_curation_bundle,
    evaluate_memory_curation_due,
    run_memory_curation,
)
from mr1.memory_feedback import (
    InsightFeedback,
    build_memory_maintenance_spec,
    evaluate_memory_feedback_due,
    maintenance_status_payload,
    submit_memory_maintenance_workflow,
    update_insight_feedback,
)
from mr1.memory_graph import (
    MemoryGraph,
    MemoryGraphStore,
    file_summary,
    graph_stats,
    project_summary,
    show_node,
    update_graph_from_events,
    workflow_template_summary,
)
from mr1.memory_queries import (
    list_filtered_insights,
    memory_search,
    memory_graph_agent_summary,
    memory_graph_capabilities,
    memory_graph_failures,
    memory_graph_top_workflows,
    memory_insight_show,
    memory_insights_search,
)
from mr1.memory_reset import MemoryResetResult, reset_memory
from mr1.memory_retrieval import RetrievalStore, update_memory_retrieval
from mr1.scheduler import WorkflowSpecError
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore

from mr1.cli.context import (
    _approval_store_for,
    _graph_store_for,
    _insight_store_for,
    _load_memory_graph,
    _runtime_root_for,
    _timeline_for,
    _visible_workflows,
)
from mr1.cli.formatting import (
    _compact_text,
    _reject_invalid_flag_combination,
    _render_table,
    _short_ts,
)


def _format_memory_update(result: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(result, indent=2, sort_keys=True)
    rows = [
        ("FIELD", "VALUE"),
        ("processed_events", str(result.get("processed_events", 0))),
        ("last_processed_event_index", str(result.get("last_processed_event_index", 0))),
        ("nodes_created", str(result.get("nodes_created", 0))),
        ("nodes_updated", str(result.get("nodes_updated", 0))),
        ("edges_created", str(result.get("edges_created", 0))),
        ("edges_updated", str(result.get("edges_updated", 0))),
    ]
    return _render_table(rows)

def _format_memory_stats(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        f"nodes: {payload.get('node_count', 0)}",
        f"edges: {payload.get('edge_count', 0)}",
        f"last_processed_event_index: {payload.get('last_processed_event_index', 0)}",
        "",
        "node_types:",
    ]
    node_rows = [("TYPE", "COUNT")]
    for name, count in sorted(dict(payload.get("node_types", {})).items()):
        node_rows.append((name, str(count)))
    lines.append(_render_table(node_rows, indent="  "))
    lines.append("")
    lines.append("edge_types:")
    edge_rows = [("TYPE", "COUNT")]
    for name, count in sorted(dict(payload.get("edge_types", {})).items()):
        edge_rows.append((name, str(count)))
    lines.append(_render_table(edge_rows, indent="  "))
    return "\n".join(lines)

def _format_memory_templates(items: list[dict[str, Any]], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(items, indent=2, sort_keys=True)
    if not items:
        return "No workflow templates."
    rows = [("TEMPLATE_ID", "NAME", "CRED", "SUCCESS", "FAIL", "BLOCKED")]
    for item in items:
        stats = item.get("stats", {})
        rows.append((
            item["node_id"],
            item["name"][:24],
            f"{float(stats.get('credibility_score', 0.0)):.3f}",
            str(int(stats.get("success_count", 0))),
            str(int(stats.get("failure_count", 0))),
            str(int(stats.get("blocked_count", 0))),
        ))
    return _render_table(rows)

def _format_memory_capabilities(items: list[dict[str, Any]], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(items, indent=2, sort_keys=True)
    if not items:
        return "No capabilities."
    rows = [("CAPABILITY_ID", "NAME", "RELIAB", "REQUESTS", "EXEC", "BLOCKED", "FAIL")]
    for item in items:
        stats = item.get("stats", {})
        rows.append((
            item["node_id"],
            item["name"][:24],
            f"{float(stats.get('reliability_score', 0.0)):.3f}",
            str(int(stats.get("request_count", 0))),
            str(int(stats.get("execution_count", 0))),
            str(int(stats.get("blocked_count", 0))),
            str(int(stats.get("failure_count", 0))),
        ))
    return _render_table(rows)

def _format_memory_failures(items: list[dict[str, Any]], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(items, indent=2, sort_keys=True)
    if not items:
        return "No failure modes."
    rows = [("FAILURE_ID", "NAME", "COUNT", "STATUS", "TYPE")]
    for item in items:
        stats = item.get("stats", {})
        metadata = item.get("metadata", {})
        rows.append((
            item["node_id"],
            item["name"][:32],
            str(int(stats.get("occurrence_count", 0))),
            str(metadata.get("status") or "-"),
            str(metadata.get("error_type") or "-"),
        ))
    return _render_table(rows)

def _format_memory_detail(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(payload, indent=2, sort_keys=True)

def _format_memory_retrieval_stats(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        f"retrieval_ready: {bool(payload.get('retrieval_ready'))}",
        f"document_count: {int(payload.get('document_count', 0))}",
        f"schema_version: {payload.get('schema_version') or '-'}",
        f"updated_at: {payload.get('updated_at') or '-'}",
        "",
        "doc_types:",
    ]
    doc_rows = [("TYPE", "COUNT")]
    for name, count in sorted(dict(payload.get("doc_type_counts", {})).items()):
        doc_rows.append((name, str(count)))
    lines.append(_render_table(doc_rows, indent="  "))
    lines.append("")
    lines.append("source_counts:")
    source_rows = [("SOURCE", "COUNT")]
    for name, count in sorted(dict(payload.get("source_counts", {})).items()):
        source_rows.append((name, str(count)))
    lines.append(_render_table(source_rows, indent="  "))
    return "\n".join(lines)

def _format_memory_retrieval_search(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    items = list(payload.get("items", []))
    if not items:
        return "No retrieval documents."
    rows = [("DOC_ID", "TYPE", "SCORE", "TITLE", "SUMMARY")]
    for item in items:
        rows.append((
            str(item.get("doc_id") or ""),
            str(item.get("doc_type") or ""),
            f"{float(item.get('score', 0.0)):.2f}",
            str(item.get("title") or "")[:36],
            str(item.get("summary") or "")[:56],
        ))
    return _render_table(rows)

def _format_memory_curation_due(payload: dict[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    rows = [
        ("FIELD", "VALUE"),
        ("due", str(bool(payload.get("due")))),
        ("latest_event_index", str(payload.get("latest_event_index", 0))),
        ("last_curated_event_index", str(payload.get("last_curated_event_index", 0))),
        ("important_event_count", str(payload.get("important_event_count", 0))),
        ("important_event_types", ", ".join(payload.get("important_event_types", [])) or "-"),
        ("suggested_event_window", json.dumps(payload.get("suggested_event_window", []))),
    ]
    return _render_table(rows)

def _format_compile_memory_summary(result: dict[str, Any] | Any) -> str:
    if hasattr(result, "compiled_with_memory"):
        compiled_with_memory = bool(result.compiled_with_memory)
        memory_tools_used = list(result.memory_tools_used or [])
        memory_refs_used = list(result.envelope.memory_refs_used)
        memory_context = result.memory_context_summary
        warnings = list(result.memory_ref_warnings or [])
    else:
        compiled_with_memory = bool(result.get("compiled_with_memory"))
        memory_tools_used = list(result.get("memory_tools_used", []))
        memory_refs_used = list(result.get("memory_refs_used", []))
        memory_context = str(result.get("memory_context_summary") or "")
        warnings = list(result.get("memory_ref_warnings", []))
    lines = [
        f"memory_enabled: {compiled_with_memory}",
        f"memory_tools_used: {', '.join(memory_tools_used) if memory_tools_used else '-'}",
        f"memory_refs_used: {', '.join(memory_refs_used) if memory_refs_used else '-'}",
    ]
    if memory_context:
        lines.append(f"memory_context_summary: {memory_context}")
    if warnings:
        lines.append(f"memory_ref_warnings: {', '.join(warnings)}")
    return "\n".join(lines)

def _format_memory_insights(
    items: list[MemoryInsight],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in items]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not items:
        return "No insights."
    rows = [("INSIGHT_ID", "TYPE", "SEVERITY", "STATUS", "CONF", "UPDATED", "TITLE")]
    for item in items:
        rows.append((
            item.insight_id,
            item.insight_type,
            item.severity,
            item.status,
            f"{float(item.confidence):.2f}",
            _short_ts(item.updated_at),
            item.title[:48],
        ))
    return _render_table(rows)

def _format_memory_insight(
    item: MemoryInsight,
    *,
    json_output: bool = False,
) -> str:
    payload = item.to_dict()
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(payload, indent=2, sort_keys=True)

def _format_memory_curation_runs(
    runs: list[MemoryCurationRun],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in runs]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not runs:
        return "No curation runs."
    rows = [("RUN_ID", "STATUS", "WINDOW", "OUTPUTS", "ERRORS", "STARTED")]
    for item in runs:
        rows.append((
            item.run_id,
            item.status,
            f"{item.event_start_index}-{item.event_end_index}",
            str(len(item.output_insight_ids)),
            str(len(item.errors)),
            _short_ts(item.started_at),
        ))
    return _render_table(rows)

def _format_memory_feedback(
    items: list[InsightFeedback],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in items]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not items:
        return "No feedback."
    rows = [("FEEDBACK_ID", "INSIGHT_ID", "OUTCOME", "DELTA", "EVENT_TYPE", "CREATED")]
    for item in items:
        rows.append((
            item.feedback_id,
            item.insight_id,
            item.outcome,
            f"{float(item.confidence_delta):+.2f}",
            str(item.metadata.get("event_type") or "-"),
            _short_ts(item.created_at),
        ))
    return _render_table(rows)

def _format_memory_insight_effectiveness(
    items: list[MemoryInsight],
    *,
    json_output: bool = False,
) -> str:
    payload = [item.to_dict() for item in items]
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    if not items:
        return "No active or stale insights."
    rows = [("INSIGHT_ID", "STATUS", "CONF", "USED", "POS", "NEG", "NEU", "EFF", "UPDATED")]
    for item in items:
        stats = dict(item.stats)
        rows.append((
            item.insight_id,
            item.status,
            f"{float(item.confidence):.2f}",
            str(int(stats.get("used_count", 0))),
            str(int(stats.get("positive_outcome_count", 0))),
            str(int(stats.get("negative_outcome_count", 0))),
            str(int(stats.get("neutral_outcome_count", 0))),
            f"{float(stats.get('effectiveness_score', 0.0)):.2f}",
            _short_ts(item.updated_at),
        ))
    return _render_table(rows)

def _format_doctor_report(
    report: Any,
    *,
    json_output: bool = False,
    errors_only: bool = False,
) -> str:
    if json_output:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)
    display_report = filter_doctor_report(report, errors_only=errors_only)
    if not display_report.checks:
        return "No matching doctor checks."
    rows = [("STATUS", "CATEGORY", "CHECK", "SUMMARY")]
    for item in display_report.checks:
        rows.append((
            item.status.upper(),
            item.category,
            item.title,
            item.summary,
        ))
    recommendations: list[str] = []
    seen: set[str] = set()
    for item in display_report.checks:
        for recommendation in item.recommendations:
            if recommendation in seen:
                continue
            seen.add(recommendation)
            recommendations.append(recommendation)
    lines = [_render_table(rows)]
    if recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
    return "\n".join(lines)

def _ordered_snapshot_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in (
        "snapshot_id",
        "created_at",
        "source_root",
        "doctor_status_at_creation",
        "latest_event_index",
        "file_count",
        "total_bytes",
        "included_paths",
        "errors",
    ):
        if key in payload:
            ordered[key] = payload[key]
    for key in sorted(set(payload) - set(ordered)):
        ordered[key] = payload[key]
    return ordered

def _format_snapshot_manifest(payload: dict[str, Any], *, json_output: bool = False) -> str:
    ordered = _ordered_snapshot_manifest(payload)
    if json_output:
        return json.dumps(ordered, indent=2, sort_keys=True)
    lines = [
        f"snapshot_id:               {ordered.get('snapshot_id', '-')}",
        f"created_at:                {ordered.get('created_at', '-')}",
        f"source_root:               {ordered.get('source_root', '-')}",
        f"doctor_status_at_creation: {ordered.get('doctor_status_at_creation', '-')}",
        f"latest_event_index:        {ordered.get('latest_event_index', '-')}",
        f"file_count:                {ordered.get('file_count', '-')}",
        f"total_bytes:               {ordered.get('total_bytes', '-')}",
        "included_paths:",
    ]
    included = ordered.get("included_paths") or []
    if included:
        for item in included:
            lines.append(f"  - {item}")
    else:
        lines.append("  -")
    lines.append("errors:")
    errors = ordered.get("errors") or []
    if errors:
        for item in errors:
            lines.append(f"  - {item}")
    else:
        lines.append("  -")
    return "\n".join(lines)

def _format_snapshot_list(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No snapshots."
    rows = [("SNAPSHOT_ID", "CREATED_AT", "DOCTOR_STATUS", "FILES", "BYTES")]
    for item in items:
        rows.append((
            str(item.get("snapshot_id", "-")),
            str(item.get("created_at", "-")),
            str(item.get("doctor_status_at_creation", "-")),
            str(item.get("file_count", "-")),
            str(item.get("total_bytes", "-")),
        ))
    return _render_table(rows)

def _cmd_memory_reset(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    memory_root = _runtime_root_for(store)
    try:
        result = reset_memory(memory_root, scope=args.scope)
    except ValueError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    if args.json:
        print(__import__("json").dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    lines = [f"memory reset: scope={result.scope}"]
    if result.deleted_paths:
        lines.append(f"  deleted ({len(result.deleted_paths)}):")
        for p in result.deleted_paths:
            lines.append(f"    {p}")
    if result.recreated_paths:
        lines.append(f"  recreated ({len(result.recreated_paths)}):")
        for p in result.recreated_paths:
            lines.append(f"    {p}")
    if result.skipped_paths:
        lines.append(f"  skipped ({len(result.skipped_paths)} missing paths)")
    if result.warnings:
        lines.append(f"  warnings ({len(result.warnings)}):")
        for w in result.warnings:
            lines.append(f"    {w}")
    print("\n".join(lines))
    return 0

def _cmd_memory_update(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        result = update_graph_from_events(_timeline_for(store), _graph_store_for(store))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_update(result.to_dict(), json_output=args.json))
    return 0

def _cmd_memory_stats(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, cursor = _load_memory_graph(store)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_stats(graph_stats(graph, last_processed_event_index=cursor), json_output=args.json))
    return 0

def _cmd_memory_graph_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = show_node(graph, args.node_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0

def _cmd_memory_top_workflows(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_graph_top_workflows(store.root.parent, limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_templates(payload["items"], json_output=args.json))
    return 0

def _cmd_memory_capabilities(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_graph_capabilities(store.root.parent, limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_capabilities(payload["items"], json_output=args.json))
    return 0

def _cmd_memory_failures(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_graph_failures(store.root.parent, limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_failures(payload["items"], json_output=args.json))
    return 0

def _cmd_memory_agent(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    if not scoped_agents.is_visible(caller_agent_id, args.agent_id):
        print("error: access denied: agent not in scope", file=sys.stderr)
        return 2
    try:
        payload = memory_graph_agent_summary(store.root.parent, agent_id=args.agent_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not payload["found"]:
        print(f"error: agent not found: {args.agent_id}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload["summary"], json_output=args.json))
    return 0

def _cmd_memory_project(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = project_summary(graph, args.project_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0

def _cmd_memory_file(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = file_summary(graph, args.file_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0

def _cmd_memory_workflow_template(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        graph, _ = _load_memory_graph(store)
        payload = workflow_template_summary(graph, args.template_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0

def _cmd_memory_curation_due(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = evaluate_memory_curation_due(
            _timeline_for(store),
            _insight_store_for(store),
        ).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_curation_due(payload, json_output=args.json))
    return 0

def _cmd_memory_curate(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        run = run_memory_curation(
            event_log=_timeline_for(store),
            graph_store=_graph_store_for(store),
            insight_store=_insight_store_for(store),
            trigger_reason="memory_curate_cli",
            persist_not_due=True,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(run.to_dict(), json_output=args.json))
    return 0

def _load_insights_filtered(
    store: WorkflowStore,
    *,
    insight_types: Optional[set[str]] = None,
    include_statuses: Optional[set[str]] = None,
) -> list[MemoryInsight]:
    return list_filtered_insights(
        store.root.parent,
        insight_types=insight_types,
        include_statuses=include_statuses,
    )

def _cmd_memory_insights_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(store)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0

def _cmd_memory_insights_show(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_insight_show(store.root.parent, insight_id=args.insight_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not payload["found"]:
        print(f"error: insight not found: {args.insight_id}", file=sys.stderr)
        return 2
    insight = MemoryInsight.from_dict(dict(payload["insight"]))
    print(_format_memory_insight(insight, json_output=args.json))
    return 0

def _cmd_memory_insights_recommendations(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(
            store,
            insight_types={"workflow_recommendation", "scope_recommendation", "system_design_lesson"},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0

def _cmd_memory_insights_friction(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(
            store,
            insight_types={"capability_friction", "approval_friction"},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0

def _cmd_memory_insights_failures(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(
            store,
            insight_types={"failure_pattern"},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insights(items, json_output=args.json))
    return 0

def _cmd_memory_insights_effectiveness(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _load_insights_filtered(store, include_statuses={"active", "stale"})
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_insight_effectiveness(items, json_output=args.json))
    return 0

def _cmd_memory_curation_runs(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        runs = _insight_store_for(store).load_runs(limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_curation_runs(runs, json_output=args.json))
    return 0

def _cmd_memory_feedback_due(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = evaluate_memory_feedback_due(
            _timeline_for(store),
            _insight_store_for(store),
            store,
        ).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0

def _cmd_memory_feedback_update(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = update_insight_feedback(
            _timeline_for(store),
            _insight_store_for(store),
            store,
        ).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0

def _cmd_memory_retrieval_update(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = update_memory_retrieval(store.root.parent).to_dict()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_detail(payload, json_output=args.json))
    return 0

def _cmd_memory_retrieval_stats(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    retrieval_store = RetrievalStore(store.root.parent)
    payload: dict[str, Any] = {
        "retrieval_ready": False,
        "document_count": 0,
        "schema_version": None,
        "updated_at": None,
        "doc_type_counts": {},
        "source_counts": {},
    }
    try:
        manifest = retrieval_store.load_manifest() if retrieval_store.manifest_path.exists() else {}
        documents = retrieval_store.load_documents() if retrieval_store.documents_path.exists() else []
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if manifest or documents:
        doc_type_counts: dict[str, int] = {}
        for item in documents:
            doc_type_counts[item.doc_type] = doc_type_counts.get(item.doc_type, 0) + 1
        payload = {
            "retrieval_ready": retrieval_store.exists(),
            "document_count": int(manifest.get("document_count", len(documents))),
            "schema_version": manifest.get("schema_version"),
            "updated_at": manifest.get("updated_at"),
            "doc_type_counts": dict(sorted(doc_type_counts.items())),
            "source_counts": dict(sorted(dict(manifest.get("source_counts", {})).items())),
        }
    print(_format_memory_retrieval_stats(payload, json_output=args.json))
    return 0

def _cmd_memory_retrieval_search(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = memory_search(
            store.root.parent,
            query=args.query,
            limit=args.limit,
            types=list(args.types or []),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_retrieval_search(payload, json_output=args.json))
    return 0

def _cmd_memory_feedback_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _insight_store_for(store).load_feedback(limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_feedback(items, json_output=args.json))
    return 0

def _cmd_memory_feedback_insight(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        items = _insight_store_for(store).load_feedback(limit=args.limit, insight_id=args.insight_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_memory_feedback(items, json_output=args.json))
    return 0

def _cmd_doctor(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        report = run_doctor(_runtime_root_for(store), categories=list(args.categories or []))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_doctor_report(report, json_output=args.json, errors_only=bool(args.errors_only)))
    return 0

def _cmd_snapshot_create(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        manifest = create_snapshot(
            _runtime_root_for(store),
            fail_on_error=bool(args.fail_on_error),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_snapshot_manifest(manifest))
    return 0

def _cmd_snapshot_list(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del args, caller_agent_id, scoped_agents
    print(_format_snapshot_list(list_snapshots(_runtime_root_for(store))))
    return 0

def _cmd_snapshot_inspect(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    try:
        payload = inspect_snapshot(_runtime_root_for(store), args.snapshot_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_snapshot_manifest(payload, json_output=bool(args.json)))
    return 0


def _cmd_repair_state(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    """Quarantine a corrupt MR1 state file so that MR1 can restart cleanly."""
    del caller_agent_id, scoped_agents
    runtime_root = _runtime_root_for(store)
    state_path = Path(getattr(args, "state_path", None) or "") or (
        runtime_root / "active" / "mr1_state.json"
    )
    try:
        result = repair_state_file(state_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result["message"])
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0

def _cmd_memory_maintenance_spec(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del store, caller_agent_id, scoped_agents
    print(json.dumps(build_memory_maintenance_spec(), indent=2, sort_keys=True))
    return 0

def _cmd_memory_maintenance_run(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id
    try:
        workflow_id = submit_memory_maintenance_workflow(store, scoped_agent_store=scoped_agents)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"workflow_id": workflow_id}, indent=2, sort_keys=True))
    else:
        print(workflow_id)
    return 0

def _cmd_memory_maintenance_status(
    args: argparse.Namespace,
    store: WorkflowStore,
    caller_agent_id: str,
    scoped_agents: PersistentAgentStore,
) -> int:
    del caller_agent_id, scoped_agents
    print(_format_memory_detail(maintenance_status_payload(store), json_output=args.json))
    return 0
