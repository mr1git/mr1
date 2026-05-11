"""CLI entry point: parser construction and `main()`.

`python -m mr1.workflow_cli ...` and `python main.py ...` both end up
here. The `_build_parser()` function wires every sub-command's
`set_defaults(func=_cmd_X)` so dispatch is data-driven.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from mr1.event_log import bind_correlation_id, cli_correlation_id
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_store import WorkflowStore

# Handler imports — every `_cmd_X` function used in `set_defaults(func=...)`
# by _build_parser. Grouped by domain for readability.
from mr1.cli.agents import (
    _cmd_agent,
    _cmd_agent_assign,
    _cmd_agent_grant_scope,
    _cmd_agent_revoke_scope,
    _cmd_agent_run,
    _cmd_agent_scopes,
    _cmd_agent_step,
    _cmd_agents,
)
from mr1.cli.capabilities import (
    _cmd_approvals_decide,
    _cmd_approvals_list,
    _cmd_approvals_show,
    _cmd_capabilities,
    _cmd_capability,
    _cmd_capability_audit_list,
    _cmd_capability_audit_show,
    _cmd_capability_call,
    _cmd_tool,
    _cmd_tools,
)
from mr1.cli.events import (
    _cmd_events,
    _cmd_timeline_agent,
    _cmd_timeline_approvals,
    _cmd_timeline_blocked,
    _cmd_timeline_list,
    _cmd_timeline_recent,
    _cmd_timeline_show,
    _cmd_timeline_trace,
    _cmd_timeline_workflow,
)
from mr1.cli.memory import (
    _cmd_doctor,
    _cmd_memory_agent,
    _cmd_memory_capabilities,
    _cmd_memory_curate,
    _cmd_memory_curation_due,
    _cmd_memory_curation_runs,
    _cmd_memory_failures,
    _cmd_memory_feedback_due,
    _cmd_memory_feedback_insight,
    _cmd_memory_feedback_list,
    _cmd_memory_feedback_update,
    _cmd_memory_file,
    _cmd_memory_graph_show,
    _cmd_memory_insights_effectiveness,
    _cmd_memory_insights_failures,
    _cmd_memory_insights_friction,
    _cmd_memory_insights_list,
    _cmd_memory_insights_recommendations,
    _cmd_memory_insights_show,
    _cmd_memory_maintenance_run,
    _cmd_memory_maintenance_spec,
    _cmd_memory_maintenance_status,
    _cmd_memory_project,
    _cmd_memory_reset,
    _cmd_memory_retrieval_search,
    _cmd_memory_retrieval_stats,
    _cmd_memory_retrieval_update,
    _cmd_memory_stats,
    _cmd_memory_top_workflows,
    _cmd_memory_update,
    _cmd_memory_workflow_template,
    _cmd_snapshot_create,
    _cmd_snapshot_inspect,
    _cmd_snapshot_list,
)
from mr1.cli.messages import (
    _cmd_inbox,
    _cmd_inbox_triage,
    _cmd_message,
    _cmd_message_archive,
    _cmd_message_read,
    _cmd_message_send,
    _cmd_outbox,
)
from mr1.cli.workflows import (
    _cmd_append_workflow,
    _cmd_artifacts,
    _cmd_cancel_task,
    _cmd_cancel_workflow,
    _cmd_compile_workflow,
    _cmd_inputs,
    _cmd_insert_workflow,
    _cmd_jobs,
    _cmd_rerun,
    _cmd_replace_workflow,
    _cmd_result,
    _cmd_schema,
    _cmd_submit,
    _cmd_task,
    _cmd_trigger,
    _cmd_watchers,
    _cmd_workflow,
    _cmd_workflows,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mr1.workflow_cli",
        description="Submit and inspect MR1 workflows without an LLM in the loop.",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=None,
        help="Override the workflow store root (defaults to mr1/memory/workflows).",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(subparser: argparse.ArgumentParser, *, include_example: bool) -> None:
        subparser.add_argument("--json", action="store_true", dest="json")
        subparser.add_argument("--brief", action="store_true", dest="brief")
        if include_example:
            subparser.add_argument("--example", action="store_true", dest="example")
        else:
            subparser.set_defaults(example=False)

    p_submit = subs.add_parser("submit", help="Write a workflow spec to the store.")
    p_submit.add_argument("path", help="Path to a workflow JSON spec.")
    p_submit.set_defaults(func=_cmd_submit)

    p_compile = subs.add_parser(
        "compile-workflow",
        help="Compile a workflow request file into a validated envelope.",
    )
    p_compile.add_argument("path", help="Path to a text file containing the workflow request.")
    p_compile.add_argument("--owner", default=None)
    p_compile.add_argument("--submit", action="store_true")
    p_compile.add_argument("--use-memory", action="store_true")
    p_compile.add_argument("--no-memory", action="store_true")
    p_compile.add_argument("--show-memory", action="store_true")
    p_compile.add_argument("--memory-limit", type=int, default=5)
    p_compile.set_defaults(func=_cmd_compile_workflow)

    p_rerun = subs.add_parser("rerun", help="Rerun one task in a workflow.")
    p_rerun.add_argument("workflow_id")
    p_rerun.add_argument("task_label_or_id")
    p_rerun.set_defaults(func=_cmd_rerun)

    p_list = subs.add_parser("workflows", help="List all workflows.")
    p_list.set_defaults(func=_cmd_workflows)

    p_wf = subs.add_parser("workflow", help="Show one workflow's detail.")
    p_wf.add_argument("workflow_id")
    p_wf.set_defaults(func=_cmd_workflow)

    p_task = subs.add_parser("task", help="Show one task's detail.")
    p_task.add_argument("task_id")
    p_task.set_defaults(func=_cmd_task)

    p_cancel_task = subs.add_parser("cancel-task", help="Cancel one task.")
    p_cancel_task.add_argument("task_id")
    p_cancel_task.set_defaults(func=_cmd_cancel_task)

    p_cancel_workflow = subs.add_parser("cancel-workflow", help="Cancel one workflow.")
    p_cancel_workflow.add_argument("workflow_id")
    p_cancel_workflow.set_defaults(func=_cmd_cancel_workflow)

    p_jobs = subs.add_parser("jobs", help="List live tasks across all workflows.")
    p_jobs.set_defaults(func=_cmd_jobs)

    p_events = subs.add_parser("events", help="Show events for a workflow.")
    p_events.add_argument("workflow_id")
    p_events.add_argument("--since", default=None)
    p_events.add_argument("--until", default=None)
    p_events.add_argument("--task", default=None, dest="task")
    p_events.add_argument("--limit", type=int, default=None)
    p_events.set_defaults(func=_cmd_events)

    p_doctor = subs.add_parser("doctor", help="Run deterministic runtime health checks.")
    p_doctor.add_argument("--category", action="append", dest="categories", default=[])
    p_doctor.add_argument("--errors-only", action="store_true")
    add_common_flags(p_doctor, include_example=False)
    p_doctor.set_defaults(func=_cmd_doctor)

    p_snapshot = subs.add_parser("snapshot", help="Create and inspect runtime snapshots.")
    snapshot_subs = p_snapshot.add_subparsers(dest="snapshot_command", required=True)

    p_snapshot_create = snapshot_subs.add_parser("create", help="Create a read-only runtime snapshot.")
    p_snapshot_create.add_argument("--fail-on-error", action="store_true")
    p_snapshot_create.set_defaults(func=_cmd_snapshot_create)

    p_snapshot_list = snapshot_subs.add_parser("list", help="List complete runtime snapshots.")
    p_snapshot_list.set_defaults(func=_cmd_snapshot_list)

    p_snapshot_inspect = snapshot_subs.add_parser("inspect", help="Inspect one runtime snapshot manifest.")
    p_snapshot_inspect.add_argument("snapshot_id")
    p_snapshot_inspect.add_argument("--json", action="store_true", dest="json")
    p_snapshot_inspect.set_defaults(func=_cmd_snapshot_inspect)

    p_timeline = subs.add_parser("timeline", help="Inspect the unified runtime timeline.")
    timeline_subs = p_timeline.add_subparsers(dest="timeline_command", required=True)

    p_timeline_list = timeline_subs.add_parser("list", help="List visible timeline events.")
    p_timeline_list.add_argument("--limit", type=int, default=None)
    add_common_flags(p_timeline_list, include_example=False)
    p_timeline_list.set_defaults(func=_cmd_timeline_list)

    p_timeline_recent = timeline_subs.add_parser("recent", help="Show recent timeline events.")
    p_timeline_recent.add_argument("--limit", type=int, default=20)
    add_common_flags(p_timeline_recent, include_example=False)
    p_timeline_recent.set_defaults(func=_cmd_timeline_recent)

    p_timeline_show = timeline_subs.add_parser("show", help="Show one timeline event.")
    p_timeline_show.add_argument("event_id")
    add_common_flags(p_timeline_show, include_example=False)
    p_timeline_show.set_defaults(func=_cmd_timeline_show)

    p_timeline_trace = timeline_subs.add_parser("trace", help="Trace one correlation id.")
    p_timeline_trace.add_argument("correlation_id")
    add_common_flags(p_timeline_trace, include_example=False)
    p_timeline_trace.set_defaults(func=_cmd_timeline_trace)

    p_timeline_blocked = timeline_subs.add_parser("blocked", help="Show currently blocked timeline items.")
    add_common_flags(p_timeline_blocked, include_example=False)
    p_timeline_blocked.set_defaults(func=_cmd_timeline_blocked)

    p_timeline_approvals = timeline_subs.add_parser("approvals", help="Show approval lifecycle events.")
    add_common_flags(p_timeline_approvals, include_example=False)
    p_timeline_approvals.set_defaults(func=_cmd_timeline_approvals)

    p_timeline_agent = timeline_subs.add_parser("agent", help="Show timeline events related to one agent.")
    p_timeline_agent.add_argument("agent_id")
    add_common_flags(p_timeline_agent, include_example=False)
    p_timeline_agent.set_defaults(func=_cmd_timeline_agent)

    p_timeline_workflow = timeline_subs.add_parser("workflow", help="Show timeline events for one workflow.")
    p_timeline_workflow.add_argument("workflow_id")
    add_common_flags(p_timeline_workflow, include_example=False)
    p_timeline_workflow.set_defaults(func=_cmd_timeline_workflow)

    p_watchers = subs.add_parser("watchers", help="List active watcher tasks.")
    p_watchers.set_defaults(func=_cmd_watchers)

    p_capabilities = subs.add_parser("capabilities", help="List registered capabilities.")
    add_common_flags(p_capabilities, include_example=False)
    p_capabilities.set_defaults(func=_cmd_capabilities)

    p_capability = subs.add_parser("capability", help="Show one capability description.")
    p_capability.add_argument("name")
    add_common_flags(p_capability, include_example=True)
    p_capability.set_defaults(func=_cmd_capability)

    p_capability_call = subs.add_parser(
        "capability-call",
        help="Invoke a direct-callable capability once and print the result.",
    )
    p_capability_call.add_argument("name", help="Capability name.")
    p_capability_call.add_argument("config_file", help="Path to a JSON config file.")
    p_capability_call.set_defaults(func=_cmd_capability_call)

    p_tools = subs.add_parser("tools", help="List registered deterministic workflow tools.")
    add_common_flags(p_tools, include_example=False)
    p_tools.set_defaults(func=_cmd_tools)

    p_agents = subs.add_parser("agents", help="List persistent scoped agents.")
    add_common_flags(p_agents, include_example=False)
    p_agents.set_defaults(func=_cmd_agents)

    p_schema = subs.add_parser("schema", help="Show workflow schema metadata.")
    p_schema.add_argument("section", nargs="?")
    add_common_flags(p_schema, include_example=False)
    p_schema.set_defaults(func=_cmd_schema)

    p_tool = subs.add_parser("tool", help="Show one tool description.")
    p_tool.add_argument("tool_type")
    add_common_flags(p_tool, include_example=True)
    p_tool.set_defaults(func=_cmd_tool)

    p_agent = subs.add_parser("agent", help="Manage scoped agents or inspect runtime agent profiles.")
    p_agent.add_argument("parts", nargs="+")
    add_common_flags(p_agent, include_example=False)
    p_agent.set_defaults(func=_cmd_agent)

    p_agent_assign = subs.add_parser("agent-assign", help="Assign a mission file to a persistent scoped agent.")
    p_agent_assign.add_argument("agent_id")
    p_agent_assign.add_argument("mission_file")
    p_agent_assign.set_defaults(func=_cmd_agent_assign)

    p_agent_step = subs.add_parser("agent-step", help="Run one bounded MRn step for a persistent scoped agent.")
    p_agent_step.add_argument("agent_id")
    p_agent_step.set_defaults(func=_cmd_agent_step)

    p_agent_run = subs.add_parser("agent-run", help="Run a bounded multi-step MRn run for a persistent scoped agent.")
    p_agent_run.add_argument("agent_id")
    p_agent_run.add_argument("--steps", type=int, default=3)
    p_agent_run.add_argument("--max-workflows", type=int, default=2)
    p_agent_run.add_argument("--no-stop-on-waiting", action="store_true")
    p_agent_run.add_argument("--stop-on-idle", action="store_true")
    p_agent_run.add_argument("--no-stop-on-workflow-running", action="store_true")
    p_agent_run.add_argument("--allow-action", action="append", default=[])
    p_agent_run.add_argument("--no-confirm-workflows", action="store_true")
    p_agent_run.add_argument("--max-runtime-s", type=int, default=None)
    p_agent_run.set_defaults(func=_cmd_agent_run)

    p_agent_grant_scope = subs.add_parser("agent-grant-scope", help="Grant one normalized scope root to an agent.")
    p_agent_grant_scope.add_argument("agent_id")
    p_agent_grant_scope.add_argument("path")
    p_agent_grant_scope.add_argument("--reason", default="scope grant")
    p_agent_grant_scope.set_defaults(func=_cmd_agent_grant_scope)

    p_agent_revoke_scope = subs.add_parser("agent-revoke-scope", help="Revoke one normalized scope root from an agent.")
    p_agent_revoke_scope.add_argument("agent_id")
    p_agent_revoke_scope.add_argument("path")
    p_agent_revoke_scope.set_defaults(func=_cmd_agent_revoke_scope)

    p_agent_scopes = subs.add_parser("agent-scopes", help="Show effective scope roots and grant provenance for an agent.")
    p_agent_scopes.add_argument("agent_id")
    add_common_flags(p_agent_scopes, include_example=False)
    p_agent_scopes.set_defaults(func=_cmd_agent_scopes)

    p_approvals = subs.add_parser("approvals", help="List capability approval requests.")
    approvals_subs = p_approvals.add_subparsers(dest="approvals_command", required=True)

    p_approvals_list = approvals_subs.add_parser("list", help="List visible approval requests.")
    add_common_flags(p_approvals_list, include_example=False)
    p_approvals_list.set_defaults(func=_cmd_approvals_list)

    p_approvals_show = approvals_subs.add_parser("show", help="Show one approval request.")
    p_approvals_show.add_argument("approval_request_id")
    add_common_flags(p_approvals_show, include_example=False)
    p_approvals_show.set_defaults(func=_cmd_approvals_show)

    p_approvals_approve = approvals_subs.add_parser("approve", help="Approve one pending approval request.")
    p_approvals_approve.add_argument("approval_request_id")
    p_approvals_approve.add_argument("--grant-scope", action="store_true")
    p_approvals_approve.add_argument("--reason", default="approved")
    p_approvals_approve.set_defaults(func=_cmd_approvals_decide, decision="approved")

    p_approvals_deny = approvals_subs.add_parser("deny", help="Deny one pending approval request.")
    p_approvals_deny.add_argument("approval_request_id")
    p_approvals_deny.add_argument("--reason", default="denied")
    p_approvals_deny.set_defaults(func=_cmd_approvals_decide, decision="denied", grant_scope=False)

    p_capability_audit = subs.add_parser("capability-audit", help="Inspect indexed capability audit records.")
    capability_audit_subs = p_capability_audit.add_subparsers(dest="capability_audit_command", required=True)

    p_capability_audit_list = capability_audit_subs.add_parser("list", help="List visible capability audits.")
    p_capability_audit_list.add_argument("--agent", dest="agent_id", default=None)
    add_common_flags(p_capability_audit_list, include_example=False)
    p_capability_audit_list.set_defaults(func=_cmd_capability_audit_list)

    p_capability_audit_show = capability_audit_subs.add_parser("show", help="Show one capability audit record.")
    p_capability_audit_show.add_argument("audit_id")
    add_common_flags(p_capability_audit_show, include_example=False)
    p_capability_audit_show.set_defaults(func=_cmd_capability_audit_show)

    p_memory = subs.add_parser("memory", help="Inspect deterministic graph memory.")
    memory_subs = p_memory.add_subparsers(dest="memory_command", required=True)

    p_memory_reset = memory_subs.add_parser("reset", help="Reset one or all memory scopes.")
    p_memory_reset.add_argument(
        "scope",
        choices=["active", "graph", "insights", "rag", "all"],
        help="Memory scope to reset.",
    )
    add_common_flags(p_memory_reset, include_example=False)
    p_memory_reset.set_defaults(func=_cmd_memory_reset)

    p_memory_update = memory_subs.add_parser("update", help="Update graph memory from timeline events.")
    add_common_flags(p_memory_update, include_example=False)
    p_memory_update.set_defaults(func=_cmd_memory_update)

    p_memory_stats = memory_subs.add_parser("stats", help="Show graph memory counts and cursor state.")
    add_common_flags(p_memory_stats, include_example=False)
    p_memory_stats.set_defaults(func=_cmd_memory_stats)

    p_memory_due = memory_subs.add_parser("curation-due", help="Check whether memory curation is due.")
    add_common_flags(p_memory_due, include_example=False)
    p_memory_due.set_defaults(func=_cmd_memory_curation_due)

    p_memory_curate = memory_subs.add_parser("curate", help="Run one bounded memory curation pass.")
    add_common_flags(p_memory_curate, include_example=False)
    p_memory_curate.set_defaults(func=_cmd_memory_curate)

    p_memory_retrieval = memory_subs.add_parser("retrieval", help="Inspect unified retrieval documents.")
    memory_retrieval_subs = p_memory_retrieval.add_subparsers(dest="memory_retrieval_command", required=True)

    p_memory_retrieval_update = memory_retrieval_subs.add_parser("update", help="Rebuild unified retrieval documents.")
    add_common_flags(p_memory_retrieval_update, include_example=False)
    p_memory_retrieval_update.set_defaults(func=_cmd_memory_retrieval_update)

    p_memory_retrieval_stats = memory_retrieval_subs.add_parser("stats", help="Show retrieval manifest and document counts.")
    add_common_flags(p_memory_retrieval_stats, include_example=False)
    p_memory_retrieval_stats.set_defaults(func=_cmd_memory_retrieval_stats)

    p_memory_retrieval_search = memory_retrieval_subs.add_parser("search", help="Search unified retrieval documents.")
    p_memory_retrieval_search.add_argument("query")
    p_memory_retrieval_search.add_argument("--limit", type=int, default=5)
    p_memory_retrieval_search.add_argument("--type", action="append", dest="types", default=[])
    add_common_flags(p_memory_retrieval_search, include_example=False)
    p_memory_retrieval_search.set_defaults(func=_cmd_memory_retrieval_search)

    p_memory_insights = memory_subs.add_parser("insights", help="Inspect curated memory insights.")
    memory_insights_subs = p_memory_insights.add_subparsers(dest="memory_insights_command", required=True)

    p_memory_insights_list = memory_insights_subs.add_parser("list", help="List active insights.")
    add_common_flags(p_memory_insights_list, include_example=False)
    p_memory_insights_list.set_defaults(func=_cmd_memory_insights_list)

    p_memory_insights_show = memory_insights_subs.add_parser("show", help="Show one insight.")
    p_memory_insights_show.add_argument("insight_id")
    add_common_flags(p_memory_insights_show, include_example=False)
    p_memory_insights_show.set_defaults(func=_cmd_memory_insights_show)

    p_memory_insights_recommendations = memory_insights_subs.add_parser(
        "recommendations",
        help="List recommendation-oriented insights.",
    )
    add_common_flags(p_memory_insights_recommendations, include_example=False)
    p_memory_insights_recommendations.set_defaults(func=_cmd_memory_insights_recommendations)

    p_memory_insights_friction = memory_insights_subs.add_parser(
        "friction",
        help="List capability and approval friction insights.",
    )
    add_common_flags(p_memory_insights_friction, include_example=False)
    p_memory_insights_friction.set_defaults(func=_cmd_memory_insights_friction)

    p_memory_insights_failures = memory_insights_subs.add_parser(
        "failures",
        help="List failure-pattern insights.",
    )
    add_common_flags(p_memory_insights_failures, include_example=False)
    p_memory_insights_failures.set_defaults(func=_cmd_memory_insights_failures)

    p_memory_insights_effectiveness = memory_insights_subs.add_parser(
        "effectiveness",
        help="Show active and stale insight effectiveness stats.",
    )
    add_common_flags(p_memory_insights_effectiveness, include_example=False)
    p_memory_insights_effectiveness.set_defaults(func=_cmd_memory_insights_effectiveness)

    p_memory_runs = memory_subs.add_parser("curation-runs", help="List recent memory curation runs.")
    p_memory_runs.add_argument("--limit", type=int, default=20)
    add_common_flags(p_memory_runs, include_example=False)
    p_memory_runs.set_defaults(func=_cmd_memory_curation_runs)

    p_memory_feedback = memory_subs.add_parser("feedback", help="Inspect insight feedback.")
    memory_feedback_subs = p_memory_feedback.add_subparsers(dest="memory_feedback_command", required=True)

    p_memory_feedback_due = memory_feedback_subs.add_parser("due", help="Check whether insight feedback is due.")
    add_common_flags(p_memory_feedback_due, include_example=False)
    p_memory_feedback_due.set_defaults(func=_cmd_memory_feedback_due)

    p_memory_feedback_update = memory_feedback_subs.add_parser("update", help="Run one insight feedback update pass.")
    add_common_flags(p_memory_feedback_update, include_example=False)
    p_memory_feedback_update.set_defaults(func=_cmd_memory_feedback_update)

    p_memory_feedback_list = memory_feedback_subs.add_parser("list", help="List recent insight feedback.")
    p_memory_feedback_list.add_argument("--limit", type=int, default=20)
    add_common_flags(p_memory_feedback_list, include_example=False)
    p_memory_feedback_list.set_defaults(func=_cmd_memory_feedback_list)

    p_memory_feedback_insight = memory_feedback_subs.add_parser("insight", help="List feedback for one insight.")
    p_memory_feedback_insight.add_argument("insight_id")
    p_memory_feedback_insight.add_argument("--limit", type=int, default=20)
    add_common_flags(p_memory_feedback_insight, include_example=False)
    p_memory_feedback_insight.set_defaults(func=_cmd_memory_feedback_insight)

    p_memory_maintenance = memory_subs.add_parser("maintenance", help="Build or submit the memory maintenance workflow.")
    memory_maintenance_subs = p_memory_maintenance.add_subparsers(dest="memory_maintenance_command", required=True)

    p_memory_maintenance_spec = memory_maintenance_subs.add_parser("spec", help="Print the memory maintenance workflow spec.")
    add_common_flags(p_memory_maintenance_spec, include_example=False)
    p_memory_maintenance_spec.set_defaults(func=_cmd_memory_maintenance_spec)

    p_memory_maintenance_run = memory_maintenance_subs.add_parser("run", help="Submit the memory maintenance workflow.")
    add_common_flags(p_memory_maintenance_run, include_example=False)
    p_memory_maintenance_run.set_defaults(func=_cmd_memory_maintenance_run)

    p_memory_maintenance_status = memory_maintenance_subs.add_parser("status", help="Show the latest memory maintenance workflow status.")
    add_common_flags(p_memory_maintenance_status, include_example=False)
    p_memory_maintenance_status.set_defaults(func=_cmd_memory_maintenance_status)

    p_memory_graph = memory_subs.add_parser("graph", help="Inspect graph memory views.")
    memory_graph_subs = p_memory_graph.add_subparsers(dest="memory_graph_command", required=True)

    p_memory_graph_show = memory_graph_subs.add_parser("show", help="Show one graph node and its incident edges.")
    p_memory_graph_show.add_argument("node_id")
    add_common_flags(p_memory_graph_show, include_example=False)
    p_memory_graph_show.set_defaults(func=_cmd_memory_graph_show)

    p_memory_top = memory_graph_subs.add_parser("top-workflows", help="Show top workflow templates.")
    p_memory_top.add_argument("--limit", type=int, default=10)
    add_common_flags(p_memory_top, include_example=False)
    p_memory_top.set_defaults(func=_cmd_memory_top_workflows)

    p_memory_caps = memory_graph_subs.add_parser("capabilities", help="Show capability graph stats.")
    p_memory_caps.add_argument("--limit", type=int, default=None)
    add_common_flags(p_memory_caps, include_example=False)
    p_memory_caps.set_defaults(func=_cmd_memory_capabilities)

    p_memory_failures = memory_graph_subs.add_parser("failures", help="Show common failure modes.")
    p_memory_failures.add_argument("--limit", type=int, default=10)
    add_common_flags(p_memory_failures, include_example=False)
    p_memory_failures.set_defaults(func=_cmd_memory_failures)

    p_memory_agent = memory_graph_subs.add_parser("agent", help="Show one agent graph summary.")
    p_memory_agent.add_argument("agent_id")
    add_common_flags(p_memory_agent, include_example=False)
    p_memory_agent.set_defaults(func=_cmd_memory_agent)

    p_memory_project = memory_graph_subs.add_parser("project", help="Show one project graph summary.")
    p_memory_project.add_argument("project_id")
    add_common_flags(p_memory_project, include_example=False)
    p_memory_project.set_defaults(func=_cmd_memory_project)

    p_memory_file = memory_graph_subs.add_parser("file", help="Show one file graph summary.")
    p_memory_file.add_argument("file_id")
    add_common_flags(p_memory_file, include_example=False)
    p_memory_file.set_defaults(func=_cmd_memory_file)

    p_memory_template = memory_graph_subs.add_parser("workflow-template", help="Show one workflow template summary.")
    p_memory_template.add_argument("template_id")
    add_common_flags(p_memory_template, include_example=False)
    p_memory_template.set_defaults(func=_cmd_memory_workflow_template)

    p_inbox = subs.add_parser("inbox", help="List inbox messages for an agent.")
    p_inbox.add_argument("--agent", default=None)
    p_inbox.add_argument("--archived", action="store_true")
    add_common_flags(p_inbox, include_example=False)
    p_inbox.set_defaults(func=_cmd_inbox)

    p_inbox_triage = subs.add_parser("inbox-triage", help="Run one bounded root inbox triage pass.")
    p_inbox_triage.add_argument("--max-messages", type=int, default=10)
    p_inbox_triage.add_argument("--max-actions", type=int, default=3)
    p_inbox_triage.set_defaults(func=_cmd_inbox_triage)

    p_outbox = subs.add_parser("outbox", help="List outbox messages for an agent.")
    p_outbox.add_argument("--agent", default=None)
    p_outbox.add_argument("--archived", action="store_true")
    add_common_flags(p_outbox, include_example=False)
    p_outbox.set_defaults(func=_cmd_outbox)

    p_message = subs.add_parser("message", help="Show one message.")
    p_message.add_argument("message_id")
    add_common_flags(p_message, include_example=False)
    p_message.set_defaults(func=_cmd_message)

    p_message_read = subs.add_parser("message-read", help="Mark one message as read.")
    p_message_read.add_argument("message_id")
    p_message_read.set_defaults(func=_cmd_message_read)

    p_message_archive = subs.add_parser("message-archive", help="Archive one message.")
    p_message_archive.add_argument("message_id")
    p_message_archive.set_defaults(func=_cmd_message_archive)

    p_message_send = subs.add_parser("message-send", help="Send one persistent message.")
    p_message_send.add_argument("to_agent_id")
    p_message_send.add_argument("subject")
    p_message_send.add_argument("body_file")
    p_message_send.add_argument(
        "--kind",
        choices=["report", "question", "alert", "status", "request"],
        default="request",
    )
    p_message_send.set_defaults(func=_cmd_message_send)

    p_result = subs.add_parser("result", help="Show normalized task output.")
    p_result.add_argument("task_id")
    p_result.set_defaults(func=_cmd_result)

    p_inputs = subs.add_parser("inputs", help="Show materialized task inputs.")
    p_inputs.add_argument("task_id")
    p_inputs.set_defaults(func=_cmd_inputs)

    p_artifacts = subs.add_parser("artifacts", help="List artifacts for a workflow.")
    p_artifacts.add_argument("workflow_id")
    p_artifacts.set_defaults(func=_cmd_artifacts)

    p_trigger = subs.add_parser("trigger", help="Trigger a manual_event watcher.")
    p_trigger.add_argument("workflow_id")
    p_trigger.add_argument("label_or_task_id")
    p_trigger.add_argument("event_name", nargs="?")
    p_trigger.set_defaults(func=_cmd_trigger)

    p_append_workflow = subs.add_parser("append-workflow", help="Append task(s) to a workflow.")
    p_append_workflow.add_argument("workflow_id")
    p_append_workflow.add_argument("path")
    p_append_workflow.set_defaults(func=_cmd_append_workflow)

    p_insert_workflow = subs.add_parser("insert-workflow", help="Insert one task after an existing task.")
    p_insert_workflow.add_argument("workflow_id")
    p_insert_workflow.add_argument("after_task")
    p_insert_workflow.add_argument("path")
    p_insert_workflow.set_defaults(func=_cmd_insert_workflow)

    p_replace_workflow = subs.add_parser("replace-workflow", help="Replace one task in a workflow.")
    p_replace_workflow.add_argument("-r", "--rerun", action="store_true")
    p_replace_workflow.add_argument("workflow_id")
    p_replace_workflow.add_argument("task_label_or_id")
    p_replace_workflow.add_argument("path")
    p_replace_workflow.set_defaults(func=_cmd_replace_workflow)

    return parser


def main(
    argv: Optional[list[str]] = None,
    *,
    store: Optional[WorkflowStore] = None,
    caller_agent_id: Optional[str] = None,
    scoped_agent_store: Optional[PersistentAgentStore] = None,
    message_store: Optional[MessageStore] = None,
    workflow_compiler: Optional[Any] = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    active_store = store if store is not None else WorkflowStore(root=args.store_root)
    active_scoped_store = scoped_agent_store or PersistentAgentStore(
        root=active_store.root.parent / "agents"
    )
    active_message_store = message_store or MessageStore(
        root=active_store.root.parent / "messages",
        scoped_agent_store=active_scoped_store,
    )
    resolved_caller_agent_id = caller_agent_id or active_scoped_store.root_agent_id
    setattr(args, "workflow_compiler", workflow_compiler)
    setattr(args, "message_store", active_message_store)
    command_name = " ".join(
        item for item in [
            getattr(args, "command", None),
            getattr(args, "snapshot_command", None),
            getattr(args, "timeline_command", None),
            getattr(args, "approvals_command", None),
            getattr(args, "capability_audit_command", None),
            getattr(args, "memory_command", None),
            getattr(args, "memory_feedback_command", None),
            getattr(args, "memory_insights_command", None),
            getattr(args, "memory_maintenance_command", None),
            getattr(args, "memory_graph_command", None),
            getattr(args, "memory_retrieval_command", None),
        ]
        if item
    )
    with bind_correlation_id(cli_correlation_id(resolved_caller_agent_id, command_name or "cli")):
        return args.func(args, active_store, resolved_caller_agent_id, active_scoped_store)
