"""
Deterministic workflow inspection and submission CLI.

`python -m mr1.workflow_cli <cmd>`

The CLI NEVER instantiates a scheduler. `submit` writes a workflow
directory to disk and exits; the MR1-owned scheduler picks it up on its
next tick. Read commands load state directly from the store and
pretty-print. `compile-workflow` is the one CLI command that may invoke
the workflow compiler agent.

Facade
------
The implementation lives in the `mr1.cli` package, split by domain:
    mr1.cli.main          — `_build_parser()` and `main()` entry point
    mr1.cli.formatting    — shared formatting/render helpers
    mr1.cli.context       — store/visibility/audit helpers
    mr1.cli.messages      — message/inbox/outbox commands
    mr1.cli.agents        — agent/agents/scope/lifecycle commands
    mr1.cli.memory        — memory/graph/insights/snapshot/doctor commands
    mr1.cli.capabilities  — capability/tool/approval/audit commands
    mr1.cli.workflows     — workflow/jobs/tasks/result/inputs/artifacts/watchers/schema/trigger
    mr1.cli.events        — events + timeline commands

This module remains as a facade re-exporting every public symbol that
existing tests and orchestrator modules import from `mr1.workflow_cli`.
"""

from __future__ import annotations

# Re-export classes / functions accessed externally as
# `workflow_cli.<symbol>` by tests (`@patch("mr1.workflow_cli.run_agent_health")`)
# and by `mr1.orchestrator.root_builtins`.
from mr1.agents import run_agent_health as run_agent_health
from mr1.mrn_loop import MRnStepRunner as MRnStepRunner
from mr1.mrn_run import MRnRunRunner as MRnRunRunner
from mr1.scheduler import submit_spec_to_disk as submit_spec_to_disk
from mr1.workflow_models import Provenance as Provenance

# Re-export the CLI entry point.
from mr1.cli.main import _build_parser as _build_parser, main as main

# Re-export every formatter / handler / context helper. Keeps existing
# `from mr1.workflow_cli import _format_X` and direct `workflow_cli._X`
# call sites in tests and `mr1.orchestrator.root_builtins` working
# unchanged.
from mr1.cli.formatting import (
    _brief_description as _brief_description,
    _brief_schema_view as _brief_schema_view,
    _compact_text as _compact_text,
    _config_shape_for_tool as _config_shape_for_tool,
    _describe_schema_section as _describe_schema_section,
    _format_description_text as _format_description_text,
    _format_schema as _format_schema,
    _label_for_task_id as _label_for_task_id,
    _reject_invalid_flag_combination as _reject_invalid_flag_combination,
    _render_table as _render_table,
    _select_description_view as _select_description_view,
    _short_ts as _short_ts,
)
from mr1.cli.context import (
    _approval_store_for as _approval_store_for,
    _audit_entries_for_agent as _audit_entries_for_agent,
    _event_visible as _event_visible,
    _find_visible_audit_entry as _find_visible_audit_entry,
    _find_workflow_for_task as _find_workflow_for_task,
    _graph_store_for as _graph_store_for,
    _insight_store_for as _insight_store_for,
    _load_json_file as _load_json_file,
    _load_memory_graph as _load_memory_graph,
    _load_scoped_workflow as _load_scoped_workflow,
    _load_text_file as _load_text_file,
    _require_message as _require_message,
    _require_visible_approval as _require_visible_approval,
    _resolve_mailbox_agent_id as _resolve_mailbox_agent_id,
    _runtime_access_for as _runtime_access_for,
    _runtime_root_for as _runtime_root_for,
    _timeline_for as _timeline_for,
    _visible_approvals as _visible_approvals,
    _visible_audit_entries as _visible_audit_entries,
    _visible_timeline_events as _visible_timeline_events,
    _visible_workflows as _visible_workflows,
)
from mr1.cli.messages import (
    _cmd_inbox as _cmd_inbox,
    _cmd_inbox_triage as _cmd_inbox_triage,
    _cmd_message as _cmd_message,
    _cmd_message_archive as _cmd_message_archive,
    _cmd_message_read as _cmd_message_read,
    _cmd_message_send as _cmd_message_send,
    _cmd_outbox as _cmd_outbox,
    _format_inbox_triage_result as _format_inbox_triage_result,
    _format_message_detail as _format_message_detail,
    _format_message_preview_line as _format_message_preview_line,
    _format_messages_table as _format_messages_table,
    _message_preview_payload as _message_preview_payload,
    _pending_parent_messages as _pending_parent_messages,
)
from mr1.cli.agents import (
    _agent_runtime_activity_payload as _agent_runtime_activity_payload,
    _cmd_agent as _cmd_agent,
    _cmd_agent_assign as _cmd_agent_assign,
    _cmd_agent_grant_scope as _cmd_agent_grant_scope,
    _cmd_agent_revoke_scope as _cmd_agent_revoke_scope,
    _cmd_agent_run as _cmd_agent_run,
    _cmd_agent_scopes as _cmd_agent_scopes,
    _cmd_agent_step as _cmd_agent_step,
    _cmd_agents as _cmd_agents,
    _format_agent as _format_agent,
    _format_agents as _format_agents,
    _format_mrn_run_result as _format_mrn_run_result,
    _format_mrn_step_result as _format_mrn_step_result,
    _format_runtime_agent as _format_runtime_agent,
    _format_runtime_agent_health as _format_runtime_agent_health,
    _format_runtime_agents as _format_runtime_agents,
    _format_scope_grants as _format_scope_grants,
    _agent_payload as _agent_payload,
    _print_agent_usage_error as _print_agent_usage_error,
    _summarize_last_action as _summarize_last_action,
)
from mr1.cli.memory import (
    _cmd_doctor as _cmd_doctor,
    _cmd_repair_state as _cmd_repair_state,
    _cmd_memory_agent as _cmd_memory_agent,
    _cmd_memory_capabilities as _cmd_memory_capabilities,
    _cmd_memory_curate as _cmd_memory_curate,
    _cmd_memory_curation_due as _cmd_memory_curation_due,
    _cmd_memory_curation_runs as _cmd_memory_curation_runs,
    _cmd_memory_failures as _cmd_memory_failures,
    _cmd_memory_feedback_due as _cmd_memory_feedback_due,
    _cmd_memory_feedback_insight as _cmd_memory_feedback_insight,
    _cmd_memory_feedback_list as _cmd_memory_feedback_list,
    _cmd_memory_feedback_update as _cmd_memory_feedback_update,
    _cmd_memory_file as _cmd_memory_file,
    _cmd_memory_graph_show as _cmd_memory_graph_show,
    _cmd_memory_insights_effectiveness as _cmd_memory_insights_effectiveness,
    _cmd_memory_insights_failures as _cmd_memory_insights_failures,
    _cmd_memory_insights_friction as _cmd_memory_insights_friction,
    _cmd_memory_insights_list as _cmd_memory_insights_list,
    _cmd_memory_insights_recommendations as _cmd_memory_insights_recommendations,
    _cmd_memory_insights_show as _cmd_memory_insights_show,
    _cmd_memory_maintenance_run as _cmd_memory_maintenance_run,
    _cmd_memory_maintenance_spec as _cmd_memory_maintenance_spec,
    _cmd_memory_maintenance_status as _cmd_memory_maintenance_status,
    _cmd_memory_project as _cmd_memory_project,
    _cmd_memory_reset as _cmd_memory_reset,
    _cmd_memory_retrieval_search as _cmd_memory_retrieval_search,
    _cmd_memory_retrieval_stats as _cmd_memory_retrieval_stats,
    _cmd_memory_retrieval_update as _cmd_memory_retrieval_update,
    _cmd_memory_stats as _cmd_memory_stats,
    _cmd_memory_top_workflows as _cmd_memory_top_workflows,
    _cmd_memory_update as _cmd_memory_update,
    _cmd_memory_workflow_template as _cmd_memory_workflow_template,
    _cmd_snapshot_create as _cmd_snapshot_create,
    _cmd_snapshot_inspect as _cmd_snapshot_inspect,
    _cmd_snapshot_list as _cmd_snapshot_list,
    _format_compile_memory_summary as _format_compile_memory_summary,
    _format_doctor_report as _format_doctor_report,
    _format_memory_capabilities as _format_memory_capabilities,
    _format_memory_curation_due as _format_memory_curation_due,
    _format_memory_curation_runs as _format_memory_curation_runs,
    _format_memory_detail as _format_memory_detail,
    _format_memory_failures as _format_memory_failures,
    _format_memory_feedback as _format_memory_feedback,
    _format_memory_insight as _format_memory_insight,
    _format_memory_insight_effectiveness as _format_memory_insight_effectiveness,
    _format_memory_insights as _format_memory_insights,
    _format_memory_retrieval_search as _format_memory_retrieval_search,
    _format_memory_retrieval_stats as _format_memory_retrieval_stats,
    _format_memory_stats as _format_memory_stats,
    _format_memory_templates as _format_memory_templates,
    _format_memory_update as _format_memory_update,
    _format_snapshot_list as _format_snapshot_list,
    _format_snapshot_manifest as _format_snapshot_manifest,
    _load_insights_filtered as _load_insights_filtered,
    _ordered_snapshot_manifest as _ordered_snapshot_manifest,
)
from mr1.cli.capabilities import (
    _cmd_approvals_decide as _cmd_approvals_decide,
    _cmd_approvals_list as _cmd_approvals_list,
    _cmd_approvals_show as _cmd_approvals_show,
    _cmd_capabilities as _cmd_capabilities,
    _cmd_capability as _cmd_capability,
    _cmd_capability_audit_list as _cmd_capability_audit_list,
    _cmd_capability_audit_show as _cmd_capability_audit_show,
    _cmd_capability_call as _cmd_capability_call,
    _cmd_tool as _cmd_tool,
    _cmd_tools as _cmd_tools,
    _format_approval as _format_approval,
    _format_approval_decision_result as _format_approval_decision_result,
    _format_approvals_table as _format_approvals_table,
    _format_capabilities as _format_capabilities,
    _format_capability as _format_capability,
    _format_capability_audit_detail as _format_capability_audit_detail,
    _format_capability_audit_table as _format_capability_audit_table,
    _format_tool as _format_tool,
    _format_tools as _format_tools,
)
from mr1.cli.workflows import (
    _cmd_append_workflow as _cmd_append_workflow,
    _cmd_artifacts as _cmd_artifacts,
    _cmd_cancel_task as _cmd_cancel_task,
    _cmd_cancel_workflow as _cmd_cancel_workflow,
    _cmd_compile_workflow as _cmd_compile_workflow,
    _cmd_inputs as _cmd_inputs,
    _cmd_insert_workflow as _cmd_insert_workflow,
    _cmd_jobs as _cmd_jobs,
    _cmd_rerun as _cmd_rerun,
    _cmd_replace_workflow as _cmd_replace_workflow,
    _cmd_result as _cmd_result,
    _cmd_schema as _cmd_schema,
    _cmd_submit as _cmd_submit,
    _cmd_task as _cmd_task,
    _cmd_trigger as _cmd_trigger,
    _cmd_watchers as _cmd_watchers,
    _cmd_workflow as _cmd_workflow,
    _cmd_workflows as _cmd_workflows,
    _format_artifacts as _format_artifacts,
    _format_inline_value as _format_inline_value,
    _format_inputs as _format_inputs,
    _format_jobs as _format_jobs,
    _format_result as _format_result,
    _format_task_detail as _format_task_detail,
    _format_watchers as _format_watchers,
    _format_workflow_detail as _format_workflow_detail,
    _format_workflows_table as _format_workflows_table,
)
from mr1.cli.events import (
    _cmd_events as _cmd_events,
    _cmd_timeline_agent as _cmd_timeline_agent,
    _cmd_timeline_approvals as _cmd_timeline_approvals,
    _cmd_timeline_blocked as _cmd_timeline_blocked,
    _cmd_timeline_list as _cmd_timeline_list,
    _cmd_timeline_recent as _cmd_timeline_recent,
    _cmd_timeline_show as _cmd_timeline_show,
    _cmd_timeline_trace as _cmd_timeline_trace,
    _cmd_timeline_workflow as _cmd_timeline_workflow,
    _format_events as _format_events,
    _format_timeline_event_detail as _format_timeline_event_detail,
    _format_timeline_events as _format_timeline_events,
)


if __name__ == "__main__":
    raise SystemExit(main())
