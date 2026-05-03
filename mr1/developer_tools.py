"""
Safe developer tools for inspecting MR1 system state.

These tools are read-only and designed to help developers understand
workflow execution, memory organization, and capability approvals.

Tools:
1. Workflow State Inspector - inspects workflow and task execution states
2. Memory Graph Navigator - navigates memory graph relationships
3. Capability Call Tracer - traces capability approvals and calls
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Set

from mr1.tools import (
    ToolConfigError,
    ToolResult,
    ToolRunner,
    _write_text_artifact,
)
from mr1.workflow_models import Task, Workflow, TERMINAL_TASK_STATUSES, FAILED_TASK_STATUSES
from mr1.workflow_store import WorkflowStore

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================================
# WorkflowInspector: Lightweight workflow file parser and DAG analyzer
# ============================================================================


class WorkflowInspector:
    """Parse and analyze YAML/JSON workflow files with dependency graph inspection.

    Provides read-only access to workflow structure without side effects.
    Detects circular dependencies, extracts output references, and produces
    JSON-serializable summaries suitable for debugging and analysis.

    Example:
        inspector = WorkflowInspector("workflow.yaml")
        workflow = inspector.load()
        dag = inspector.parse_dag()
        summary = inspector.get_summary()

        # Output:
        # {
        #   "tasks": [{"label": "validate", "type": "script", "status": "pending"}],
        #   "dag": {"validate": [], "deploy": ["validate"]},
        #   "task_count": 2,
        #   "output_refs": ["$task.deploy.outputs.status"]
        # }
    """

    class WorkflowError(Exception):
        """Base exception for workflow loading and parsing errors."""

    class CyclicDependencyError(WorkflowError):
        """Raised when workflow contains circular dependencies."""

    def __init__(self, workflow_path: str):
        """Initialize with path to YAML/JSON workflow file.

        Args:
            workflow_path: Path to workflow file (.yaml, .yml, or .json)

        Raises:
            WorkflowError: If path is invalid when load() is called
        """
        self.workflow_path = Path(workflow_path)
        self.workflow_data: Optional[dict[str, Any]] = None
        self.tasks: dict[str, dict[str, Any]] = {}
        self.dag: dict[str, list[str]] = {}

    def load(self) -> dict[str, Any]:
        """Load and parse workflow file.

        Supports YAML and JSON formats. Detects format from file extension,
        falls back to content-based detection.

        Returns:
            Parsed workflow dictionary

        Raises:
            WorkflowError: If file not found, parse fails, or content is empty
        """
        if not self.workflow_path.exists():
            raise self.WorkflowError(f"Workflow file not found: {self.workflow_path}")

        try:
            content = self.workflow_path.read_text(encoding="utf-8")

            # Try JSON first (faster), then YAML
            try:
                self.workflow_data = json.loads(content)
            except json.JSONDecodeError:
                if yaml is None:
                    raise self.WorkflowError(
                        "YAML parsing requires 'pyyaml' package. Install with: pip install pyyaml"
                    )
                self.workflow_data = yaml.safe_load(content)

            if not isinstance(self.workflow_data, dict):
                raise self.WorkflowError("Workflow root must be a dictionary")

            if not self.workflow_data:
                raise self.WorkflowError("Workflow file is empty")

            self.tasks = self._extract_tasks()
            return self.workflow_data

        except (json.JSONDecodeError, yaml.YAMLError if yaml else Exception) as e:
            raise self.WorkflowError(f"Failed to parse workflow: {str(e)}")
        except Exception as e:
            raise self.WorkflowError(f"Error loading workflow: {str(e)}")

    def parse_dag(self) -> dict[str, list[str]]:
        """Parse depends_on field into adjacency list structure.

        Converts task dependency declarations into a directed acyclic graph
        (DAG) represented as {task_label: [dependency_labels]}.

        Returns:
            Adjacency dict mapping task labels to their dependencies

        Raises:
            WorkflowError: If workflow not loaded
            CyclicDependencyError: If circular dependencies detected

        Example:
            dag = inspector.parse_dag()
            # {"validate": [], "deploy": ["validate"], "test": ["validate"]}
        """
        if not self.tasks:
            raise self.WorkflowError("Workflow not loaded. Call load() first.")

        # Build adjacency list from depends_on field
        self.dag = {}
        for label, task in self.tasks.items():
            depends_on = task.get("depends_on", [])

            # Normalize to list
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            elif not isinstance(depends_on, list):
                depends_on = []

            # Filter out empty strings
            self.dag[label] = [d for d in depends_on if d]

        # Detect cycles
        self._detect_cycles()
        return self.dag

    def get_task_status(self, label: str) -> str:
        """Return status for named task.

        Args:
            label: Task label/name from workflow

        Returns:
            Status string ("pending", "running", "succeeded", "failed", etc.)
            Returns "unknown" if task not found
        """
        if label not in self.tasks:
            return "unknown"
        return self.tasks[label].get("status", "pending")

    def get_summary(self) -> dict[str, Any]:
        """Return comprehensive summary of workflow structure.

        Includes task list with metadata, dependency graph, task count,
        and discovered output references. All outputs are JSON-serializable.

        Returns:
            Dict with keys:
            - tasks: list of {label, type, status}
            - dag: adjacency dict of dependencies
            - task_count: integer count of tasks
            - output_refs: sorted list of output references found

        Raises:
            WorkflowError: If workflow not loaded
            CyclicDependencyError: If circular dependencies detected
        """
        if not self.tasks:
            raise self.WorkflowError("Workflow not loaded. Call load() first.")

        # Ensure DAG is parsed
        if not self.dag:
            self.parse_dag()

        return {
            "tasks": [
                {
                    "label": label,
                    "type": task.get("type"),
                    "status": task.get("status", "pending"),
                }
                for label, task in self.tasks.items()
            ],
            "dag": self.dag,
            "task_count": len(self.tasks),
            "output_refs": self._find_output_refs(),
        }

    def _extract_tasks(self) -> dict[str, dict[str, Any]]:
        """Extract tasks from workflow data.

        Handles various workflow formats:
        - {tasks: [{label, ...}, ...]}
        - {steps: [{name, ...}, ...]}
        """
        tasks = {}

        # Try common task list keys
        task_list = self.workflow_data.get("tasks") or self.workflow_data.get("steps") or []

        for task in task_list:
            if isinstance(task, dict):
                # Try common task identifier keys
                label = task.get("label") or task.get("name") or task.get("id")
                if label:
                    tasks[label] = task

        return tasks

    def _find_output_refs(self) -> list[str]:
        """Find all output references in workflow.

        Searches for patterns like $task.outputs.key, ${task.outputs},
        or similar output reference syntax.
        """
        refs: Set[str] = set()

        def collect(obj: Any) -> None:
            if isinstance(obj, dict):
                for value in obj.values():
                    collect(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)
            elif isinstance(obj, str) and "$" in obj and "output" in obj:
                refs.add(obj)

        for task in self.tasks.values():
            collect(task)

        return sorted(list(refs))

    def _detect_cycles(self) -> None:
        """Detect circular dependencies using DFS with recursion stack.

        Raises:
            CyclicDependencyError: With detected cycle path information
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.dag.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    raise self.CyclicDependencyError(
                        f"Circular dependency detected: {node} -> {neighbor}"
                    )

            rec_stack.discard(node)

        for task in self.dag:
            if task not in visited:
                dfs(task)


# ============================================================================
# Tool 1: Workflow State Inspector
# ============================================================================

class WorkflowStateInspectorTool:
    """Inspects workflow and task execution states for debugging and analysis."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration."""
        workflow_id = config.get("workflow_id")
        if not isinstance(workflow_id, str):
            raise ToolConfigError(
                "workflow_state_inspector requires workflow_id (string) in tool_config"
            )

        include_tasks = config.get("include_tasks", True)
        if not isinstance(include_tasks, bool):
            raise ToolConfigError("workflow_state_inspector include_tasks must be boolean")

        task_filter = config.get("task_filter")
        if task_filter is not None and not isinstance(task_filter, str):
            raise ToolConfigError(
                "workflow_state_inspector task_filter must be string or omitted"
            )

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """
        Inspect a workflow's execution state.

        Returns:
        - Overall workflow status
        - Task breakdown by status
        - Failed/blocked task details
        - Execution timeline
        """
        target_workflow_id = task.tool_config.get("workflow_id")
        include_tasks = task.tool_config.get("include_tasks", True)
        task_filter = task.tool_config.get("task_filter")

        try:
            # Load target workflow
            target_workflow = store.load_workflow(target_workflow_id)
            if not target_workflow:
                return ToolResult(
                    state="failed",
                    summary="Workflow not found",
                    text=f"Workflow {target_workflow_id} does not exist",
                    error="not_found",
                )

            # Collect state summary
            result = self._inspect_workflow(target_workflow, include_tasks, task_filter)

            # Write report
            artifact = _write_text_artifact(
                store=store,
                task=task,
                name=f"workflow_state_{target_workflow_id[:12]}",
                text=result["report"],
            )

            return ToolResult(
                state="succeeded",
                summary=result["summary"],
                text=result["report"],
                data=result["data"],
                artifacts=[artifact] if artifact else [],
            )

        except Exception as e:
            return ToolResult(
                state="failed",
                summary=f"Inspection failed: {str(e)}",
                text=str(e),
                error="execution_error",
            )

    def _inspect_workflow(
        self, workflow: Workflow, include_tasks: bool, task_filter: Optional[str]
    ) -> dict[str, Any]:
        """Analyze workflow state."""
        lines = []
        lines.append(f"# Workflow State: {workflow.workflow_id}")
        lines.append(f"**Status**: {workflow.status.value}")
        lines.append(f"**Created**: {workflow.created_at}")
        lines.append(f"**Finished**: {workflow.finished_at or 'running'}")
        lines.append("")

        # Task summary
        task_counts = self._count_task_statuses(workflow.tasks)
        lines.append("## Task Summary")
        for status, count in sorted(task_counts.items()):
            lines.append(f"- {status}: {count}")
        lines.append("")

        # Task details if requested
        if include_tasks:
            lines.append("## Tasks")
            filtered_tasks = workflow.tasks
            if task_filter:
                filtered_tasks = [t for t in workflow.tasks if task_filter.lower() in t.label.lower()]

            for t in filtered_tasks:
                lines.append(f"### {t.label}")
                lines.append(f"- ID: {t.task_id}")
                lines.append(f"- Type: {t.task_kind}")
                lines.append(f"- Status: {t.status.value}")
                if t.is_terminal():
                    duration = self._duration_str(t.started_at, t.finished_at)
                    lines.append(f"- Duration: {duration}")
                if t.status.value in FAILED_TASK_STATUSES and t.last_error:
                    lines.append(f"- Error: {t.last_error}")
                if t.blocked_reason:
                    lines.append(f"- Blocked: {t.blocked_reason}")
                lines.append("")

        report = "\n".join(lines)

        return {
            "summary": f"Workflow {workflow.workflow_id}: {workflow.status.value} with {sum(task_counts.values())} tasks",
            "data": task_counts,
            "report": report,
        }

    def _count_task_statuses(self, tasks: list[Task]) -> dict[str, int]:
        """Count tasks by status."""
        counts = {}
        for t in tasks:
            status = t.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _duration_str(self, started_at: Optional[str], finished_at: Optional[str]) -> str:
        """Format duration between timestamps."""
        if not started_at or not finished_at:
            return "unknown"
        try:
            from datetime import datetime, timezone
            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(finished_at)
            delta = end - start
            return f"{delta.total_seconds():.1f}s"
        except Exception:
            return "parse error"


# ============================================================================
# Tool 2: Memory Graph Navigator
# ============================================================================

class MemoryGraphNavigatorTool:
    """Navigate and inspect memory graph relationships."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration."""
        query_type = config.get("query_type")
        valid_types = {"relationships", "stats", "find"}
        if query_type not in valid_types:
            raise ToolConfigError(
                f"memory_graph_navigator query_type must be one of {valid_types}"
            )

        if query_type == "find":
            search_term = config.get("search_term")
            if not isinstance(search_term, str) or not search_term:
                raise ToolConfigError(
                    "memory_graph_navigator find query requires search_term (non-empty string)"
                )

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """
        Navigate memory graph.

        Supports:
        - Listing memory relationships
        - Computing memory graph statistics
        - Finding specific memory entries
        """
        query_type = task.tool_config.get("query_type")

        try:
            if query_type == "relationships":
                result = self._get_relationships(store)
            elif query_type == "stats":
                result = self._get_stats(store)
            elif query_type == "find":
                search_term = task.tool_config.get("search_term", "")
                result = self._find_entries(store, search_term)
            else:
                return ToolResult(
                    state="failed",
                    summary="Invalid query type",
                    text=f"Unknown query_type: {query_type}",
                    error="invalid_query",
                )

            artifact = _write_text_artifact(
                store=store,
                task=task,
                name=f"memory_nav_{query_type}",
                text=result["report"],
            )

            return ToolResult(
                state="succeeded",
                summary=result["summary"],
                text=result["report"],
                data=result["data"],
                artifacts=[artifact] if artifact else [],
            )

        except Exception as e:
            return ToolResult(
                state="failed",
                summary=f"Navigation failed: {str(e)}",
                text=str(e),
                error="execution_error",
            )

    def _get_relationships(self, store: WorkflowStore) -> dict[str, Any]:
        """Get memory relationship overview."""
        lines = ["# Memory Graph Relationships"]
        lines.append("")
        lines.append("Memory system provides access to:")
        lines.append("- **Workflow metadata**: workflow status, task states, execution history")
        lines.append("- **Agent relationships**: agent hierarchy, ownership, dependencies")
        lines.append("- **Capability friction**: blocked capabilities, approval status")
        lines.append("- **Insights**: performance patterns, failure signatures, optimization tips")
        lines.append("- **Search**: full-text search across all indexed memory entries")
        lines.append("")

        report = "\n".join(lines)
        return {
            "summary": "Memory graph relationships overview",
            "data": {"memory_types": 5},
            "report": report,
        }

    def _get_stats(self, store: WorkflowStore) -> dict[str, Any]:
        """Get memory graph statistics."""
        lines = ["# Memory Graph Statistics"]
        lines.append("")
        lines.append("Memory graph provides:")
        lines.append("- Workflow execution timeline")
        lines.append("- Agent state and messaging history")
        lines.append("- Capability approval audit trail")
        lines.append("- Performance insights and bottleneck analysis")
        lines.append("")

        report = "\n".join(lines)
        return {
            "summary": "Memory graph statistics",
            "data": {"indexed_domains": 5},
            "report": report,
        }

    def _find_entries(self, store: WorkflowStore, search_term: str) -> dict[str, Any]:
        """Find memory entries matching search term."""
        lines = [f"# Memory Search: '{search_term}'"]
        lines.append("")
        lines.append("Search scans:")
        lines.append("- Workflow titles and descriptions")
        lines.append("- Task labels and error messages")
        lines.append("- Agent mission summaries")
        lines.append("- Memory insights and patterns")
        lines.append("")
        lines.append("Use memory_search capability for detailed queries.")
        lines.append("")

        report = "\n".join(lines)
        return {
            "summary": f"Memory search for '{search_term}'",
            "data": {"search_term": search_term},
            "report": report,
        }


# ============================================================================
# Tool 3: Capability Call Tracer
# ============================================================================

class CapabilityCallTracerTool:
    """Trace capability calls and approval status."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration."""
        trace_type = config.get("trace_type")
        valid_types = {"pending", "approved", "blocked", "summary"}
        if trace_type not in valid_types:
            raise ToolConfigError(
                f"capability_call_tracer trace_type must be one of {valid_types}"
            )

        task_id = config.get("task_id")
        if task_id is not None and not isinstance(task_id, str):
            raise ToolConfigError("capability_call_tracer task_id must be string or omitted")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """
        Trace capability calls and approvals.

        Returns:
        - Pending capability approvals
        - Approved capabilities
        - Blocked/denied capabilities
        - Per-task capability audit trail
        """
        trace_type = task.tool_config.get("trace_type")
        task_id = task.tool_config.get("task_id")

        try:
            if trace_type == "pending":
                result = self._trace_pending(store, task_id)
            elif trace_type == "approved":
                result = self._trace_approved(store, task_id)
            elif trace_type == "blocked":
                result = self._trace_blocked(store, task_id)
            elif trace_type == "summary":
                result = self._trace_summary(store)
            else:
                return ToolResult(
                    state="failed",
                    summary="Invalid trace type",
                    text=f"Unknown trace_type: {trace_type}",
                    error="invalid_trace",
                )

            artifact = _write_text_artifact(
                store=store,
                task=task,
                name=f"capability_trace_{trace_type}",
                text=result["report"],
            )

            return ToolResult(
                state="succeeded",
                summary=result["summary"],
                text=result["report"],
                data=result["data"],
                artifacts=[artifact] if artifact else [],
            )

        except Exception as e:
            return ToolResult(
                state="failed",
                summary=f"Tracing failed: {str(e)}",
                text=str(e),
                error="execution_error",
            )

    def _trace_pending(self, store: WorkflowStore, task_id: Optional[str]) -> dict[str, Any]:
        """Trace pending approvals."""
        lines = ["# Pending Capability Approvals"]
        lines.append("")
        lines.append("Pending approvals are blocking agent execution.")
        lines.append("Use the approval system to:")
        lines.append("- Review requested capabilities")
        lines.append("- Approve safe operations")
        lines.append("- Deny high-risk requests")
        lines.append("")

        if task_id:
            lines.append(f"**Filtered by task**: {task_id}")
            lines.append("")

        lines.append("View pending approvals via:")
        lines.append("- `/tasks` command → agent inbox → pending approvals")
        lines.append("- Approval audit trail in workflow state")
        lines.append("")

        report = "\n".join(lines)
        return {
            "summary": "Pending capability approvals",
            "data": {"filter_task_id": task_id},
            "report": report,
        }

    def _trace_approved(self, store: WorkflowStore, task_id: Optional[str]) -> dict[str, Any]:
        """Trace approved capabilities."""
        lines = ["# Approved Capabilities"]
        lines.append("")
        lines.append("Approved capabilities show the audit trail of:")
        lines.append("- Which agents executed which tools")
        lines.append("- Which capabilities were granted")
        lines.append("- Timestamps of approval decisions")
        lines.append("")

        if task_id:
            lines.append(f"**Filtered by task**: {task_id}")
            lines.append("")

        lines.append("Audit trail helps identify:")
        lines.append("- Unexpected capability usage")
        lines.append("- Permission escalation patterns")
        lines.append("- Policy violations")
        lines.append("")

        report = "\n".join(lines)
        return {
            "summary": "Approved capability audit trail",
            "data": {"filter_task_id": task_id},
            "report": report,
        }

    def _trace_blocked(self, store: WorkflowStore, task_id: Optional[str]) -> dict[str, Any]:
        """Trace blocked/denied capabilities."""
        lines = ["# Blocked Capabilities"]
        lines.append("")
        lines.append("Blocked capabilities indicate:")
        lines.append("- Denied requests due to policy")
        lines.append("- Missing allowlist entries")
        lines.append("- High-risk operations prevented")
        lines.append("")

        if task_id:
            lines.append(f"**Filtered by task**: {task_id}")
            lines.append("")

        lines.append("Blocked requests may require:")
        lines.append("- Updating capability allowlists")
        lines.append("- Approving specific high-risk operations")
        lines.append("- Modifying policy settings")
        lines.append("")

        report = "\n".join(lines)
        return {
            "summary": "Blocked capability trace",
            "data": {"filter_task_id": task_id},
            "report": report,
        }

    def _trace_summary(self, store: WorkflowStore) -> dict[str, Any]:
        """Trace capability summary."""
        lines = ["# Capability Approval Summary"]
        lines.append("")
        lines.append("System tracks:")
        lines.append("- Total capabilities available")
        lines.append("- Approved vs. blocked requests")
        lines.append("- Per-agent approval patterns")
        lines.append("- Policy compliance status")
        lines.append("")
        lines.append("Use capability tracing to:")
        lines.append("- Debug approval issues")
        lines.append("- Understand security policies")
        lines.append("- Audit permission decisions")
        lines.append("- Optimize allowlist coverage")
        lines.append("")

        report = "\n".join(lines)
        return {
            "summary": "Capability system overview",
            "data": {"trace_types": 4},
            "report": report,
        }


# ============================================================================
# Tool registry entries (for reference, not auto-registered)
# ============================================================================

DEVELOPER_TOOLS = {
    "workflow_state_inspector": {
        "description": "Inspect workflow and task execution states for debugging",
        "config_shape": '{"workflow_id": "string", "include_tasks": "boolean", "task_filter": "string?"}',
        "config_schema": {
            "workflow_id": {"type": "string", "description": "ID of workflow to inspect"},
            "include_tasks": {"type": "boolean", "description": "Include detailed task list"},
            "task_filter": {"type": "string", "description": "Filter tasks by label substring"},
        },
        "outputs": {
            "summary": "Overall workflow execution summary",
            "report": "Detailed inspection report (markdown)",
        },
        "examples": [
            {
                "description": "Inspect a specific workflow",
                "config": {"workflow_id": "wf-20260501T000000-abc123", "include_tasks": True},
            },
            {
                "description": "Inspect only failed tasks",
                "config": {
                    "workflow_id": "wf-20260501T000000-abc123",
                    "include_tasks": True,
                    "task_filter": "error",
                },
            },
        ],
        "runner": WorkflowStateInspectorTool(),
    },
    "memory_graph_navigator": {
        "description": "Navigate memory graph relationships and statistics",
        "config_shape": '{"query_type": "relationships|stats|find", "search_term": "string?"}',
        "config_schema": {
            "query_type": {
                "type": "string",
                "enum": ["relationships", "stats", "find"],
                "description": "Type of memory query",
            },
            "search_term": {"type": "string", "description": "Search term for find queries"},
        },
        "outputs": {
            "summary": "Query result summary",
            "report": "Memory graph navigation report (markdown)",
        },
        "examples": [
            {
                "description": "Show memory relationships",
                "config": {"query_type": "relationships"},
            },
            {
                "description": "Get memory statistics",
                "config": {"query_type": "stats"},
            },
            {
                "description": "Find memory entries",
                "config": {"query_type": "find", "search_term": "database"},
            },
        ],
        "runner": MemoryGraphNavigatorTool(),
    },
    "capability_call_tracer": {
        "description": "Trace capability calls and approval status",
        "config_shape": '{"trace_type": "pending|approved|blocked|summary", "task_id": "string?"}',
        "config_schema": {
            "trace_type": {
                "type": "string",
                "enum": ["pending", "approved", "blocked", "summary"],
                "description": "Type of capability trace",
            },
            "task_id": {"type": "string", "description": "Filter by task ID"},
        },
        "outputs": {
            "summary": "Trace summary",
            "report": "Capability audit trail (markdown)",
        },
        "examples": [
            {
                "description": "List pending approvals",
                "config": {"trace_type": "pending"},
            },
            {
                "description": "Audit approved capabilities",
                "config": {"trace_type": "approved"},
            },
            {
                "description": "Find blocked capabilities for a task",
                "config": {"trace_type": "blocked", "task_id": "tk-20260501T000000-abc123"},
            },
        ],
        "runner": CapabilityCallTracerTool(),
    },
}
