"""
Production-ready workflow tools for advanced orchestration patterns.

These tools solve critical workflow automation problems:

Orchestration Tools:
1. Task Dependency Resolver - aggregates outputs from multiple dependent tasks
2. Conditional Executor - gates task execution based on conditions from previous results
3. Batch Result Aggregator - aggregates, transforms, and summarizes parallel task results

General-Purpose Tools:
4. Filesystem Navigator - efficiently navigate and explore directory structures
5. Data Query Transform - query and transform data using flexible patterns
6. Data Validator - validate data against schemas and rules
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mr1.tools import (
    ToolConfigError,
    ToolResult,
    ToolRunner,
    _artifact,
    _write_text_artifact,
)
from mr1.workflow_models import Task, Workflow
from mr1.workflow_store import WorkflowStore


# ============================================================================
# Tool 1: Task Dependency Resolver
# ============================================================================
# Purpose: Resolves and aggregates outputs from multiple dependent tasks,
# enabling complex data flow patterns and cross-task result consolidation.
#
# Use cases:
# - Merge results from parallel processing steps
# - Extract and combine specific fields from multiple task outputs
# - Create consolidated reports from distributed workflows
# - Enable dynamic task chains with complex dependencies
# ============================================================================

class TaskDependencyResolverTool:
    """Resolves and aggregates outputs from multiple workflow tasks."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required fields and types are present."""
        tasks_config = config.get("tasks")
        if not isinstance(tasks_config, list) or not tasks_config:
            raise ToolConfigError(
                "task_dependency_resolver requires non-empty tool_config.tasks list"
            )

        for i, task_spec in enumerate(tasks_config):
            if not isinstance(task_spec, dict):
                raise ToolConfigError(
                    f"task_dependency_resolver tasks[{i}] must be a JSON object"
                )
            task_name = task_spec.get("task_name")
            if not isinstance(task_name, str) or not task_name:
                raise ToolConfigError(
                    f"task_dependency_resolver tasks[{i}] requires task_name (string)"
                )
            output_path = task_spec.get("output_path")
            if not isinstance(output_path, str) or not output_path:
                raise ToolConfigError(
                    f"task_dependency_resolver tasks[{i}] requires output_path (string)"
                )

        merge_strategy = config.get("merge_strategy", "merge")
        if merge_strategy not in ("merge", "array", "first_found"):
            raise ToolConfigError(
                "task_dependency_resolver merge_strategy must be 'merge', 'array', or 'first_found'"
            )

        fail_on_missing = config.get("fail_on_missing", False)
        if not isinstance(fail_on_missing, bool):
            raise ToolConfigError("task_dependency_resolver fail_on_missing must be a boolean")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """
        Resolve and aggregate outputs from multiple tasks.

        Returns consolidated data with results from specified task outputs.
        Each task output is extracted from result.data and merged according
        to the merge_strategy.
        """
        tasks_config = task.tool_config.get("tasks", [])
        merge_strategy = task.tool_config.get("merge_strategy", "merge")
        fail_on_missing = task.tool_config.get("fail_on_missing", False)

        resolved_outputs = {}
        missing_tasks = []
        errors = []

        for task_spec in tasks_config:
            task_name = task_spec["task_name"]
            output_path = task_spec["output_path"]

            upstream_task = None
            for t in workflow.tasks:
                if t.task_id == task_name or t.name == task_name:
                    upstream_task = t
                    break

            if upstream_task is None:
                missing_tasks.append(task_name)
                if fail_on_missing:
                    return ToolResult(
                        state="failed",
                        summary=f"dependency resolver: task '{task_name}' not found in workflow",
                        text="",
                        data={"resolved": {}, "missing_tasks": missing_tasks},
                        error=f"required task not found: {task_name}",
                    )
                continue

            try:
                task_result_path = (
                    store.task_artifacts_dir(task.workflow_id, upstream_task.task_id)
                    / "task_result.json"
                )
                if not task_result_path.exists():
                    if fail_on_missing:
                        return ToolResult(
                            state="failed",
                            summary=f"dependency resolver: no result for task '{task_name}'",
                            text="",
                            data={"resolved": resolved_outputs, "missing_tasks": missing_tasks},
                            error=f"task result not found: {task_name}",
                        )
                    missing_tasks.append(task_name)
                    continue

                task_result_data = json.loads(task_result_path.read_text())
                value = _extract_nested_value(task_result_data, output_path)

                if value is None and fail_on_missing:
                    return ToolResult(
                        state="failed",
                        summary=f"dependency resolver: path '{output_path}' not in task '{task_name}'",
                        text="",
                        data={"resolved": resolved_outputs, "missing_tasks": missing_tasks},
                        error=f"output path not found: {output_path} in {task_name}",
                    )

                resolved_outputs[task_name] = value

            except json.JSONDecodeError as e:
                error_msg = f"failed to parse result for task '{task_name}': {e}"
                errors.append(error_msg)
                if fail_on_missing:
                    return ToolResult(
                        state="failed",
                        summary=f"dependency resolver: invalid JSON for task '{task_name}'",
                        text="",
                        data={"resolved": resolved_outputs, "errors": errors},
                        error=error_msg,
                    )

        merged_result = _merge_results(resolved_outputs, merge_strategy)
        artifacts = []

        if merged_result:
            artifacts.append(
                _write_text_artifact(
                    store,
                    task,
                    "resolved_outputs",
                    json.dumps(merged_result, indent=2),
                )
            )

        state = "succeeded" if not missing_tasks else "succeeded"
        return ToolResult(
            state=state,
            summary=f"resolved {len(resolved_outputs)} task output(s), {len(missing_tasks)} missing",
            text=json.dumps(merged_result, indent=2) if merged_result else "",
            data={
                "resolved_count": len(resolved_outputs),
                "missing_count": len(missing_tasks),
                "missing_tasks": missing_tasks,
                "merge_strategy": merge_strategy,
                "merged_result": merged_result,
            },
            artifacts=artifacts,
            metadata={"errors": errors} if errors else {},
        )


# ============================================================================
# Tool 2: Conditional Executor
# ============================================================================
# Purpose: Gates task execution based on conditions evaluated against previous
# task results, enabling conditional branching in workflows.
#
# Use cases:
# - Skip tasks based on upstream output conditions
# - Branch workflows based on result predicates
# - Implement approval gates with human-readable criteria
# - Enable adaptive workflows that respond to results
# ============================================================================

class ConditionalExecutorTool:
    """Evaluates conditions against task outputs and determines execution."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate condition specifications."""
        conditions = config.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ToolConfigError(
                "conditional_executor requires non-empty tool_config.conditions list"
            )

        for i, cond in enumerate(conditions):
            if not isinstance(cond, dict):
                raise ToolConfigError(
                    f"conditional_executor conditions[{i}] must be a JSON object"
                )
            field = cond.get("field")
            if not isinstance(field, str) or not field:
                raise ToolConfigError(
                    f"conditional_executor conditions[{i}] requires field (string)"
                )
            operator = cond.get("operator")
            if operator not in ("equals", "not_equals", "contains", "not_contains",
                               "greater_than", "less_than", "in_list", "not_in_list"):
                raise ToolConfigError(
                    f"conditional_executor conditions[{i}] operator must be one of: "
                    "equals, not_equals, contains, not_contains, greater_than, less_than, in_list, not_in_list"
                )

        logic = config.get("logic", "and")
        if logic not in ("and", "or"):
            raise ToolConfigError("conditional_executor logic must be 'and' or 'or'")

        on_true = config.get("on_true", "allow")
        on_false = config.get("on_false", "block")
        if on_true not in ("allow", "block"):
            raise ToolConfigError("conditional_executor on_true must be 'allow' or 'block'")
        if on_false not in ("allow", "block"):
            raise ToolConfigError("conditional_executor on_false must be 'allow' or 'block'")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """
        Evaluate conditions against task outputs.

        Returns execution decision (allow/block) based on condition evaluation.
        """
        conditions = task.tool_config.get("conditions", [])
        logic = task.tool_config.get("logic", "and")
        on_true = task.tool_config.get("on_true", "allow")
        on_false = task.tool_config.get("on_false", "block")
        task_name = task.tool_config.get("target_task")

        target_task = None
        if task_name:
            for t in workflow.tasks:
                if t.task_id == task_name or t.name == task_name:
                    target_task = t
                    break

        results = []
        all_met = True

        for i, cond in enumerate(conditions):
            field = cond.get("field")
            operator = cond.get("operator")
            value = cond.get("value")

            try:
                condition_result = _evaluate_condition(
                    field, operator, value, target_task, store, task.workflow_id
                )
                results.append({
                    "index": i,
                    "field": field,
                    "operator": operator,
                    "expected": value,
                    "satisfied": condition_result,
                })
                if not condition_result:
                    all_met = False

            except Exception as e:
                return ToolResult(
                    state="failed",
                    summary=f"condition evaluation failed at index {i}",
                    text="",
                    data={"condition_index": i, "field": field},
                    error=str(e),
                )

        final_decision = (all_met and on_true == "allow") or (not all_met and on_false == "allow")
        decision_text = "allow" if final_decision else "block"

        return ToolResult(
            state="succeeded",
            summary=f"conditional execution: {decision_text}",
            text=decision_text,
            data={
                "decision": decision_text,
                "allow": final_decision,
                "logic": logic,
                "conditions_met": sum(1 for r in results if r["satisfied"]),
                "conditions_total": len(results),
                "condition_results": results,
            },
            artifacts=[
                _write_text_artifact(
                    store,
                    task,
                    "condition_results",
                    json.dumps(results, indent=2),
                )
            ],
        )


# ============================================================================
# Tool 3: Batch Result Aggregator
# ============================================================================
# Purpose: Aggregates, transforms, and summarizes results from multiple
# parallel tasks, enabling rollup and consolidation workflows.
#
# Use cases:
# - Aggregate results from parallel processing batches
# - Create statistical summaries from distributed tasks
# - Transform and normalize results from multiple sources
# - Consolidate metrics and logs from parallel execution
# ============================================================================

class BatchResultAggregatorTool:
    """Aggregates and transforms results from multiple parallel tasks."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate aggregation configuration."""
        aggregations = config.get("aggregations")
        if not isinstance(aggregations, list) or not aggregations:
            raise ToolConfigError(
                "batch_result_aggregator requires non-empty tool_config.aggregations list"
            )

        for i, agg in enumerate(aggregations):
            if not isinstance(agg, dict):
                raise ToolConfigError(
                    f"batch_result_aggregator aggregations[{i}] must be a JSON object"
                )
            name = agg.get("name")
            if not isinstance(name, str) or not name:
                raise ToolConfigError(
                    f"batch_result_aggregator aggregations[{i}] requires name (string)"
                )
            input_path = agg.get("input_path")
            if not isinstance(input_path, str) or not input_path:
                raise ToolConfigError(
                    f"batch_result_aggregator aggregations[{i}] requires input_path (string)"
                )
            function = agg.get("function")
            if function not in ("sum", "count", "avg", "min", "max", "concat", "list", "first"):
                raise ToolConfigError(
                    f"batch_result_aggregator aggregations[{i}] function must be one of: "
                    "sum, count, avg, min, max, concat, list, first"
                )

        pattern = config.get("task_pattern", ".*")
        if not isinstance(pattern, str):
            raise ToolConfigError("batch_result_aggregator task_pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as e:
            raise ToolConfigError(f"batch_result_aggregator task_pattern invalid regex: {e}")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """
        Aggregate results from matching parallel tasks.

        Returns consolidated aggregated results from all matching tasks.
        """
        aggregations = task.tool_config.get("aggregations", [])
        task_pattern = task.tool_config.get("task_pattern", ".*")

        pattern = re.compile(task_pattern)
        matching_tasks = [
            t for t in workflow.tasks
            if pattern.match(t.name or "") or pattern.match(t.task_id or "")
        ]

        if not matching_tasks:
            return ToolResult(
                state="succeeded",
                summary="batch aggregator: no matching tasks found",
                text="",
                data={
                    "aggregations": {},
                    "matched_count": 0,
                    "pattern": task_pattern,
                },
                artifacts=[
                    _write_text_artifact(
                        store,
                        task,
                        "aggregation_results",
                        json.dumps({"aggregations": {}, "matched_count": 0}, indent=2),
                    )
                ],
            )

        aggregated = {}
        errors = []

        for agg in aggregations:
            agg_name = agg["name"]
            input_path = agg["input_path"]
            function = agg["function"]

            values = []
            for matching_task in matching_tasks:
                try:
                    task_result_path = (
                        store.task_artifacts_dir(task.workflow_id, matching_task.task_id)
                        / "task_result.json"
                    )
                    if task_result_path.exists():
                        task_result_data = json.loads(task_result_path.read_text())
                        value = _extract_nested_value(task_result_data, input_path)
                        if value is not None:
                            values.append(value)
                except (json.JSONDecodeError, OSError) as e:
                    errors.append(f"error reading {matching_task.task_id}: {e}")

            try:
                aggregated[agg_name] = _aggregate_values(values, function)
            except Exception as e:
                errors.append(f"aggregation '{agg_name}' failed: {e}")
                aggregated[agg_name] = None

        result_summary = {
            "aggregations": aggregated,
            "matched_count": len(matching_tasks),
            "pattern": task_pattern,
            "errors": errors if errors else None,
        }

        return ToolResult(
            state="succeeded",
            summary=f"aggregated results from {len(matching_tasks)} task(s)",
            text=json.dumps(aggregated, indent=2),
            data=result_summary,
            artifacts=[
                _write_text_artifact(
                    store,
                    task,
                    "aggregation_results",
                    json.dumps(result_summary, indent=2),
                )
            ],
            metadata={"error_count": len(errors)} if errors else {},
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_nested_value(data: Any, path: str) -> Any:
    """Extract a value from nested data using dot notation (e.g., 'data.result.value')."""
    if not path or not isinstance(data, dict):
        return None
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _merge_results(data: dict[str, Any], strategy: str) -> Any:
    """Merge multiple task outputs according to merge strategy."""
    if strategy == "array":
        return list(data.values())
    elif strategy == "first_found":
        return next(iter(data.values()), None) if data else None
    else:  # merge
        merged = {}
        for key, value in data.items():
            if isinstance(value, dict):
                merged.update(value)
            else:
                merged[key] = value
        return merged


def _evaluate_condition(
    field: str,
    operator: str,
    expected_value: Any,
    target_task: Optional[Task],
    store: WorkflowStore,
    workflow_id: str,
) -> bool:
    """Evaluate a single condition against task output."""
    if target_task is None:
        return False

    try:
        task_result_path = (
            store.task_artifacts_dir(workflow_id, target_task.task_id)
            / "task_result.json"
        )
        if not task_result_path.exists():
            return False

        task_result = json.loads(task_result_path.read_text())
        actual_value = _extract_nested_value(task_result, field)

        if operator == "equals":
            return actual_value == expected_value
        elif operator == "not_equals":
            return actual_value != expected_value
        elif operator == "contains":
            return isinstance(actual_value, str) and isinstance(expected_value, str) and expected_value in actual_value
        elif operator == "not_contains":
            return isinstance(actual_value, str) and isinstance(expected_value, str) and expected_value not in actual_value
        elif operator == "greater_than":
            return actual_value > expected_value
        elif operator == "less_than":
            return actual_value < expected_value
        elif operator == "in_list":
            return actual_value in (expected_value if isinstance(expected_value, list) else [expected_value])
        elif operator == "not_in_list":
            return actual_value not in (expected_value if isinstance(expected_value, list) else [expected_value])
        else:
            return False

    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _aggregate_values(values: list[Any], function: str) -> Any:
    """Aggregate a list of values using the specified function."""
    if not values:
        return None

    if function == "sum":
        return sum(v for v in values if isinstance(v, (int, float)))
    elif function == "count":
        return len(values)
    elif function == "avg":
        numeric = [v for v in values if isinstance(v, (int, float))]
        return sum(numeric) / len(numeric) if numeric else None
    elif function == "min":
        numeric = [v for v in values if isinstance(v, (int, float))]
        return min(numeric) if numeric else None
    elif function == "max":
        numeric = [v for v in values if isinstance(v, (int, float))]
        return max(numeric) if numeric else None
    elif function == "concat":
        return "".join(str(v) for v in values if v is not None)
    elif function == "list":
        return values
    elif function == "first":
        return values[0] if values else None
    else:
        return None


# ============================================================================
# Tool 4: Filesystem Navigator
# ============================================================================
# Purpose: Efficiently navigate and explore directory structures with
# filtering and pattern matching capabilities.
#
# Use cases:
# - Find files matching patterns (extension, name, size)
# - Explore directory trees with depth limits
# - Collect file metadata for analysis
# - Filter files by various criteria
# ============================================================================

class FilesystemNavigatorTool:
    """Navigate and explore directory structures with filtering."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate filesystem navigation configuration."""
        path = config.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ToolConfigError("filesystem_navigator requires tool_config.path")

        max_depth = config.get("max_depth", 3)
        if not isinstance(max_depth, int) or max_depth < 0 or max_depth > 20:
            raise ToolConfigError("filesystem_navigator max_depth must be int between 0-20")

        pattern = config.get("pattern")
        if pattern is not None and not isinstance(pattern, str):
            raise ToolConfigError("filesystem_navigator pattern must be a string")
        if pattern:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ToolConfigError(f"filesystem_navigator pattern invalid regex: {e}")

        include_dirs = config.get("include_dirs", True)
        if not isinstance(include_dirs, bool):
            raise ToolConfigError("filesystem_navigator include_dirs must be boolean")

        max_size_bytes = config.get("max_size_bytes")
        if max_size_bytes is not None:
            if not isinstance(max_size_bytes, int) or max_size_bytes < 0:
                raise ToolConfigError("filesystem_navigator max_size_bytes must be non-negative int")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """Navigate filesystem and return filtered results."""
        from mr1.capability_policy import normalize_path

        del workflow
        base_path = normalize_path(task.tool_config["path"])

        if not base_path.exists():
            return ToolResult(
                state="failed",
                summary=f"filesystem_navigator: path does not exist: {base_path}",
                text="",
                data={"path": str(base_path)},
                error=f"path does not exist: {base_path}",
            )

        if not base_path.is_dir():
            return ToolResult(
                state="failed",
                summary=f"filesystem_navigator: path is not a directory: {base_path}",
                text="",
                data={"path": str(base_path)},
                error=f"path is not a directory: {base_path}",
            )

        max_depth = task.tool_config.get("max_depth", 3)
        pattern = task.tool_config.get("pattern")
        include_dirs = task.tool_config.get("include_dirs", True)
        max_size_bytes = task.tool_config.get("max_size_bytes")

        pattern_re = re.compile(pattern) if pattern else None
        results = []

        def traverse(current_path: Path, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                for entry in sorted(current_path.iterdir()):
                    if entry.is_dir():
                        if include_dirs and (not pattern_re or pattern_re.search(entry.name)):
                            results.append({
                                "path": str(entry),
                                "name": entry.name,
                                "type": "directory",
                                "depth": depth,
                            })
                        traverse(entry, depth + 1)
                    elif entry.is_file():
                        if not pattern_re or pattern_re.search(entry.name):
                            stat_info = entry.stat()
                            size = stat_info.st_size
                            if max_size_bytes is None or size <= max_size_bytes:
                                results.append({
                                    "path": str(entry),
                                    "name": entry.name,
                                    "type": "file",
                                    "size_bytes": size,
                                    "depth": depth,
                                })
            except (OSError, PermissionError):
                pass

        traverse(base_path, 0)

        results_text = "\n".join(
            f"{r['path']} ({r['type']}, size={r.get('size_bytes', 'N/A')})" for r in results
        )

        return ToolResult(
            state="succeeded",
            summary=f"found {len(results)} item(s) in {base_path}",
            text=results_text,
            data={
                "path": str(base_path),
                "max_depth": max_depth,
                "pattern": pattern,
                "items_found": len(results),
                "items": results,
            },
            artifacts=[
                _write_text_artifact(
                    store,
                    task,
                    "filesystem_results",
                    json.dumps({
                        "path": str(base_path),
                        "items": results,
                        "count": len(results),
                    }, indent=2),
                )
            ],
        )


# ============================================================================
# Tool 5: Data Query Transform
# ============================================================================
# Purpose: Query and transform data using flexible patterns with support
# for filtering, mapping, and aggregation operations.
#
# Use cases:
# - Extract specific fields from complex data structures
# - Apply transformations to data values
# - Filter data based on conditions
# - Transform structured data formats
# ============================================================================

class DataQueryTransformTool:
    """Query and transform data using flexible patterns and SQLite databases."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate data query and transform configuration."""
        mode = config.get("mode", "data")
        if mode not in ("data", "sqlite"):
            raise ToolConfigError("data_query_transform mode must be: data or sqlite")

        if mode == "sqlite":
            db_path = config.get("db_path")
            if not isinstance(db_path, str) or not db_path.strip():
                raise ToolConfigError("data_query_transform sqlite mode requires db_path")
            query = config.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolConfigError("data_query_transform sqlite mode requires query")
        else:
            operation = config.get("operation")
            if operation not in ("filter", "map", "extract", "aggregate", "transform"):
                raise ToolConfigError(
                    "data_query_transform operation must be: filter, map, extract, aggregate, or transform"
                )

            if operation in ("filter", "map", "extract"):
                pattern = config.get("pattern")
                if not isinstance(pattern, str) or not pattern.strip():
                    raise ToolConfigError(f"data_query_transform {operation} requires pattern")

            if operation == "aggregate":
                function = config.get("function")
                if function not in ("sum", "count", "avg", "min", "max", "concat"):
                    raise ToolConfigError(
                        "data_query_transform aggregate function must be: sum, count, avg, min, max, concat"
                    )

            input_data = config.get("input_data")
            if not isinstance(input_data, (dict, list)):
                raise ToolConfigError("data_query_transform input_data must be dict or list")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """Execute data query and transformation."""
        del workflow

        mode = task.tool_config.get("mode", "data")

        try:
            if mode == "sqlite":
                return self._run_sqlite_query(task, store)
            else:
                return self._run_data_transform(task, store)

        except Exception as e:
            return ToolResult(
                state="failed",
                summary="data_query_transform execution failed",
                text="",
                error=str(e),
            )

    def _run_sqlite_query(self, task: Task, store: WorkflowStore) -> ToolResult:
        """Execute SQLite query with optional data transformation."""
        db_path = task.tool_config.get("db_path")
        query = task.tool_config.get("query")
        params = task.tool_config.get("params", {})
        limit = task.tool_config.get("limit", 10000)
        transform_op = task.tool_config.get("transform_op")

        path = Path(db_path).expanduser()
        if not path.exists():
            return ToolResult(
                state="failed",
                summary=f"SQLite database not found: {db_path}",
                text="",
                data={"db_path": str(path)},
                error=f"database file does not exist: {path}",
            )

        try:
            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if limit and limit > 0:
                safe_limit = min(limit, 100000)
                query_upper = query.rstrip().upper()
                if "LIMIT" not in query_upper:
                    query = f"{query.rstrip()} LIMIT {safe_limit}"

            if params:
                if isinstance(params, dict):
                    cursor.execute(query, params)
                elif isinstance(params, (list, tuple)):
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
            else:
                cursor.execute(query)

            rows = cursor.fetchall()
            result_data = [dict(row) for row in rows]

            if transform_op:
                try:
                    result_data = self._apply_transform(result_data, transform_op)
                except Exception as e:
                    slice_data = result_data[:100] if isinstance(result_data, list) else result_data
                    return ToolResult(
                        state="failed",
                        summary="SQLite query succeeded but transformation failed",
                        text=json.dumps(slice_data, indent=2, default=str),
                        data={"query": query, "rows_retrieved": len(result_data) if isinstance(result_data, list) else "unknown"},
                        error=f"transformation failed: {e}",
                    )

            conn.close()

            display_data = result_data[:100] if isinstance(result_data, list) else result_data
            row_count = len(result_data) if isinstance(result_data, list) else 1
            rows_displayed = min(len(result_data), 100) if isinstance(result_data, list) else 1

            return ToolResult(
                state="succeeded",
                summary=f"SQLite query completed: {row_count} rows" if isinstance(result_data, list) else "SQLite query completed",
                text=json.dumps(display_data, indent=2, default=str),
                data={
                    "db_path": str(path),
                    "query": query,
                    "rows_retrieved": row_count,
                    "rows_displayed": rows_displayed,
                    "result": result_data,
                },
                artifacts=[
                    _write_text_artifact(
                        store,
                        task,
                        "query_results",
                        json.dumps(result_data, indent=2, default=str),
                    )
                ],
                metadata={
                    "db_path": str(path),
                    "row_count": row_count,
                    "has_transform": transform_op is not None,
                },
            )

        except sqlite3.DatabaseError as e:
            return ToolResult(
                state="failed",
                summary="SQLite database error",
                text="",
                data={"db_path": str(path)},
                error=f"database error: {e}",
            )
        except sqlite3.OperationalError as e:
            return ToolResult(
                state="failed",
                summary="SQLite query error",
                text="",
                data={"query": query},
                error=f"query error: {e}",
            )
        except sqlite3.Error as e:
            return ToolResult(
                state="failed",
                summary="SQLite execution error",
                text="",
                error=f"SQLite error: {e}",
            )

    def _run_data_transform(self, task: Task, store: WorkflowStore) -> ToolResult:
        """Execute in-memory data transformation."""
        operation = task.tool_config.get("operation")
        input_data = task.tool_config.get("input_data")
        pattern = task.tool_config.get("pattern")
        function = task.tool_config.get("function")

        if operation == "filter":
            result_data = _filter_data(input_data, pattern)
        elif operation == "map":
            result_data = _map_data(input_data, pattern)
        elif operation == "extract":
            result_data = _extract_data(input_data, pattern)
        elif operation == "aggregate":
            result_data = _aggregate_data(input_data, function)
        elif operation == "transform":
            result_data = _transform_data(input_data, pattern)
        else:
            return ToolResult(
                state="failed",
                summary=f"unknown operation: {operation}",
                text="",
                error=f"unknown operation: {operation}",
            )

        return ToolResult(
            state="succeeded",
            summary=f"data_query_transform {operation} completed",
            text=json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data),
            data={
                "operation": operation,
                "result": result_data,
                "input_count": len(input_data) if isinstance(input_data, (dict, list)) else 1,
            },
            artifacts=[
                _write_text_artifact(
                    store,
                    task,
                    "transform_result",
                    json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data),
                )
            ],
        )

    def _apply_transform(self, data: list[dict[str, Any]], op: str) -> Any:
        """Apply transformation to query results."""
        if op == "count":
            return len(data)
        elif op == "keys":
            return list(data[0].keys()) if data else []
        elif op == "flatten":
            result = []
            for row in data:
                result.extend(row.values())
            return result
        elif op == "summary":
            return {
                "row_count": len(data),
                "columns": list(data[0].keys()) if data else [],
                "sample": data[0] if data else None,
            }
        else:
            return data


# ============================================================================
# Tool 6: Data Validator
# ============================================================================
# Purpose: Validate data against schemas and rules with comprehensive
# error reporting and validation statistics.
#
# Use cases:
# - Validate data structure against schema
# - Enforce data type constraints
# - Apply custom validation rules
# - Generate validation reports
# ============================================================================

class DataValidatorTool:
    """Validate data against schemas and rules."""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate data validation configuration."""
        data = config.get("data")
        if data is None:
            raise ToolConfigError("data_validator requires tool_config.data")

        validation_mode = config.get("validation_mode", "schema")
        if validation_mode not in ("schema", "rules"):
            raise ToolConfigError("data_validator validation_mode must be 'schema' or 'rules'")

        if validation_mode == "schema":
            schema = config.get("schema")
            if not isinstance(schema, dict):
                raise ToolConfigError("data_validator schema validation requires schema dict")

        if validation_mode == "rules":
            rules = config.get("rules")
            if not isinstance(rules, list) or not rules:
                raise ToolConfigError("data_validator rules validation requires non-empty rules list")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        """Validate data and return validation results."""
        del workflow

        data = task.tool_config.get("data")
        validation_mode = task.tool_config.get("validation_mode", "schema")

        try:
            if validation_mode == "schema":
                schema = task.tool_config.get("schema")
                errors, warnings = _validate_schema(data, schema)
                is_valid = len(errors) == 0
            else:
                rules = task.tool_config.get("rules", [])
                errors, warnings = _validate_rules(data, rules)
                is_valid = len(errors) == 0

            validation_result = {
                "valid": is_valid,
                "mode": validation_mode,
                "errors": errors,
                "warnings": warnings,
                "error_count": len(errors),
                "warning_count": len(warnings),
            }

            summary = f"data_validator: {'valid' if is_valid else 'invalid'} ({len(errors)} errors, {len(warnings)} warnings)"

            return ToolResult(
                state="succeeded",
                summary=summary,
                text=json.dumps(validation_result, indent=2),
                data=validation_result,
                artifacts=[
                    _write_text_artifact(
                        store,
                        task,
                        "validation_report",
                        json.dumps(validation_result, indent=2),
                    )
                ],
            )

        except Exception as e:
            return ToolResult(
                state="failed",
                summary="data_validator: validation failed",
                text="",
                error=str(e),
            )


# ============================================================================
# Data Query Transform Helpers
# ============================================================================

def _filter_data(data: Any, pattern: str) -> Any:
    """Filter data using pattern matching (exact string matching or regex on values)."""
    if isinstance(data, list):
        return [
            item for item in data
            if (isinstance(item, str) and _matches_pattern(item, pattern))
            or (isinstance(item, dict) and any(
                isinstance(v, str) and _matches_pattern(v, pattern) for v in item.values()
            ))
        ]
    elif isinstance(data, dict):
        return {
            k: v for k, v in data.items()
            if (isinstance(v, str) and _matches_pattern(v, pattern))
            or (isinstance(v, dict) and any(
                isinstance(vv, str) and _matches_pattern(vv, pattern) for vv in v.values()
            ))
        }
    return data


def _matches_pattern(text: str, pattern: str) -> bool:
    """Match text against pattern using exact match or regex."""
    if text == pattern:
        return True
    try:
        return re.search(pattern, text) is not None
    except re.error:
        return text == pattern


def _map_data(data: Any, pattern: str) -> Any:
    """Map data fields using pattern (field selector)."""
    if isinstance(data, list):
        return [_extract_from_dict(item, pattern) if isinstance(item, dict) else item for item in data]
    elif isinstance(data, dict):
        return _extract_from_dict(data, pattern)
    return data


def _extract_data(data: Any, pattern: str) -> Any:
    """Extract nested values using dot notation."""
    if isinstance(data, list):
        return [_extract_nested_value(item, pattern) if isinstance(item, dict) else None for item in data]
    elif isinstance(data, dict):
        return _extract_nested_value(data, pattern)
    return None


def _aggregate_data(data: Any, function: str) -> Any:
    """Aggregate data using specified function."""
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            flat_values = []
            for item in data:
                if isinstance(item, dict):
                    flat_values.extend(item.values())
                else:
                    flat_values.append(item)
            return _aggregate_values(flat_values, function)
        else:
            return _aggregate_values(data, function)
    elif isinstance(data, dict):
        return _aggregate_values(list(data.values()), function)
    return None


def _transform_data(data: Any, pattern: str) -> Any:
    """Transform data structure based on pattern (JSON path transformation)."""
    if not pattern:
        return data
    try:
        transformation = json.loads(pattern)
        return _apply_transformation(data, transformation)
    except json.JSONDecodeError:
        return None


def _extract_from_dict(data: dict[str, Any], field: str) -> Any:
    """Extract a single field from a dictionary."""
    return data.get(field)


def _apply_transformation(data: Any, transformation: dict[str, Any]) -> dict[str, Any]:
    """Apply transformation rules to data."""
    result = {}
    for new_key, source_path in transformation.items():
        if isinstance(source_path, str):
            result[new_key] = _extract_nested_value(data, source_path)
        else:
            result[new_key] = source_path
    return result


# ============================================================================
# Data Validator Helpers
# ============================================================================

def _validate_schema(data: Any, schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate data against schema."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        errors.append(f"expected dict, got {type(data).__name__}")
        return errors, warnings

    required_fields = schema.get("required", [])
    field_schemas = schema.get("properties", {})

    for field in required_fields:
        if field not in data:
            errors.append(f"missing required field: {field}")

    for field, value in data.items():
        if field not in field_schemas:
            warnings.append(f"unexpected field: {field}")
            continue

        field_schema = field_schemas[field]
        expected_type = field_schema.get("type")

        if expected_type:
            actual_type = type(value).__name__
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"field '{field}': expected string, got {actual_type}")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"field '{field}': expected integer, got {actual_type}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"field '{field}': expected number, got {actual_type}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"field '{field}': expected boolean, got {actual_type}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"field '{field}': expected array, got {actual_type}")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"field '{field}': expected object, got {actual_type}")

        if expected_type == "object" and isinstance(value, dict):
            nested_required = field_schema.get("required", [])
            nested_properties = field_schema.get("properties", {})
            for req_field in nested_required:
                if req_field not in value:
                    errors.append(f"field '{field}.{req_field}': missing required field")
            for nested_field, nested_value in value.items():
                if nested_field in nested_properties:
                    nested_schema = nested_properties[nested_field]
                    nested_type = nested_schema.get("type")
                    if nested_type == "string" and not isinstance(nested_value, str):
                        errors.append(f"field '{field}.{nested_field}': expected string, got {type(nested_value).__name__}")
                    elif nested_type == "integer" and not isinstance(nested_value, int):
                        errors.append(f"field '{field}.{nested_field}': expected integer, got {type(nested_value).__name__}")

        if expected_type == "array" and isinstance(value, list):
            if "minItems" in field_schema and len(value) < field_schema["minItems"]:
                errors.append(f"field '{field}': array length {len(value)} less than minimum {field_schema['minItems']}")
            if "maxItems" in field_schema and len(value) > field_schema["maxItems"]:
                errors.append(f"field '{field}': array length {len(value)} greater than maximum {field_schema['maxItems']}")
            items_schema = field_schema.get("items")
            if items_schema and isinstance(items_schema, dict):
                item_type = items_schema.get("type")
                item_properties = items_schema.get("properties", {})
                item_required = items_schema.get("required", [])
                for i, item in enumerate(value):
                    if item_type:
                        if item_type == "string" and not isinstance(item, str):
                            errors.append(f"field '{field}[{i}]': expected string, got {type(item).__name__}")
                        elif item_type == "integer" and not isinstance(item, int):
                            errors.append(f"field '{field}[{i}]': expected integer, got {type(item).__name__}")
                        elif item_type == "object" and isinstance(item, dict):
                            for req_field in item_required:
                                if req_field not in item:
                                    errors.append(f"field '{field}[{i}].{req_field}': missing required field")
                            for key, val in item.items():
                                if key in item_properties:
                                    item_field_schema = item_properties[key]
                                    item_field_type = item_field_schema.get("type")
                                    if item_field_type == "string" and not isinstance(val, str):
                                        errors.append(f"field '{field}[{i}].{key}': expected string, got {type(val).__name__}")
                                    elif item_field_type == "integer" and not isinstance(val, int):
                                        errors.append(f"field '{field}[{i}].{key}': expected integer, got {type(val).__name__}")

        if "minimum" in field_schema and isinstance(value, (int, float)):
            if value < field_schema["minimum"]:
                errors.append(f"field '{field}': value {value} less than minimum {field_schema['minimum']}")

        if "maximum" in field_schema and isinstance(value, (int, float)):
            if value > field_schema["maximum"]:
                errors.append(f"field '{field}': value {value} greater than maximum {field_schema['maximum']}")

        if "minLength" in field_schema and isinstance(value, str):
            if len(value) < field_schema["minLength"]:
                errors.append(f"field '{field}': string length {len(value)} less than minimum {field_schema['minLength']}")

        if "maxLength" in field_schema and isinstance(value, str):
            if len(value) > field_schema["maxLength"]:
                errors.append(f"field '{field}': string length {len(value)} greater than maximum {field_schema['maxLength']}")

        if "pattern" in field_schema and isinstance(value, str):
            if not re.match(field_schema["pattern"], value):
                errors.append(f"field '{field}': does not match pattern {field_schema['pattern']}")

        if "enum" in field_schema:
            if value not in field_schema["enum"]:
                errors.append(f"field '{field}': value '{value}' not in enum {field_schema['enum']}")

    return errors, warnings


def _validate_rules(data: Any, rules: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Validate data against custom rules."""
    errors: list[str] = []
    warnings: list[str] = []

    for rule in rules:
        rule_name = rule.get("name", "unnamed")
        rule_type = rule.get("type")

        if rule_type == "field_required":
            field = rule.get("field")
            if not isinstance(data, dict) or field not in data or data[field] is None:
                errors.append(f"rule '{rule_name}': required field '{field}' missing or null")

        elif rule_type == "field_type":
            field = rule.get("field")
            expected_type = rule.get("expected_type")
            if isinstance(data, dict) and field in data:
                actual_type = type(data[field]).__name__
                if expected_type == "string" and not isinstance(data[field], str):
                    errors.append(f"rule '{rule_name}': field '{field}' type mismatch")
                elif expected_type == "integer" and not isinstance(data[field], int):
                    errors.append(f"rule '{rule_name}': field '{field}' type mismatch")

        elif rule_type == "pattern_match":
            field = rule.get("field")
            pattern = rule.get("pattern")
            if isinstance(data, dict) and field in data and isinstance(data[field], str):
                if not re.match(pattern, data[field]):
                    errors.append(f"rule '{rule_name}': field '{field}' does not match pattern")

        elif rule_type == "value_in_list":
            field = rule.get("field")
            allowed_values = rule.get("allowed_values", [])
            if isinstance(data, dict) and field in data:
                if data[field] not in allowed_values:
                    errors.append(f"rule '{rule_name}': field '{field}' value not in allowed list")

        elif rule_type == "custom_check":
            check_name = rule.get("check")
            if check_name == "non_empty_string":
                field = rule.get("field")
                if isinstance(data, dict) and (field not in data or not data[field]):
                    errors.append(f"rule '{rule_name}': field '{field}' must be non-empty string")

    return errors, warnings
