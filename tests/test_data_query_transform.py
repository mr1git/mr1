"""Comprehensive tests for data_query_transform tool with SQLite support."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mr1.worker_runner import MockRunner
from mr1.scheduler import Scheduler
from mr1.tools import ToolConfigError, default_tool_registry
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="agent", id="MR1")


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def scheduler(store):
    sched = Scheduler(store, MockRunner(), auto_tick=False, agent_id="scheduler")
    yield sched
    sched.shutdown()


@pytest.fixture
def sample_db(tmp_path):
    """Create a sample SQLite database for testing."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            status TEXT
        )
    """)

    users = [
        (1, "Alice", "alice@example.com", 30),
        (2, "Bob", "bob@example.com", 25),
        (3, "Charlie", "charlie@example.com", 35),
        (4, "Diana", "diana@example.com", 28),
        (5, "Eve", "eve@example.com", 32),
    ]

    orders = [
        (1, 1, 100.50, "completed"),
        (2, 1, 250.00, "completed"),
        (3, 2, 50.25, "pending"),
        (4, 3, 500.00, "completed"),
        (5, 2, 75.00, "cancelled"),
        (6, 4, 200.00, "completed"),
        (7, 5, 150.00, "pending"),
    ]

    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users)
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()

    return db_path


def _task_by_label(wf, label):
    return wf.task_by_label(label)


def _tool_workflow(task: dict, downstream: dict | None = None) -> dict:
    tasks = [task]
    if downstream is not None:
        tasks.append(downstream)
    return {"title": "Tool workflow", "tasks": tasks}


# ============================================================================
# SQLite Query Validation Tests
# ============================================================================

class TestDataQueryTransformSQLiteValidation:
    def test_sqlite_missing_db_path_rejected(self):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="requires db_path"):
            registry.validate_spec("data_query_transform", {
                "mode": "sqlite",
                "query": "SELECT * FROM users",
            })

    def test_sqlite_empty_db_path_rejected(self):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="requires db_path"):
            registry.validate_spec("data_query_transform", {
                "mode": "sqlite",
                "db_path": "",
                "query": "SELECT * FROM users",
            })

    def test_sqlite_missing_query_rejected(self):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="requires query"):
            registry.validate_spec("data_query_transform", {
                "mode": "sqlite",
                "db_path": "/tmp/test.db",
            })

    def test_sqlite_empty_query_rejected(self):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="requires query"):
            registry.validate_spec("data_query_transform", {
                "mode": "sqlite",
                "db_path": "/tmp/test.db",
                "query": "",
            })

    def test_sqlite_valid_config_accepted(self):
        registry = default_tool_registry()
        registry.validate_spec("data_query_transform", {
            "mode": "sqlite",
            "db_path": "/tmp/test.db",
            "query": "SELECT * FROM users",
        })

    def test_sqlite_valid_config_with_params_accepted(self):
        registry = default_tool_registry()
        registry.validate_spec("data_query_transform", {
            "mode": "sqlite",
            "db_path": "/tmp/test.db",
            "query": "SELECT * FROM users WHERE id = ?",
            "params": [1],
        })


# ============================================================================
# SQLite Query Execution Tests
# ============================================================================

class TestDataQueryTransformSQLiteExecution:
    def test_sqlite_invalid_query_fails(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM nonexistent_table",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.FAILED

    def test_sqlite_simple_query_succeeds(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        assert output is not None
        result = output.data.get("result", [])
        assert len(result) == 5
        assert result[0]["name"] == "Alice"

    def test_sqlite_query_with_where_clause(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users WHERE age > 30",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) == 2
        assert all(r["age"] > 30 for r in result)

    def test_sqlite_query_with_parameters(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users WHERE id = ?",
                    "params": [2],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) == 1
        assert result[0]["name"] == "Bob"

    def test_sqlite_query_with_join(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": """
                        SELECT u.name, COUNT(o.id) as order_count
                        FROM users u
                        LEFT JOIN orders o ON u.id = o.user_id
                        GROUP BY u.id
                    """,
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) == 5
        assert any(r["order_count"] >= 2 for r in result)

    def test_sqlite_invalid_query_fails(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "INVALID SQL SYNTAX HERE",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.FAILED

    def test_sqlite_query_empty_result(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users WHERE id = 999",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) == 0

    def test_sqlite_query_with_limit(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users",
                    "limit": 2,
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) <= 2


# ============================================================================
# SQLite Query with Transformation Tests
# ============================================================================

class TestDataQueryTransformSQLiteTransform:
    def test_sqlite_query_with_count_transform(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users",
                    "transform_op": "count",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        assert output.text == "5" or "5" in output.text

    def test_sqlite_query_with_keys_transform(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users LIMIT 1",
                    "transform_op": "keys",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        assert "id" in output.text
        assert "name" in output.text

    def test_sqlite_query_with_summary_transform(self, scheduler, store, sample_db):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users",
                    "transform_op": "summary",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        text_data = json.loads(output.text)
        assert "row_count" in text_data
        assert text_data["row_count"] == 5
        assert "columns" in text_data


# ============================================================================
# Data Transformation Validation Tests
# ============================================================================

class TestDataQueryTransformDataValidation:
    def test_data_transform_missing_operation_rejected(self):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="operation must be"):
            registry.validate_spec("data_query_transform", {
                "mode": "data",
                "operation": "invalid",
                "input_data": {},
            })

    def test_data_transform_filter_missing_pattern_rejected(self):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="requires pattern"):
            registry.validate_spec("data_query_transform", {
                "mode": "data",
                "operation": "filter",
                "input_data": [],
            })

    def test_data_transform_invalid_aggregate_function_rejected(self):
        registry = default_tool_registry()
        with pytest.raises(ToolConfigError, match="aggregate function"):
            registry.validate_spec("data_query_transform", {
                "mode": "data",
                "operation": "aggregate",
                "function": "invalid",
                "input_data": [],
            })

    def test_data_transform_valid_filter_accepted(self):
        registry = default_tool_registry()
        registry.validate_spec("data_query_transform", {
            "mode": "data",
            "operation": "filter",
            "pattern": "test",
            "input_data": ["test", "data"],
        })


# ============================================================================
# Data Transformation Execution Tests
# ============================================================================

class TestDataQueryTransformDataExecution:
    def test_data_filter_operation(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "transform",
                "title": "Transform",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "data",
                    "operation": "filter",
                    "pattern": "alice",
                    "input_data": ["alice", "bob", "alice_smith"],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "transform")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        assert output is not None
        result = output.data.get("result", [])
        assert len(result) >= 1

    def test_data_aggregate_sum(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "transform",
                "title": "Transform",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "data",
                    "operation": "aggregate",
                    "function": "sum",
                    "input_data": [10, 20, 30],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "transform")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        assert "60" in output.text

    def test_data_aggregate_count(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "transform",
                "title": "Transform",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "data",
                    "operation": "aggregate",
                    "function": "count",
                    "input_data": [1, 2, 3, 4, 5],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "transform")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        assert "5" in output.text

    def test_data_aggregate_avg(self, scheduler, store):
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "transform",
                "title": "Transform",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "data",
                    "operation": "aggregate",
                    "function": "avg",
                    "input_data": [10, 20, 30],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "transform")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        assert "20" in output.text


# ============================================================================
# SQL Injection Protection Tests
# ============================================================================

class TestDataQueryTransformSQLInjectionProtection:
    def test_parameterized_query_prevents_injection(self, scheduler, store, sample_db):
        """Test that parameterized queries prevent SQL injection."""
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users WHERE name = ?",
                    "params": ["Alice' OR '1'='1"],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) == 0

    def test_safe_parameter_substitution(self, scheduler, store, sample_db):
        """Test that parameter substitution works correctly."""
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(sample_db),
                    "query": "SELECT * FROM users WHERE name = ? AND age > ?",
                    "params": ["Alice", 25],
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) == 1
        assert result[0]["name"] == "Alice"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestDataQueryTransformErrorHandling:
    def test_malformed_json_in_transform_pattern(self, scheduler, store):
        """Test error handling for malformed JSON in transform operation."""
        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "transform",
                "title": "Transform",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "data",
                    "operation": "transform",
                    "pattern": "{invalid json",
                    "input_data": {"test": "data"},
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "transform")
        assert task.status is TaskStatus.SUCCEEDED

    def test_database_operations(self, tmp_path, scheduler, store):
        """Test normal database operations."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        cursor.execute("INSERT INTO test VALUES (1, 'test')")
        conn.commit()
        conn.close()

        wf_id = scheduler.submit_workflow(
            _tool_workflow({
                "label": "query",
                "title": "Query",
                "task_kind": "tool",
                "tool_type": "data_query_transform",
                "tool_config": {
                    "mode": "sqlite",
                    "db_path": str(db_path),
                    "query": "SELECT * FROM test",
                },
            }),
            PROV,
        )
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task = _task_by_label(wf, "query")
        assert task.status is TaskStatus.SUCCEEDED
        output = store.load_task_output(wf_id, task.task_id)
        result = output.data.get("result", [])
        assert len(result) == 1
