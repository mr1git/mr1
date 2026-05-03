"""Test suite for WorkflowInspector utility class."""

import json
import tempfile
from pathlib import Path

import pytest

from mr1.developer_tools import WorkflowInspector


class TestWorkflowInspectorBasic:
    """Test basic loading and parsing functionality."""

    def test_load_yaml_workflow(self, tmp_path):
        """Test loading a simple YAML workflow."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: validate
    type: script
    status: pending
  - label: test
    type: test
    depends_on: validate
    status: pending
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        workflow = inspector.load()

        assert "tasks" in workflow
        assert len(workflow["tasks"]) == 2

    def test_load_json_workflow(self, tmp_path):
        """Test loading a JSON workflow."""
        workflow_file = tmp_path / "workflow.json"
        workflow_data = {
            "tasks": [
                {"label": "build", "type": "build"},
                {"label": "test", "type": "test", "depends_on": "build"},
            ]
        }
        workflow_file.write_text(json.dumps(workflow_data))

        inspector = WorkflowInspector(str(workflow_file))
        workflow = inspector.load()

        assert len(workflow["tasks"]) == 2

    def test_load_nonexistent_file(self):
        """Test error handling for missing files."""
        inspector = WorkflowInspector("/nonexistent/workflow.yaml")

        with pytest.raises(WorkflowInspector.WorkflowError) as exc_info:
            inspector.load()

        assert "not found" in str(exc_info.value).lower()

    def test_load_empty_file(self, tmp_path):
        """Test error handling for empty files."""
        workflow_file = tmp_path / "empty.yaml"
        workflow_file.write_text("")

        inspector = WorkflowInspector(str(workflow_file))

        with pytest.raises(WorkflowInspector.WorkflowError) as exc_info:
            inspector.load()

        # Empty YAML parses as None, which is not a dict
        assert "dictionary" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()

    def test_load_invalid_format(self, tmp_path):
        """Test error handling for invalid JSON/YAML."""
        workflow_file = tmp_path / "invalid.yaml"
        workflow_file.write_text("{ invalid yaml [ } }")

        inspector = WorkflowInspector(str(workflow_file))

        with pytest.raises(WorkflowInspector.WorkflowError) as exc_info:
            inspector.load()

        assert "parse" in str(exc_info.value).lower()


class TestWorkflowInspectorDAG:
    """Test DAG parsing and dependency handling."""

    def test_parse_dag_no_dependencies(self, tmp_path):
        """Test DAG parsing for tasks with no dependencies."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: build
    type: build
  - label: test
    type: test
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag["build"] == []
        assert dag["test"] == []

    def test_parse_dag_single_dependency(self, tmp_path):
        """Test DAG parsing with single dependencies."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: build
    type: build
  - label: test
    type: test
    depends_on: build
  - label: deploy
    type: deployment
    depends_on: test
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag["build"] == []
        assert dag["test"] == ["build"]
        assert dag["deploy"] == ["test"]

    def test_parse_dag_multiple_dependencies(self, tmp_path):
        """Test DAG parsing with multiple dependencies."""
        workflow_file = tmp_path / "workflow.json"
        workflow_data = {
            "tasks": [
                {"label": "build", "type": "build"},
                {"label": "unit_test", "type": "test", "depends_on": "build"},
                {"label": "integration_test", "type": "test", "depends_on": "build"},
                {
                    "label": "deploy",
                    "type": "deployment",
                    "depends_on": ["unit_test", "integration_test"],
                },
            ]
        }
        workflow_file.write_text(json.dumps(workflow_data))

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag["deploy"] == ["unit_test", "integration_test"]

    def test_parse_dag_requires_load(self, tmp_path):
        """Test that parse_dag requires load to be called first."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text("tasks: []")

        inspector = WorkflowInspector(str(workflow_file))

        with pytest.raises(WorkflowInspector.WorkflowError) as exc_info:
            inspector.parse_dag()

        assert "load" in str(exc_info.value).lower()


class TestWorkflowInspectorCycles:
    """Test circular dependency detection."""

    def test_detect_simple_cycle(self, tmp_path):
        """Test detection of simple cycle A -> B -> A."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: A
    type: task
    depends_on: B
  - label: B
    type: task
    depends_on: A
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()

        with pytest.raises(WorkflowInspector.CyclicDependencyError) as exc_info:
            inspector.parse_dag()

        assert "circular" in str(exc_info.value).lower()

    def test_detect_complex_cycle(self, tmp_path):
        """Test detection of complex cycle A -> B -> C -> A."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: A
    type: task
    depends_on: B
  - label: B
    type: task
    depends_on: C
  - label: C
    type: task
    depends_on: A
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()

        with pytest.raises(WorkflowInspector.CyclicDependencyError):
            inspector.parse_dag()

    def test_no_cycle_in_dag(self, tmp_path):
        """Test that valid DAGs don't raise cycle errors."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: A
    type: task
  - label: B
    type: task
    depends_on: A
  - label: C
    type: task
    depends_on: A
  - label: D
    type: task
    depends_on: [B, C]
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        dag = inspector.parse_dag()  # Should not raise

        assert dag["D"] == ["B", "C"]


class TestWorkflowInspectorTaskStatus:
    """Test task status retrieval."""

    def test_get_task_status_existing(self, tmp_path):
        """Test getting status of existing task."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: build
    type: build
    status: running
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()

        assert inspector.get_task_status("build") == "running"

    def test_get_task_status_default(self, tmp_path):
        """Test default status for task without status field."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: build
    type: build
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()

        assert inspector.get_task_status("build") == "pending"

    def test_get_task_status_nonexistent(self, tmp_path):
        """Test status for nonexistent task."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text("tasks: []")

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()

        assert inspector.get_task_status("nonexistent") == "unknown"


class TestWorkflowInspectorSummary:
    """Test get_summary output."""

    def test_summary_structure(self, tmp_path):
        """Test that summary has required structure."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: build
    type: build
  - label: test
    type: test
    depends_on: build
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        summary = inspector.get_summary()

        assert "tasks" in summary
        assert "dag" in summary
        assert "task_count" in summary
        assert "output_refs" in summary
        assert summary["task_count"] == 2

    def test_summary_task_list(self, tmp_path):
        """Test task list in summary."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: build
    type: build
    status: pending
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        summary = inspector.get_summary()

        assert len(summary["tasks"]) == 1
        task = summary["tasks"][0]
        assert task["label"] == "build"
        assert task["type"] == "build"
        assert task["status"] == "pending"

    def test_summary_output_refs_discovery(self, tmp_path):
        """Test output reference discovery in summary."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: query
    type: fetch
    outputs:
      path: ./result.json

  - label: transform
    type: transform
    depends_on: query
    input: $task.query.outputs.result
    output: $task.transform.outputs.transformed
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        summary = inspector.get_summary()

        assert "$task.query.outputs.result" in summary["output_refs"]
        assert "$task.transform.outputs.transformed" in summary["output_refs"]

    def test_summary_json_serializable(self, tmp_path):
        """Test that summary is JSON-serializable."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: build
    type: build
  - label: test
    type: test
    depends_on: build
"""
        )

        inspector = WorkflowInspector(str(workflow_file))
        inspector.load()
        summary = inspector.get_summary()

        # Should not raise
        json_str = json.dumps(summary)
        assert json_str  # Non-empty

    def test_summary_requires_load(self, tmp_path):
        """Test that get_summary requires load to be called first."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text("tasks: []")

        inspector = WorkflowInspector(str(workflow_file))

        with pytest.raises(WorkflowInspector.WorkflowError) as exc_info:
            inspector.get_summary()

        assert "load" in str(exc_info.value).lower()


class TestWorkflowInspectorIntegration:
    """Integration tests with real workflow examples."""

    def test_full_workflow_analysis(self, tmp_path):
        """Test complete analysis flow on realistic workflow."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(
            """
tasks:
  - label: checkout
    type: vcs
    command: git checkout

  - label: build
    type: build
    depends_on: checkout
    command: make build

  - label: unit_test
    type: test
    depends_on: build
    command: make test

  - label: integration_test
    type: test
    depends_on: build
    command: make test-integration

  - label: lint
    type: lint
    depends_on: checkout
    command: make lint

  - label: deploy
    type: deployment
    depends_on: [unit_test, integration_test, lint]
    command: ./deploy.sh
"""
        )

        inspector = WorkflowInspector(str(workflow_file))

        # Load
        workflow = inspector.load()
        assert workflow is not None

        # Parse DAG
        dag = inspector.parse_dag()
        assert len(dag) == 6
        assert dag["deploy"] == ["unit_test", "integration_test", "lint"]

        # Get summary
        summary = inspector.get_summary()
        assert summary["task_count"] == 6
        assert len(summary["tasks"]) == 6
        assert summary["dag"] == dag
