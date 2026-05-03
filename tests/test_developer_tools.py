"""
Tests for developer tools.

Tests verify that the tools safely inspect system state without modification
and handle edge cases gracefully.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mr1.developer_tools import (
    WorkflowStateInspectorTool,
    MemoryGraphNavigatorTool,
    CapabilityCallTracerTool,
    WorkflowInspector,
    DEVELOPER_TOOLS,
)
from mr1.tools import ToolConfigError
from mr1.workflow_models import (
    Workflow,
    Task,
    TaskStatus,
    WorkflowStatus,
    Provenance,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_store():
    """Create a mock WorkflowStore."""
    store = MagicMock()
    return store


@pytest.fixture
def sample_workflow():
    """Create a sample workflow with multiple tasks."""
    workflow = Workflow(
        workflow_id="wf-20260501T000000-abc123",
        title="Test Workflow",
        status=WorkflowStatus.RUNNING,
        created_at="2026-05-01T00:00:00+00:00",
        created_by=Provenance(type="test", id="test_user"),
        tasks=[
            Task(
                task_id="tk-1",
                workflow_id="wf-20260501T000000-abc123",
                label="task1",
                title="Task 1",
                task_kind="agent",
                agent_type="mrn",
                prompt="Test task 1",
                status=TaskStatus.SUCCEEDED,
                created_at="2026-05-01T00:00:00+00:00",
                started_at="2026-05-01T00:00:01+00:00",
                finished_at="2026-05-01T00:00:05+00:00",
            ),
            Task(
                task_id="tk-2",
                workflow_id="wf-20260501T000000-abc123",
                label="task2",
                title="Task 2",
                task_kind="tool",
                agent_type=None,
                tool_type="read_file",
                prompt="Test task 2",
                status=TaskStatus.FAILED,
                last_error="File not found",
                created_at="2026-05-01T00:00:00+00:00",
                started_at="2026-05-01T00:00:06+00:00",
                finished_at="2026-05-01T00:00:10+00:00",
            ),
            Task(
                task_id="tk-3",
                workflow_id="wf-20260501T000000-abc123",
                label="task3",
                title="Task 3",
                task_kind="watcher",
                agent_type=None,
                watcher_type="delay",
                prompt="Test task 3",
                status=TaskStatus.RUNNING,
                created_at="2026-05-01T00:00:00+00:00",
                started_at="2026-05-01T00:00:11+00:00",
            ),
        ],
    )
    return workflow


@pytest.fixture
def sample_task():
    """Create a sample inspector task."""
    return Task(
        task_id="tk-inspector",
        workflow_id="wf-20260501T000000-inspector",
        label="inspect",
        title="Workflow Inspector",
        task_kind="tool",
        agent_type=None,
        tool_type="workflow_state_inspector",
        prompt="Inspect workflow",
        status=TaskStatus.RUNNING,
    )


@pytest.fixture
def sample_workflow_for_inspector():
    """Create a workflow for the inspector task to run in."""
    return Workflow(
        workflow_id="wf-20260501T000000-inspector",
        title="Inspector Workflow",
        status=WorkflowStatus.RUNNING,
        created_at="2026-05-01T00:00:00+00:00",
        tasks=[],
    )


# ============================================================================
# WorkflowStateInspectorTool Tests
# ============================================================================


class TestWorkflowStateInspectorTool:
    """Tests for workflow state inspection."""

    def test_validate_config_missing_workflow_id(self):
        """Config validation requires workflow_id."""
        tool = WorkflowStateInspectorTool()
        with pytest.raises(ToolConfigError, match="workflow_id"):
            tool.validate_config({})

    def test_validate_config_invalid_workflow_id(self):
        """Config validation requires workflow_id to be string."""
        tool = WorkflowStateInspectorTool()
        with pytest.raises(ToolConfigError, match="workflow_id"):
            tool.validate_config({"workflow_id": 123})

    def test_validate_config_invalid_include_tasks(self):
        """Config validation requires include_tasks to be boolean."""
        tool = WorkflowStateInspectorTool()
        with pytest.raises(ToolConfigError, match="include_tasks"):
            tool.validate_config({"workflow_id": "wf-123", "include_tasks": "yes"})

    def test_validate_config_invalid_task_filter(self):
        """Config validation requires task_filter to be string or omitted."""
        tool = WorkflowStateInspectorTool()
        with pytest.raises(ToolConfigError, match="task_filter"):
            tool.validate_config({"workflow_id": "wf-123", "task_filter": 123})

    def test_validate_config_valid(self):
        """Valid configs pass validation."""
        tool = WorkflowStateInspectorTool()
        tool.validate_config({"workflow_id": "wf-123"})
        tool.validate_config({"workflow_id": "wf-123", "include_tasks": True})
        tool.validate_config(
            {"workflow_id": "wf-123", "include_tasks": False, "task_filter": "error"}
        )

    def test_run_workflow_not_found(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns failed result when workflow not found."""
        mock_store.load_workflow.return_value = None

        tool = WorkflowStateInspectorTool()
        sample_task.tool_config = {"workflow_id": "wf-missing"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "failed"
        assert "does not exist" in result.text.lower()
        assert result.error == "not_found"

    def test_run_workflow_success(self, mock_store, sample_task, sample_workflow, sample_workflow_for_inspector):
        """Successfully inspects workflow state."""
        mock_store.load_workflow.return_value = sample_workflow

        tool = WorkflowStateInspectorTool()
        sample_task.tool_config = {"workflow_id": "wf-20260501T000000-abc123"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Workflow State" in result.text
        assert "Task Summary" in result.text
        assert sample_workflow.workflow_id in result.text

    def test_run_without_task_details(self, mock_store, sample_task, sample_workflow, sample_workflow_for_inspector):
        """Can skip task details when requested."""
        mock_store.load_workflow.return_value = sample_workflow

        tool = WorkflowStateInspectorTool()
        sample_task.tool_config = {
            "workflow_id": "wf-20260501T000000-abc123",
            "include_tasks": False,
        }

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "## Tasks" not in result.text  # No detailed tasks section

    def test_run_with_task_filter(self, mock_store, sample_task, sample_workflow, sample_workflow_for_inspector):
        """Can filter tasks by name."""
        mock_store.load_workflow.return_value = sample_workflow

        tool = WorkflowStateInspectorTool()
        sample_task.tool_config = {
            "workflow_id": "wf-20260501T000000-abc123",
            "include_tasks": True,
            "task_filter": "task2",
        }

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "task2" in result.text
        assert "task1" not in result.text

    def test_task_status_counting(self):
        """Tool correctly counts task statuses."""
        tool = WorkflowStateInspectorTool()
        tasks = [
            Task(
                task_id="1",
                workflow_id="w",
                label="a",
                title="A",
                task_kind="agent",
                agent_type="mrn",
                prompt="p",
                status=TaskStatus.SUCCEEDED,
            ),
            Task(
                task_id="2",
                workflow_id="w",
                label="b",
                title="B",
                task_kind="agent",
                agent_type="mrn",
                prompt="p",
                status=TaskStatus.SUCCEEDED,
            ),
            Task(
                task_id="3",
                workflow_id="w",
                label="c",
                title="C",
                task_kind="tool",
                agent_type=None,
                prompt="p",
                status=TaskStatus.FAILED,
            ),
        ]

        counts = tool._count_task_statuses(tasks)
        assert counts["succeeded"] == 2
        assert counts["failed"] == 1

    def test_duration_calculation(self):
        """Tool correctly calculates task duration."""
        tool = WorkflowStateInspectorTool()

        duration = tool._duration_str(
            "2026-05-01T00:00:00+00:00",
            "2026-05-01T00:00:05+00:00",
        )
        assert "5.0s" in duration

    def test_duration_missing_timestamps(self):
        """Tool handles missing timestamps gracefully."""
        tool = WorkflowStateInspectorTool()

        assert tool._duration_str(None, None) == "unknown"
        assert tool._duration_str("2026-05-01T00:00:00+00:00", None) == "unknown"


# ============================================================================
# MemoryGraphNavigatorTool Tests
# ============================================================================


class TestMemoryGraphNavigatorTool:
    """Tests for memory graph navigation."""

    def test_validate_config_missing_query_type(self):
        """Config validation requires query_type."""
        tool = MemoryGraphNavigatorTool()
        with pytest.raises(ToolConfigError, match="query_type"):
            tool.validate_config({})

    def test_validate_config_invalid_query_type(self):
        """Config validation requires valid query_type."""
        tool = MemoryGraphNavigatorTool()
        with pytest.raises(ToolConfigError, match="query_type"):
            tool.validate_config({"query_type": "invalid"})

    def test_validate_config_find_without_search_term(self):
        """Find queries require search_term."""
        tool = MemoryGraphNavigatorTool()
        with pytest.raises(ToolConfigError, match="search_term"):
            tool.validate_config({"query_type": "find"})

    def test_validate_config_find_empty_search_term(self):
        """Find queries require non-empty search_term."""
        tool = MemoryGraphNavigatorTool()
        with pytest.raises(ToolConfigError, match="search_term"):
            tool.validate_config({"query_type": "find", "search_term": ""})

    def test_validate_config_valid_relationships(self):
        """Valid relationships config passes."""
        tool = MemoryGraphNavigatorTool()
        tool.validate_config({"query_type": "relationships"})

    def test_validate_config_valid_stats(self):
        """Valid stats config passes."""
        tool = MemoryGraphNavigatorTool()
        tool.validate_config({"query_type": "stats"})

    def test_validate_config_valid_find(self):
        """Valid find config passes."""
        tool = MemoryGraphNavigatorTool()
        tool.validate_config({"query_type": "find", "search_term": "test"})

    def test_run_relationships(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns memory relationships."""
        tool = MemoryGraphNavigatorTool()
        sample_task.tool_config = {"query_type": "relationships"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Memory Graph Relationships" in result.text
        assert "Workflow metadata" in result.text

    def test_run_stats(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns memory statistics."""
        tool = MemoryGraphNavigatorTool()
        sample_task.tool_config = {"query_type": "stats"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Memory Graph Statistics" in result.text
        assert "Workflow execution timeline" in result.text

    def test_run_find(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns search results."""
        tool = MemoryGraphNavigatorTool()
        sample_task.tool_config = {"query_type": "find", "search_term": "database"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Memory Search" in result.text
        assert "database" in result.text

    def test_run_invalid_query_type(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns failed result for invalid query type."""
        tool = MemoryGraphNavigatorTool()
        sample_task.tool_config = {"query_type": "invalid"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "failed"
        assert "Invalid query type" in result.summary


# ============================================================================
# CapabilityCallTracerTool Tests
# ============================================================================


class TestCapabilityCallTracerTool:
    """Tests for capability call tracing."""

    def test_validate_config_missing_trace_type(self):
        """Config validation requires trace_type."""
        tool = CapabilityCallTracerTool()
        with pytest.raises(ToolConfigError, match="trace_type"):
            tool.validate_config({})

    def test_validate_config_invalid_trace_type(self):
        """Config validation requires valid trace_type."""
        tool = CapabilityCallTracerTool()
        with pytest.raises(ToolConfigError, match="trace_type"):
            tool.validate_config({"trace_type": "invalid"})

    def test_validate_config_invalid_task_id(self):
        """Config validation requires task_id to be string or omitted."""
        tool = CapabilityCallTracerTool()
        with pytest.raises(ToolConfigError, match="task_id"):
            tool.validate_config({"trace_type": "pending", "task_id": 123})

    def test_validate_config_valid_pending(self):
        """Valid pending config passes."""
        tool = CapabilityCallTracerTool()
        tool.validate_config({"trace_type": "pending"})

    def test_validate_config_valid_approved(self):
        """Valid approved config passes."""
        tool = CapabilityCallTracerTool()
        tool.validate_config({"trace_type": "approved"})

    def test_validate_config_valid_blocked(self):
        """Valid blocked config passes."""
        tool = CapabilityCallTracerTool()
        tool.validate_config({"trace_type": "blocked"})

    def test_validate_config_valid_summary(self):
        """Valid summary config passes."""
        tool = CapabilityCallTracerTool()
        tool.validate_config({"trace_type": "summary"})

    def test_run_pending(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns pending approvals."""
        tool = CapabilityCallTracerTool()
        sample_task.tool_config = {"trace_type": "pending"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Pending Capability Approvals" in result.text
        assert "blocking agent execution" in result.text

    def test_run_approved(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns approved capabilities."""
        tool = CapabilityCallTracerTool()
        sample_task.tool_config = {"trace_type": "approved"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Approved Capabilities" in result.text
        assert "audit trail" in result.text

    def test_run_blocked(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns blocked capabilities."""
        tool = CapabilityCallTracerTool()
        sample_task.tool_config = {"trace_type": "blocked"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Blocked Capabilities" in result.text
        assert "policy" in result.text

    def test_run_blocked_with_task_filter(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Can filter blocked approvals by task."""
        tool = CapabilityCallTracerTool()
        sample_task.tool_config = {"trace_type": "blocked", "task_id": "tk-specific"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "tk-specific" in result.text

    def test_run_summary(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns capability summary."""
        tool = CapabilityCallTracerTool()
        sample_task.tool_config = {"trace_type": "summary"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "succeeded"
        assert "Capability Approval Summary" in result.text
        assert "Approved vs. blocked" in result.text

    def test_run_invalid_trace_type(self, mock_store, sample_task, sample_workflow_for_inspector):
        """Returns failed result for invalid trace type."""
        tool = CapabilityCallTracerTool()
        sample_task.tool_config = {"trace_type": "invalid"}

        result = tool.run(sample_task, mock_store, sample_workflow_for_inspector)

        assert result.state == "failed"
        assert "Invalid trace type" in result.summary


# ============================================================================
# Tool Registry Tests
# ============================================================================


class TestDeveloperToolsRegistry:
    """Tests for tool registry entries."""

    def test_registry_has_all_tools(self):
        """Registry contains all three tools."""
        assert "workflow_state_inspector" in DEVELOPER_TOOLS
        assert "memory_graph_navigator" in DEVELOPER_TOOLS
        assert "capability_call_tracer" in DEVELOPER_TOOLS

    def test_registry_entries_have_runner(self):
        """Each registry entry has a runner instance."""
        assert isinstance(DEVELOPER_TOOLS["workflow_state_inspector"]["runner"], WorkflowStateInspectorTool)
        assert isinstance(DEVELOPER_TOOLS["memory_graph_navigator"]["runner"], MemoryGraphNavigatorTool)
        assert isinstance(DEVELOPER_TOOLS["capability_call_tracer"]["runner"], CapabilityCallTracerTool)

    def test_registry_entries_have_description(self):
        """Each registry entry has a description."""
        for tool_name in DEVELOPER_TOOLS:
            assert "description" in DEVELOPER_TOOLS[tool_name]
            assert len(DEVELOPER_TOOLS[tool_name]["description"]) > 0

    def test_registry_entries_have_config_schema(self):
        """Each registry entry has a config schema."""
        for tool_name in DEVELOPER_TOOLS:
            assert "config_schema" in DEVELOPER_TOOLS[tool_name]
            assert "config_shape" in DEVELOPER_TOOLS[tool_name]

    def test_registry_entries_have_examples(self):
        """Each registry entry has examples."""
        for tool_name in DEVELOPER_TOOLS:
            assert "examples" in DEVELOPER_TOOLS[tool_name]
            assert len(DEVELOPER_TOOLS[tool_name]["examples"]) > 0

    def test_registry_examples_are_valid(self):
        """Registry examples are syntactically valid."""
        for tool_name, entry in DEVELOPER_TOOLS.items():
            runner = entry["runner"]
            for example in entry["examples"]:
                assert "config" in example
                # Should not raise validation error
                runner.validate_config(example["config"])


# ============================================================================
# WorkflowInspector Fixtures
# ============================================================================


@pytest.fixture
def tmp_workflow_dir():
    """Create a temporary directory for test workflow files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def linear_dag_json(tmp_workflow_dir):
    """Linear DAG: A -> B -> C."""
    workflow = {
        "name": "linear_pipeline",
        "tasks": [
            {"label": "validate", "type": "script", "depends_on": []},
            {"label": "build", "type": "script", "depends_on": "validate"},
            {"label": "deploy", "type": "script", "depends_on": "build"},
        ],
    }
    path = tmp_workflow_dir / "linear.json"
    path.write_text(json.dumps(workflow))
    return path


@pytest.fixture
def parallel_dag_json(tmp_workflow_dir):
    """Parallel DAG: A -> C and B -> C."""
    workflow = {
        "name": "parallel_pipeline",
        "tasks": [
            {"label": "validate_code", "type": "script", "depends_on": []},
            {"label": "validate_config", "type": "script", "depends_on": []},
            {"label": "merge", "type": "script", "depends_on": ["validate_code", "validate_config"]},
        ],
    }
    path = tmp_workflow_dir / "parallel.json"
    path.write_text(json.dumps(workflow))
    return path


@pytest.fixture
def complex_dag_json(tmp_workflow_dir):
    """Complex DAG with branches: multiple paths and convergence."""
    workflow = {
        "name": "complex_pipeline",
        "tasks": [
            {"label": "init", "type": "script", "depends_on": []},
            {"label": "lint", "type": "script", "depends_on": "init"},
            {"label": "test_unit", "type": "script", "depends_on": "init"},
            {"label": "test_integration", "type": "script", "depends_on": "init"},
            {"label": "analyze", "type": "script", "depends_on": ["lint", "test_unit"]},
            {"label": "build", "type": "script", "depends_on": "analyze"},
            {"label": "verify", "type": "script", "depends_on": ["build", "test_integration"]},
        ],
    }
    path = tmp_workflow_dir / "complex.json"
    path.write_text(json.dumps(workflow))
    return path


@pytest.fixture
def cyclic_dag_json(tmp_workflow_dir):
    """Cyclic DAG: A -> B -> C -> A (circular dependency)."""
    workflow = {
        "name": "cyclic_pipeline",
        "tasks": [
            {"label": "taskA", "type": "script", "depends_on": "taskC"},
            {"label": "taskB", "type": "script", "depends_on": "taskA"},
            {"label": "taskC", "type": "script", "depends_on": "taskB"},
        ],
    }
    path = tmp_workflow_dir / "cyclic.json"
    path.write_text(json.dumps(workflow))
    return path


@pytest.fixture
def yaml_workflow(tmp_workflow_dir):
    """Valid YAML workflow."""
    yaml_content = """
name: yaml_pipeline
tasks:
  - label: setup
    type: script
    depends_on: []
  - label: run
    type: script
    depends_on: setup
    status: pending
  - label: cleanup
    type: script
    depends_on: run
    status: running
"""
    path = tmp_workflow_dir / "workflow.yaml"
    path.write_text(yaml_content)
    return path


@pytest.fixture
def workflow_with_output_refs(tmp_workflow_dir):
    """Workflow containing output references."""
    workflow = {
        "name": "workflow_with_refs",
        "tasks": [
            {"label": "extract", "type": "script", "depends_on": []},
            {
                "label": "process",
                "type": "script",
                "depends_on": "extract",
                "input": "$task.extract.outputs.data",
            },
            {
                "label": "report",
                "type": "script",
                "depends_on": "process",
                "params": {
                    "result": "${task.process.outputs.result}",
                    "timestamp": "$task.extract.outputs.time",
                },
            },
        ],
    }
    path = tmp_workflow_dir / "with_refs.json"
    path.write_text(json.dumps(workflow))
    return path


@pytest.fixture
def workflow_with_statuses(tmp_workflow_dir):
    """Workflow with various task statuses."""
    workflow = {
        "name": "workflow_with_statuses",
        "tasks": [
            {"label": "completed_task", "type": "script", "status": "succeeded"},
            {"label": "running_task", "type": "script", "status": "running"},
            {"label": "failed_task", "type": "script", "status": "failed"},
            {"label": "pending_task", "type": "script"},  # No status = pending
        ],
    }
    path = tmp_workflow_dir / "with_statuses.json"
    path.write_text(json.dumps(workflow))
    return path


@pytest.fixture
def workflow_with_steps(tmp_workflow_dir):
    """Workflow using 'steps' instead of 'tasks'."""
    workflow = {
        "name": "workflow_with_steps",
        "steps": [
            {"name": "step1", "type": "script", "depends_on": []},
            {"name": "step2", "type": "script", "depends_on": "step1"},
        ],
    }
    path = tmp_workflow_dir / "with_steps.json"
    path.write_text(json.dumps(workflow))
    return path


# ============================================================================
# WorkflowInspector Tests
# ============================================================================


class TestWorkflowInspectorLoad:
    """Tests for WorkflowInspector.load()"""

    def test_load_valid_yaml(self, yaml_workflow):
        """Test loading a valid YAML workflow file."""
        inspector = WorkflowInspector(str(yaml_workflow))
        workflow = inspector.load()

        assert workflow is not None
        assert isinstance(workflow, dict)
        assert workflow["name"] == "yaml_pipeline"
        assert "tasks" in workflow
        assert len(workflow["tasks"]) == 3

    def test_load_valid_json(self, linear_dag_json):
        """Test loading a valid JSON workflow file."""
        inspector = WorkflowInspector(str(linear_dag_json))
        workflow = inspector.load()

        assert workflow is not None
        assert isinstance(workflow, dict)
        assert workflow["name"] == "linear_pipeline"
        assert "tasks" in workflow
        assert len(workflow["tasks"]) == 3

    def test_load_missing_file(self, tmp_workflow_dir):
        """Test that loading a missing file raises WorkflowError."""
        missing_path = tmp_workflow_dir / "nonexistent.json"
        inspector = WorkflowInspector(str(missing_path))

        with pytest.raises(WorkflowInspector.WorkflowError, match="Workflow file not found"):
            inspector.load()

    def test_load_malformed_yaml(self, tmp_workflow_dir):
        """Test that loading malformed YAML raises WorkflowError."""
        malformed_yaml = tmp_workflow_dir / "malformed.yaml"
        malformed_yaml.write_text("invalid: yaml: content: [")  # Invalid YAML syntax

        inspector = WorkflowInspector(str(malformed_yaml))

        with pytest.raises(WorkflowInspector.WorkflowError, match="Failed to parse workflow"):
            inspector.load()

    def test_load_malformed_json(self, tmp_workflow_dir):
        """Test that loading malformed JSON raises WorkflowError."""
        malformed_json = tmp_workflow_dir / "malformed.json"
        malformed_json.write_text('{invalid json: [}')  # Invalid JSON and YAML syntax

        inspector = WorkflowInspector(str(malformed_json))

        with pytest.raises(WorkflowInspector.WorkflowError, match="Failed to parse workflow"):
            inspector.load()

    def test_load_empty_workflow(self, tmp_workflow_dir):
        """Test that empty workflow files raise WorkflowError."""
        empty_file = tmp_workflow_dir / "empty.json"
        empty_file.write_text("{}")

        inspector = WorkflowInspector(str(empty_file))

        with pytest.raises(WorkflowInspector.WorkflowError, match="Workflow file is empty"):
            inspector.load()

    def test_load_invalid_root_type(self, tmp_workflow_dir):
        """Test that non-dict workflow roots raise WorkflowError."""
        invalid_file = tmp_workflow_dir / "invalid.json"
        invalid_file.write_text('["task1", "task2"]')  # Array instead of object

        inspector = WorkflowInspector(str(invalid_file))

        with pytest.raises(WorkflowInspector.WorkflowError, match="Workflow root must be a dictionary"):
            inspector.load()


class TestWorkflowInspectorParseDAG:
    """Tests for WorkflowInspector.parse_dag()"""

    def test_parse_linear_dag(self, linear_dag_json):
        """Test parsing linear DAG: A -> B -> C."""
        inspector = WorkflowInspector(str(linear_dag_json))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag == {
            "validate": [],
            "build": ["validate"],
            "deploy": ["build"],
        }

    def test_parse_parallel_dag(self, parallel_dag_json):
        """Test parsing parallel DAG: A -> C and B -> C."""
        inspector = WorkflowInspector(str(parallel_dag_json))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag == {
            "validate_code": [],
            "validate_config": [],
            "merge": ["validate_code", "validate_config"],
        }

    def test_parse_complex_dag(self, complex_dag_json):
        """Test parsing complex DAG with multiple branches and convergence."""
        inspector = WorkflowInspector(str(complex_dag_json))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag["init"] == []
        assert dag["lint"] == ["init"]
        assert dag["test_unit"] == ["init"]
        assert dag["test_integration"] == ["init"]
        assert dag["analyze"] == ["lint", "test_unit"]
        assert dag["build"] == ["analyze"]
        assert set(dag["verify"]) == {"build", "test_integration"}

    def test_detect_cyclic_dependency(self, cyclic_dag_json):
        """Test that cyclic dependencies are detected and raise CyclicDependencyError."""
        inspector = WorkflowInspector(str(cyclic_dag_json))
        inspector.load()

        with pytest.raises(
            WorkflowInspector.CyclicDependencyError, match="Circular dependency detected"
        ):
            inspector.parse_dag()

    def test_parse_dag_before_load(self, tmp_workflow_dir):
        """Test that parse_dag raises error if called before load."""
        workflow = {"name": "test", "tasks": []}
        path = tmp_workflow_dir / "test.json"
        path.write_text(json.dumps(workflow))

        inspector = WorkflowInspector(str(path))

        with pytest.raises(WorkflowInspector.WorkflowError, match="Workflow not loaded"):
            inspector.parse_dag()

    def test_string_depends_on_normalization(self, tmp_workflow_dir):
        """Test that string depends_on values are normalized to lists."""
        workflow = {
            "name": "test",
            "tasks": [
                {"label": "a", "type": "script", "depends_on": "b"},  # String, not list
                {"label": "b", "type": "script", "depends_on": []},
            ],
        }
        path = tmp_workflow_dir / "test.json"
        path.write_text(json.dumps(workflow))

        inspector = WorkflowInspector(str(path))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag["a"] == ["b"]
        assert dag["b"] == []

    def test_empty_string_dependencies_filtered(self, tmp_workflow_dir):
        """Test that empty string dependencies are filtered out."""
        workflow = {
            "name": "test",
            "tasks": [
                {"label": "a", "type": "script", "depends_on": ["b", "", "c"]},
                {"label": "b", "type": "script"},
                {"label": "c", "type": "script"},
            ],
        }
        path = tmp_workflow_dir / "test.json"
        path.write_text(json.dumps(workflow))

        inspector = WorkflowInspector(str(path))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag["a"] == ["b", "c"]
        assert "" not in dag["a"]

    def test_self_referential_dependency(self, tmp_workflow_dir):
        """Test detection of task depending on itself."""
        workflow = {
            "name": "self_ref",
            "tasks": [
                {"label": "a", "type": "script", "depends_on": "a"},
            ],
        }
        path = tmp_workflow_dir / "self_ref.json"
        path.write_text(json.dumps(workflow))

        inspector = WorkflowInspector(str(path))
        inspector.load()

        with pytest.raises(WorkflowInspector.CyclicDependencyError):
            inspector.parse_dag()

    def test_workflow_with_no_dependencies(self, tmp_workflow_dir):
        """Test workflow where all tasks are independent."""
        workflow = {
            "name": "independent",
            "tasks": [
                {"label": "a", "type": "script"},
                {"label": "b", "type": "script"},
                {"label": "c", "type": "script"},
            ],
        }
        path = tmp_workflow_dir / "independent.json"
        path.write_text(json.dumps(workflow))

        inspector = WorkflowInspector(str(path))
        inspector.load()
        dag = inspector.parse_dag()

        assert dag["a"] == []
        assert dag["b"] == []
        assert dag["c"] == []


class TestWorkflowInspectorGetSummary:
    """Tests for WorkflowInspector.get_summary()"""

    def test_get_summary_structure(self, linear_dag_json):
        """Test that get_summary returns the correct structure."""
        inspector = WorkflowInspector(str(linear_dag_json))
        inspector.load()
        summary = inspector.get_summary()

        # Check top-level keys
        assert "tasks" in summary
        assert "dag" in summary
        assert "task_count" in summary
        assert "output_refs" in summary

        # Check types
        assert isinstance(summary["tasks"], list)
        assert isinstance(summary["dag"], dict)
        assert isinstance(summary["task_count"], int)
        assert isinstance(summary["output_refs"], list)

        # Check task count
        assert summary["task_count"] == 3

        # Check task structure
        for task in summary["tasks"]:
            assert "label" in task
            assert "type" in task
            assert "status" in task

    def test_get_summary_before_load(self, tmp_workflow_dir):
        """Test that get_summary raises error if called before load."""
        workflow = {"name": "test", "tasks": []}
        path = tmp_workflow_dir / "test.json"
        path.write_text(json.dumps(workflow))

        inspector = WorkflowInspector(str(path))

        with pytest.raises(WorkflowInspector.WorkflowError, match="Workflow not loaded"):
            inspector.get_summary()

    def test_summary_is_json_serializable(self, linear_dag_json):
        """Test that summary output is JSON serializable."""
        inspector = WorkflowInspector(str(linear_dag_json))
        inspector.load()
        summary = inspector.get_summary()

        # Should not raise
        json_str = json.dumps(summary)
        assert json_str
        assert "tasks" in json_str
        assert "dag" in json_str

    def test_task_metadata_in_summary(self, workflow_with_statuses):
        """Test that task metadata (type, status) is preserved in summary."""
        inspector = WorkflowInspector(str(workflow_with_statuses))
        inspector.load()
        summary = inspector.get_summary()

        tasks_by_label = {t["label"]: t for t in summary["tasks"]}

        assert tasks_by_label["completed_task"]["status"] == "succeeded"
        assert tasks_by_label["running_task"]["status"] == "running"
        assert tasks_by_label["failed_task"]["status"] == "failed"
        assert tasks_by_label["pending_task"]["status"] == "pending"


class TestWorkflowInspectorGetTaskStatus:
    """Tests for WorkflowInspector.get_task_status()"""

    def test_get_task_status(self, workflow_with_statuses):
        """Test that get_task_status returns task metadata correctly."""
        inspector = WorkflowInspector(str(workflow_with_statuses))
        inspector.load()

        assert inspector.get_task_status("completed_task") == "succeeded"
        assert inspector.get_task_status("running_task") == "running"
        assert inspector.get_task_status("failed_task") == "failed"
        assert inspector.get_task_status("pending_task") == "pending"
        assert inspector.get_task_status("nonexistent_task") == "unknown"


class TestWorkflowInspectorOutputReferences:
    """Tests for output reference detection"""

    def test_find_output_references(self, workflow_with_output_refs):
        """Test that output references are correctly detected in workflows."""
        inspector = WorkflowInspector(str(workflow_with_output_refs))
        inspector.load()
        summary = inspector.get_summary()

        output_refs = summary["output_refs"]
        assert len(output_refs) >= 3
        assert "$task.extract.outputs.data" in output_refs
        assert "${task.process.outputs.result}" in output_refs
        assert "$task.extract.outputs.time" in output_refs


class TestWorkflowInspectorAlternateFormats:
    """Tests for alternate workflow formats"""

    def test_parse_workflow_with_steps(self, workflow_with_steps):
        """Test that workflows using 'steps' instead of 'tasks' are handled."""
        inspector = WorkflowInspector(str(workflow_with_steps))
        inspector.load()
        dag = inspector.parse_dag()

        assert "step1" in dag
        assert "step2" in dag
        assert dag["step1"] == []
        assert dag["step2"] == ["step1"]


class TestWorkflowInspectorStateManagement:
    """Tests for WorkflowInspector state management"""

    def test_multiple_load_calls(self, linear_dag_json, parallel_dag_json):
        """Test that multiple load calls properly refresh inspector state."""
        inspector = WorkflowInspector(str(linear_dag_json))
        inspector.load()
        dag1 = inspector.parse_dag()
        assert dag1["validate"] == []

        # Load different workflow
        inspector.workflow_path = Path(str(parallel_dag_json))
        inspector.load()
        dag2 = inspector.parse_dag()
        assert dag2["validate_code"] == []

        # Verify state was refreshed
        assert "build" in dag1 and "build" not in dag2
