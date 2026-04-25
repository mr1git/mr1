"""Tests for deterministic workflow schema introspection."""

from __future__ import annotations

from mr1.workflow_schema import default_workflow_schema_registry


class TestWorkflowSchemaRegistry:
    def test_describe_all_contains_expected_sections(self):
        payload = default_workflow_schema_registry().describe_all()

        assert set(payload) == {"workflow", "task", "inputs", "refs", "task-kinds"}

    def test_inputs_schema_explicitly_requires_object_name_and_from(self):
        inputs = default_workflow_schema_registry().describe_inputs()

        assert inputs["type"] == "list[InputSpec]"
        assert inputs["item_type"] == "object"
        assert inputs["item_shape"] == {
            "name": "string",
            "from": "<label>.<reference>",
        }
        assert "inputs must NEVER be strings" in inputs["rules"]

    def test_inputs_schema_includes_valid_and_invalid_examples(self):
        inputs = default_workflow_schema_registry().describe_inputs()

        assert inputs["valid_examples"] == [[
            {"name": "notes", "from": "read_notes.result.text"},
        ]]
        assert inputs["invalid_examples"] == [[
            "read_notes.result.text",
        ]]

    def test_reference_schema_includes_supported_paths(self):
        refs = default_workflow_schema_registry().describe_references()

        assert "<label>.result.text" in refs["supported_patterns"]
        assert "<label>.result.data.<key>[.<nested_key>...]" in refs["supported_patterns"]
        assert "<label>.result.metrics.<key>[.<nested_key>...]" in refs["supported_patterns"]
        assert "<label>.artifact.<artifact_name>" in refs["supported_patterns"]

    def test_task_kind_schema_includes_required_fields(self):
        task_kinds = default_workflow_schema_registry().describe_task_kinds()

        assert task_kinds["agent"]["required_fields"] == {
            "agent_type": "kazi",
            "prompt": "string",
        }
        assert task_kinds["tool"]["required_fields"] == {
            "tool_type": "string",
            "tool_config": "object",
        }
        assert task_kinds["watcher"]["required_fields"] == {
            "watcher_type": "string",
            "watch_config": "object",
        }
