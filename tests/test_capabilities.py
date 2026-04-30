"""Tests for capability discovery and inspection."""

from __future__ import annotations

import json

import pytest

from mr1 import workflow_cli
from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.kazi_runner import MockRunner
from mr1.mr1 import MR1, StateManager
from mr1.scheduler import validate_spec
from mr1.tools import default_tool_registry
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


def _build_mr1(tmp_path):
    mr1 = MR1(
        workflow_store=WorkflowStore(root=tmp_path / "workflows"),
        workflow_runner=MockRunner(),
        workflow_auto_tick=False,
        workflow_compiler=lambda *_: "{}",
    )
    mr1._state = StateManager(state_path=tmp_path / "mr1_state.json")
    return mr1


class TestCapabilityRegistry:
    def test_lists_global_capabilities(self):
        registry = default_capability_registry()

        names = registry.list_capabilities()

        assert "shell_command" in names
        assert "manual_event" in names
        assert "kazi" in names
        assert "workflow_compiler" in names
        assert names == sorted(names)

    def test_duplicate_capability_name_rejected(self):
        with pytest.raises(ValueError, match="duplicate capability name 'shell_command'"):
            CapabilityRegistry(agent_capabilities=[
                {
                    "name": "shell_command",
                    "type": "agent",
                    "description": "duplicate",
                    "inputs": {},
                    "outputs": {},
                    "examples": [],
                    "config_schema": {},
                }
            ])

    def test_shell_command_capability_shape_and_output_paths(self):
        description = default_capability_registry().describe_capability("shell_command")

        assert description["name"] == "shell_command"
        assert description["type"] == "tool"
        assert description["config_schema"]["argv"] == {
            "type": "list[string]",
            "required": True,
        }
        assert description["outputs"]["result.data.stdout"] == "captured stdout text"
        assert description["outputs"]["artifact.stdout"] == "artifact path for stdout when present"
        assert description["config_schema"]["cwd"]["required"] is True
        assert description["direct_allowed"] is False
        assert description["workflow_allowed"] is True
        assert description["path_arg_fields"] == ["cwd"]

    def test_output_keys_are_explicit_supported_reference_roots(self):
        registry = default_capability_registry()

        for description in registry.describe_all():
            for key in description["outputs"]:
                assert (
                    key == "result.text"
                    or key.startswith("result.data.")
                    or key.startswith("result.metrics.")
                    or key.startswith("artifact.")
                )

    def test_examples_are_minimal_and_valid_task_specs(self):
        registry = default_capability_registry()

        for description in registry.describe_all():
            examples = description["examples"]
            assert len(examples) == 1
            for example in examples:
                validate_spec({
                    "title": f"example {description['name']}",
                    "tasks": [example],
                })


class TestDirectCallMetadata:
    def test_all_capabilities_have_policy_fields(self):
        registry = default_capability_registry()

        for description in registry.describe_all():
            name = description["name"]
            assert "risk_score" in description, name
            assert "direct_allowed" in description, name
            assert "workflow_allowed" in description, name
            assert "requires_scope" in description, name
            assert "is_filesystem" in description, name
            assert "path_arg_fields" in description, name
            assert isinstance(description["direct_allowed"], bool), name
            assert isinstance(description["workflow_allowed"], bool), name
            assert isinstance(description["path_arg_fields"], list), name

    def test_policy_rollout_matches_spec(self):
        registry = default_capability_registry()
        enabled = {
            "read_file": (True, True, True, ["path"]),
            "file_exists": (True, True, True, ["path"]),
            "time_reached": (True, True, False, []),
            "condition_script": (True, True, True, ["path"]),
        }

        for name, (direct_allowed, workflow_allowed, requires_scope, path_fields) in enabled.items():
            desc = registry.describe_capability(name)
            assert desc["direct_allowed"] is direct_allowed, name
            assert desc["workflow_allowed"] is workflow_allowed, name
            assert desc["requires_scope"] is requires_scope, name
            assert desc["path_arg_fields"] == path_fields, name

    def test_workflow_only_capabilities_not_direct_callable(self):
        registry = default_capability_registry()

        for name in ("write_file", "shell_command", "manual_event"):
            desc = registry.describe_capability(name)
            assert desc["direct_allowed"] is False, name
            assert desc["workflow_allowed"] is True, name

    def test_agents_not_directly_callable(self):
        registry = default_capability_registry()

        for desc in registry.describe_all():
            if desc["type"] == "agent":
                assert desc["direct_allowed"] is False, desc["name"]
                assert desc["workflow_allowed"] is False, desc["name"]


class TestCapabilityCli:
    def test_capability_json_output(self, store, capsys):
        rc = workflow_cli.main(["capability", "shell_command", "--json"], store=store)

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == default_capability_registry().describe_capability("shell_command")

    def test_tool_example_output(self, store, capsys):
        rc = workflow_cli.main(["tool", "shell_command", "--example"], store=store)

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == default_tool_registry().describe_tool("shell_command")["examples"][0]

    def test_capabilities_summary_lists_agents_tools_and_watchers(self, store, capsys):
        rc = workflow_cli.main(["capabilities"], store=store)

        assert rc == 0
        out = capsys.readouterr().out
        assert "shell_command" in out
        assert "manual_event" in out
        assert "kazi" in out
        assert "workflow_compiler" in out

    def test_cli_unknown_tool_error_is_deterministic(self, store, capsys):
        rc = workflow_cli.main(["tool", "missing_tool"], store=store)

        assert rc == 2
        assert capsys.readouterr().err.strip() == "error: tool not found: missing_tool"

    def test_cli_invalid_flag_combination_is_deterministic(self, store, capsys):
        rc = workflow_cli.main(
            ["tool", "shell_command", "--example", "--brief"],
            store=store,
        )

        assert rc == 2
        assert capsys.readouterr().err.strip() == "error: invalid flag combination"

    def test_capability_call_cli_happy_path(self, tmp_path, store, capsys):
        f = tmp_path / "note.txt"
        f.write_text("hello", encoding="utf-8")
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps({"path": str(f)}), encoding="utf-8")
        scoped = __import__("mr1.scoped_agents", fromlist=["PersistentAgentStore"]).PersistentAgentStore(
            root=tmp_path / "agents"
        )

        rc = workflow_cli.main(
            ["capability-call", "read_file", str(config_path)],
            store=store,
            scoped_agent_store=scoped,
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "capability:   read_file" in out
        assert "status:       succeeded" in out
        assert "duration_ms:" in out

    def test_capability_call_cli_disallowed_raises_error(self, tmp_path, store, capsys):
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps({"path": "/tmp/x", "content": "x"}), encoding="utf-8")
        scoped = __import__("mr1.scoped_agents", fromlist=["PersistentAgentStore"]).PersistentAgentStore(
            root=tmp_path / "agents"
        )

        rc = workflow_cli.main(
            ["capability-call", "write_file", str(config_path)],
            store=store,
            scoped_agent_store=scoped,
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "status:       denied" in out

    def test_capability_call_cli_memory_search_smoke(self, tmp_path, store, capsys):
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps({"query": "file access", "limit": 5}), encoding="utf-8")
        scoped = __import__("mr1.scoped_agents", fromlist=["PersistentAgentStore"]).PersistentAgentStore(
            root=tmp_path / "agents"
        )

        rc = workflow_cli.main(
            ["capability-call", "memory_search", str(config_path)],
            store=store,
            scoped_agent_store=scoped,
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "capability:   memory_search" in out
        assert "status:       succeeded" in out

    def test_capability_text_output_shows_direct_call_metadata(self, store, capsys):
        rc = workflow_cli.main(["capability", "read_file"], store=store)

        assert rc == 0
        out = capsys.readouterr().out
        assert "risk_score:" in out
        assert "direct_allowed:  True" in out
        assert "workflow_allowed:True" in out
        assert "path_arg_fields: path" in out


class TestCapabilityBuiltins:
    def test_mr1_capabilities_builtin_lists_global_capabilities(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        output = mr1._handle_builtin("/capabilities")

        assert "shell_command" in output
        assert "manual_event" in output
        assert "kazi" in output
        assert "workflow_compiler" in output

    def test_mr1_capability_json_builtin(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        output = mr1._handle_builtin("/capability shell_command --json")
        payload = json.loads(output)

        assert payload["config_schema"]["argv"]["required"] is True
        assert payload["outputs"]["result.data.stdout"] == "captured stdout text"

    def test_mr1_capability_call_builtin_happy_path(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("hi", encoding="utf-8")
        config_json = json.dumps({"path": str(f)})
        mr1 = _build_mr1(tmp_path)

        output = mr1._handle_builtin(f"/capability call read_file '{config_json}'")

        assert "capability:   read_file" in output
        assert "status:       succeeded" in output

    def test_mr1_capability_call_builtin_disallowed(self, tmp_path):
        mr1 = _build_mr1(tmp_path)
        config_json = json.dumps({"path": "/x", "content": "x"})

        output = mr1._handle_builtin(f"/capability call write_file '{config_json}'")

        assert "status:       denied" in output

    def test_mr1_capability_call_builtin_usage_on_wrong_args(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        assert "usage:" in mr1._handle_builtin("/capability call")

    def test_mr1_builtin_errors_are_deterministic(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        assert mr1._handle_builtin("/capability missing_capability") == (
            "capability not found: missing_capability"
        )
        assert mr1._handle_builtin("/tool missing_tool") == "tool not found: missing_tool"
        assert mr1._handle_builtin("/capability shell_command --example --brief") == (
            "invalid flag combination"
        )

    def test_mr1_schema_builtin_outputs_schema_metadata(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        output = mr1._handle_builtin("/schema")

        assert '"workflow"' in output
        assert '"inputs"' in output
        assert '"task-kinds"' in output

    def test_mr1_schema_inputs_json_builtin(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        output = mr1._handle_builtin("/schema inputs --json")
        payload = json.loads(output)

        assert payload["item_shape"] == {
            "name": "string",
            "from": "<label>.<reference>",
        }
        assert "inputs must NEVER be strings" in payload["rules"]

    def test_mr1_schema_builtin_errors_are_deterministic(self, tmp_path):
        mr1 = _build_mr1(tmp_path)

        assert mr1._handle_builtin("/schema invalid_section") == (
            "error: schema section not found: invalid_section"
        )
        assert mr1._handle_builtin("/schema inputs --example") == (
            "usage: /schema [workflow|task|inputs|refs|task-kinds] [--json] [--brief]"
        )
