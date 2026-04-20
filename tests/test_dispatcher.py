"""Tests for mr1.core.dispatcher"""

import pytest
from mr1.core.dispatcher import Dispatcher, PermissionDenied


@pytest.fixture
def dispatcher():
    return Dispatcher()


class TestValidateAgent:
    def test_known_agents_pass(self, dispatcher):
        for agent in ("mr1", "mr2", "mr3", "kazi", "mem_dltr", "mem_rtvr", "ctx_pkgr", "com_smrzr"):
            assert dispatcher.validate_agent(agent) is True

    def test_unknown_agent_raises(self, dispatcher):
        with pytest.raises(PermissionDenied, match="unknown agent type"):
            dispatcher.validate_agent("rogue_agent")

    def test_mr4_resolves_to_mr3(self, dispatcher):
        """MR4+ agents resolve to MR3 permissions."""
        assert dispatcher.validate_agent("mr4") is True
        assert dispatcher.validate_agent("mr5") is True
        assert dispatcher.validate_agent("mr10") is True

    def test_kami_no_longer_known(self, dispatcher):
        with pytest.raises(PermissionDenied, match="unknown agent type"):
            dispatcher.validate_agent("kami")


class TestValidateCliFlags:
    def test_allowed_flags_pass(self, dispatcher):
        assert dispatcher.validate_cli_flags("kazi", ["-p", "--model", "--output-format"]) is True

    def test_disallowed_flag_raises(self, dispatcher):
        with pytest.raises(PermissionDenied, match="cli flag not allowed"):
            dispatcher.validate_cli_flags("kazi", ["-p", "--evil-flag"])

    def test_value_tokens_not_checked(self, dispatcher):
        # Non-flag tokens (values) should be ignored.
        assert dispatcher.validate_cli_flags("kazi", ["-p", "some prompt text"]) is True

    def test_agent_specific_flags(self, dispatcher):
        # mr1 has --append-system-prompt, kazi does not.
        assert dispatcher.validate_cli_flags("mr1", ["--append-system-prompt"]) is True
        with pytest.raises(PermissionDenied):
            dispatcher.validate_cli_flags("kazi", ["--append-system-prompt"])

    def test_mr2_has_bare_and_skip_permissions(self, dispatcher):
        assert dispatcher.validate_cli_flags(
            "mr2", ["--bare", "--dangerously-skip-permissions"]
        ) is True

    def test_mr4_inherits_mr3_flags(self, dispatcher):
        assert dispatcher.validate_cli_flags(
            "mr4", ["--bare", "--dangerously-skip-permissions", "--output-format"]
        ) is True


class TestValidateShellCommand:
    def test_allowed_command_passes(self, dispatcher):
        assert dispatcher.validate_shell_command("kazi", "ls -la /tmp") is True

    def test_disallowed_command_raises(self, dispatcher):
        with pytest.raises(PermissionDenied, match="shell command not allowed"):
            dispatcher.validate_shell_command("kazi", "rm -rf /")

    def test_shell_operators_rejected(self, dispatcher):
        for op_cmd in ("ls; rm", "cat file | grep x", "ls && echo done", "echo $(whoami)"):
            with pytest.raises(PermissionDenied, match="shell operator"):
                dispatcher.validate_shell_command("kazi", op_cmd)

    def test_empty_command_rejected(self, dispatcher):
        with pytest.raises(PermissionDenied, match="empty command"):
            dispatcher.validate_shell_command("kazi", "   ")

    def test_git_allowed_for_mr2_not_kazi(self, dispatcher):
        assert dispatcher.validate_shell_command("mr2", "git status") is True
        with pytest.raises(PermissionDenied):
            dispatcher.validate_shell_command("kazi", "git status")


class TestValidateTools:
    def test_allowed_tools_pass(self, dispatcher):
        assert dispatcher.validate_tools("kazi", ["Read", "Glob"]) is True

    def test_disallowed_tool_raises(self, dispatcher):
        with pytest.raises(PermissionDenied, match="tool not allowed"):
            dispatcher.validate_tools("kazi", ["Read", "Agent"])

    def test_agent_tool_only_for_mr1(self, dispatcher):
        assert dispatcher.validate_tools("mr1", ["Agent"]) is True
        with pytest.raises(PermissionDenied):
            dispatcher.validate_tools("mr2", ["Agent"])


class TestValidateFullSpawn:
    def test_valid_spawn_passes(self, dispatcher):
        assert dispatcher.validate_full_spawn(
            "kazi",
            ["-p", "--model", "--output-format"],
            ["Read", "Glob"],
        ) is True

    def test_invalid_flag_rejects_spawn(self, dispatcher):
        with pytest.raises(PermissionDenied):
            dispatcher.validate_full_spawn("kazi", ["--evil"], ["Read"])

    def test_invalid_tool_rejects_spawn(self, dispatcher):
        with pytest.raises(PermissionDenied):
            dispatcher.validate_full_spawn("kazi", ["-p"], ["Agent"])


class TestValidateSpawnLevel:
    def test_kazi_always_allowed(self, dispatcher):
        assert dispatcher.validate_spawn_level(1, "kazi") is True
        assert dispatcher.validate_spawn_level(4, "kazi") is True

    def test_valid_mr_spawn(self, dispatcher):
        assert dispatcher.validate_spawn_level(1, "mr2") is True
        assert dispatcher.validate_spawn_level(2, "mr3") is True
        assert dispatcher.validate_spawn_level(3, "mr4") is True

    def test_exceeds_height_limit(self, dispatcher):
        # Default height_limit is 4.
        with pytest.raises(PermissionDenied, match="exceeds height limit"):
            dispatcher.validate_spawn_level(4, "mr5")

    def test_wrong_level_increment(self, dispatcher):
        with pytest.raises(PermissionDenied, match="can only spawn"):
            dispatcher.validate_spawn_level(1, "mr3")  # Must be mr2.
        with pytest.raises(PermissionDenied, match="can only spawn"):
            dispatcher.validate_spawn_level(2, "mr4")  # Must be mr3.

    def test_unknown_child_agent(self, dispatcher):
        with pytest.raises(PermissionDenied):
            dispatcher.validate_spawn_level(1, "unknown_agent")


class TestHeightLimit:
    def test_default_height_limit(self, dispatcher):
        assert dispatcher.height_limit == 4

    def test_custom_height_limit(self, tmp_path):
        # Create a custom config.
        config_file = tmp_path / "config.yml"
        config_file.write_text("height_limit: 3\n")
        d = Dispatcher(config_path=str(config_file))
        assert d.height_limit == 3

    def test_height_limit_enforced(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("height_limit: 2\n")
        d = Dispatcher(config_path=str(config_file))

        assert d.validate_spawn_level(1, "mr2") is True
        with pytest.raises(PermissionDenied, match="exceeds height limit"):
            d.validate_spawn_level(2, "mr3")


class TestAccessors:
    def test_get_allowed_tools(self, dispatcher):
        tools = dispatcher.get_allowed_tools("kazi")
        assert "Read" in tools
        assert "Agent" not in tools

    def test_get_allowed_shell_commands(self, dispatcher):
        cmds = dispatcher.get_allowed_shell_commands("mr2")
        assert "git" in cmds

    def test_get_allowed_cli_flags(self, dispatcher):
        flags = dispatcher.get_allowed_cli_flags("mr1")
        assert "--append-system-prompt" in flags

    def test_mr4_inherits_mr3_tools(self, dispatcher):
        mr3_tools = dispatcher.get_allowed_tools("mr3")
        mr4_tools = dispatcher.get_allowed_tools("mr4")
        assert mr3_tools == mr4_tools
