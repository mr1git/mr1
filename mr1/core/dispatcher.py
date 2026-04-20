"""
MR1 Dispatcher — Fully Deterministic Permission Layer
=====================================================
No LLM logic. No heuristics. No exceptions.

Reads allowlist.yml and validates whether an agent is permitted to run
a given CLI command with given flags. If anything is not explicitly
allowed, it is rejected.

Also enforces the height limit for the MRn agent hierarchy.
"""

import os
import shlex
from pathlib import Path
from typing import Optional

import yaml


# Resolve allowlist path relative to this file's location.
_PERMISSIONS_DIR = Path(__file__).resolve().parent.parent / "permissions"
_ALLOWLIST_PATH = _PERMISSIONS_DIR / "allowlist.yml"

# Config file lives at the project root.
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yml"

# Default height limit if config.yml is missing.
_DEFAULT_HEIGHT_LIMIT = 4


class PermissionDenied(Exception):
    """Raised when an agent attempts an action not in the allowlist."""

    def __init__(self, agent_type: str, detail: str):
        self.agent_type = agent_type
        self.detail = detail
        super().__init__(f"[DENIED] agent={agent_type} | {detail}")


class Dispatcher:
    """
    Deterministic permission gate.

    Usage:
        dispatcher = Dispatcher()
        dispatcher.validate_cli_flags("kazi", ["-p", "--model", "claude-3-5-haiku-20241022"])
        dispatcher.validate_shell_command("kazi", "ls -la /tmp")
        dispatcher.validate_tools("kazi", ["Read", "Glob"])
        dispatcher.validate_spawn_level(2, "mr3")
    """

    def __init__(
        self,
        allowlist_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        path = Path(allowlist_path) if allowlist_path else _ALLOWLIST_PATH
        if not path.exists():
            raise FileNotFoundError(f"Allowlist not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if not raw or "agents" not in raw:
            raise ValueError(f"Malformed allowlist: missing 'agents' key in {path}")

        self._agents = raw["agents"]
        self._height_limit = self._load_height_limit(config_path)

    def _load_height_limit(self, config_path: Optional[str] = None) -> int:
        """Load the height limit from config.yml."""
        path = Path(config_path) if config_path else _CONFIG_PATH
        if not path.exists():
            return _DEFAULT_HEIGHT_LIMIT
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            return int(config.get("height_limit", _DEFAULT_HEIGHT_LIMIT))
        except (yaml.YAMLError, TypeError, ValueError):
            return _DEFAULT_HEIGHT_LIMIT

    @property
    def height_limit(self) -> int:
        return self._height_limit

    # ------------------------------------------------------------------
    # Agent type resolution — MR4+ use MR3 permissions.
    # ------------------------------------------------------------------

    def _resolve_agent_type(self, agent_type: str) -> str:
        """
        Map an agent type to its allowlist entry.
        MR4, MR5, ... all use the MR3 permission profile.
        """
        if agent_type in self._agents:
            return agent_type
        # Check for mrN where N >= 4 — resolve to mr3.
        if agent_type.startswith("mr") and len(agent_type) > 2:
            try:
                level = int(agent_type[2:])
                if level >= 3 and "mr3" in self._agents:
                    return "mr3"
            except ValueError:
                pass
        raise PermissionDenied(agent_type, "unknown agent type")

    def _get_agent_config(self, agent_type: str) -> dict:
        """Get the allowlist config for an agent type, resolving if needed."""
        resolved = self._resolve_agent_type(agent_type)
        return self._agents[resolved]

    # ------------------------------------------------------------------
    # Public API — each method either returns True or raises PermissionDenied
    # ------------------------------------------------------------------

    def validate_agent(self, agent_type: str) -> bool:
        """Check that agent_type is a known, registered agent."""
        self._resolve_agent_type(agent_type)  # Raises if unknown.
        return True

    def validate_cli_flags(self, agent_type: str, flags: list[str]) -> bool:
        """
        Validate a list of CLI flags against the allowlist.
        Flags are the raw tokens (e.g. ["-p", "--model", "claude-3-5-haiku-20241022"]).
        Only flag tokens starting with '-' are checked; value tokens are skipped.
        """
        config = self._get_agent_config(agent_type)
        allowed = set(config.get("cli_flags", []))

        for token in flags:
            # Only validate flag tokens, not their values.
            if token.startswith("-"):
                if token not in allowed:
                    raise PermissionDenied(
                        agent_type, f"cli flag not allowed: {token}"
                    )
        return True

    def validate_shell_command(self, agent_type: str, command: str) -> bool:
        """
        Validate a shell command string.
        Extracts the base command (first token) and checks the allowlist.
        Rejects empty commands and commands with shell operators (;, |, &&, ||, `).
        """
        self.validate_agent(agent_type)

        # Reject empty commands.
        stripped = command.strip()
        if not stripped:
            raise PermissionDenied(agent_type, "empty command")

        # Reject shell chaining / piping — each command must be validated individually.
        dangerous_operators = [";", "|", "&&", "||", "`", "$("]
        for op in dangerous_operators:
            if op in stripped:
                raise PermissionDenied(
                    agent_type,
                    f"shell operator '{op}' not permitted — submit commands individually",
                )

        # Extract the base command (first token).
        try:
            tokens = shlex.split(stripped)
        except ValueError as e:
            raise PermissionDenied(agent_type, f"unparseable command: {e}")

        base_cmd = os.path.basename(tokens[0])
        config = self._get_agent_config(agent_type)
        allowed = set(config.get("shell_commands", []))

        if base_cmd not in allowed:
            raise PermissionDenied(
                agent_type, f"shell command not allowed: {base_cmd}"
            )

        return True

    def validate_tools(self, agent_type: str, tools: list[str]) -> bool:
        """Validate that all requested tools are in the agent's allowed_tools."""
        config = self._get_agent_config(agent_type)
        allowed = set(config.get("allowed_tools", []))

        for tool in tools:
            if tool not in allowed:
                raise PermissionDenied(
                    agent_type, f"tool not allowed: {tool}"
                )
        return True

    def validate_spawn_level(self, parent_level: int, child_agent: str) -> bool:
        """
        Validate that spawning the child agent from this parent level is allowed.

        Rules:
          - Kazis can be spawned from any level.
          - MR(n+1) can be spawned only if (n+1) <= height_limit.
          - MRn can only spawn MR(n+1), not MR(n+2) or MR(n-1).
        """
        if child_agent == "kazi":
            return True

        if child_agent.startswith("mr") and len(child_agent) > 2:
            try:
                child_level = int(child_agent[2:])
            except ValueError:
                raise PermissionDenied(child_agent, "invalid agent level format")

            if child_level > self._height_limit:
                raise PermissionDenied(
                    child_agent,
                    f"level {child_level} exceeds height limit {self._height_limit}",
                )
            if child_level != parent_level + 1:
                raise PermissionDenied(
                    child_agent,
                    f"mr{parent_level} can only spawn mr{parent_level + 1}, "
                    f"not {child_agent}",
                )
            return True

        raise PermissionDenied(child_agent, "unknown child agent type")

    def get_allowed_tools(self, agent_type: str) -> list[str]:
        """Return the list of tools an agent is permitted to use."""
        config = self._get_agent_config(agent_type)
        return list(config.get("allowed_tools", []))

    def get_allowed_shell_commands(self, agent_type: str) -> list[str]:
        """Return the list of shell commands an agent is permitted to run."""
        config = self._get_agent_config(agent_type)
        return list(config.get("shell_commands", []))

    def get_allowed_cli_flags(self, agent_type: str) -> list[str]:
        """Return the list of CLI flags an agent is permitted to use."""
        config = self._get_agent_config(agent_type)
        return list(config.get("cli_flags", []))

    def validate_full_spawn(
        self,
        agent_type: str,
        cli_flags: list[str],
        tools: list[str],
    ) -> bool:
        """
        Full pre-spawn validation. Checks agent, CLI flags, and tools in one call.
        Returns True or raises PermissionDenied.
        """
        self.validate_agent(agent_type)
        self.validate_cli_flags(agent_type, cli_flags)
        self.validate_tools(agent_type, tools)
        return True
