"""
Agent registry and runtime helpers for workflow agent tasks.

Agents are runtime-managed workers. Unlike deterministic tools and
watchers, they require runtime profile validation, health checks, and
controlled invocation through a CLI process.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from mr1.core import Dispatcher, PermissionDenied


_PKG_ROOT = Path(__file__).resolve().parent
_AGENTS_DIR = _PKG_ROOT / "agents"
_DEFAULT_AGENT_TIMEOUT_S = 300
_HEALTH_PROMPT = "ping"
_HEALTH_TIMEOUT_S = 30
_AUTH_PATTERNS = (
    "not logged in",
    "login required",
    "please run claude login",
    "authentication failed",
    "auth required",
    "unauthenticated",
)


class AgentConfigError(ValueError):
    """Raised when an agent or its runtime config is invalid."""


class AgentRuntimeError(ValueError):
    """Raised when agent runtime output cannot be interpreted safely."""


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    config_schema: dict[str, Any]
    runtime: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, str]
    examples: list[dict[str, Any]]
    config_path: Path


def is_auth_error_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in _AUTH_PATTERNS)


def _validate_scalar(config_key: str, value: Any, expected_type: str) -> None:
    if expected_type == "string":
        if not isinstance(value, str):
            raise AgentConfigError(f"{config_key} must be a string")
        return
    if expected_type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise AgentConfigError(f"{config_key} must be an integer")
        return
    if expected_type == "list[string]":
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise AgentConfigError(f"{config_key} must be a list of strings")
        return
    raise AgentConfigError(f"unsupported config schema type: {expected_type}")


def validate_agent_runtime_config(
    agent_name: str,
    config: Optional[dict[str, Any]],
    registry: Optional["AgentRegistry"] = None,
) -> dict[str, Any]:
    active_registry = registry or default_agent_registry()
    definition = active_registry.get_definition(agent_name)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise AgentConfigError("agent runtime config must be a YAML mapping")

    extras = sorted(set(config) - set(definition.config_schema))
    if extras:
        raise AgentConfigError(f"unknown runtime config keys for {agent_name}: {', '.join(extras)}")

    normalized: dict[str, Any] = {}
    for key, schema in definition.config_schema.items():
        required = bool(schema.get("required"))
        has_value = key in config and config[key] is not None
        if not has_value:
            if required:
                raise AgentConfigError(f"{key} is required")
            if "default" in schema:
                normalized[key] = deepcopy(schema["default"])
            continue
        _validate_scalar(key, config[key], str(schema.get("type", "")))
        if key == "timeout_s" and int(config[key]) < 1:
            raise AgentConfigError("timeout_s must be an integer >= 1")
        normalized[key] = deepcopy(config[key])

    if "timeout_s" not in normalized:
        normalized["timeout_s"] = _DEFAULT_AGENT_TIMEOUT_S
    if "allowed_tools" not in normalized:
        normalized["allowed_tools"] = []
    return normalized


def load_agent_runtime_config(
    agent_name: str,
    registry: Optional["AgentRegistry"] = None,
) -> dict[str, Any]:
    active_registry = registry or default_agent_registry()
    definition = active_registry.get_definition(agent_name)
    try:
        with open(definition.config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise AgentConfigError(f"failed to load runtime config for {agent_name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AgentConfigError(f"runtime config for {agent_name} must be a YAML mapping")
    relevant = {
        key: raw[key]
        for key in definition.config_schema
        if key in raw
    }
    return validate_agent_runtime_config(agent_name, relevant, registry=active_registry)


def build_agent_command(
    agent_name: str,
    prompt: str,
    *,
    config: Optional[dict[str, Any]] = None,
    registry: Optional["AgentRegistry"] = None,
    binary_override: Optional[str] = None,
) -> list[str]:
    active_registry = registry or default_agent_registry()
    definition = active_registry.get_definition(agent_name)
    normalized = (
        load_agent_runtime_config(agent_name, registry=active_registry)
        if config is None else
        validate_agent_runtime_config(agent_name, config, registry=active_registry)
    )

    binary = binary_override or str(definition.runtime["binary"])
    cmd = [binary, "-p", prompt]
    model = normalized.get("model")
    allowed_tools = normalized.get("allowed_tools") or []
    if model:
        cmd.extend(["--model", model])
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])
    cmd.extend(["--output-format", "json"])
    return cmd


def parse_agent_json_envelope(raw: str) -> dict[str, Any]:
    payload = raw.strip()
    if not payload:
        raise AgentRuntimeError("empty JSON output")
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AgentRuntimeError(f"invalid JSON output: {exc}") from exc
    if not isinstance(envelope, dict):
        raise AgentRuntimeError("JSON output must be an object")

    result = envelope.get("result", "")
    if isinstance(result, str):
        text = result
    elif result is None:
        text = ""
    else:
        text = json.dumps(result, sort_keys=True)

    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {
            key: value
            for key, value in envelope.items()
            if key not in {"result", "is_error", "usage"}
        }

    return {
        "text": text,
        "raw": envelope,
        "is_error": bool(envelope.get("is_error")),
        "usage": usage,
        "metadata": metadata,
    }


def run_agent_health(
    agent_name: str,
    registry: Optional["AgentRegistry"] = None,
) -> dict[str, Any]:
    active_registry = registry or default_agent_registry()
    definition = active_registry.get_definition(agent_name)
    checks = {
        "binary": "missing",
        "version": "not_run",
        "prompt_test": "failed",
        "auth": "failed",
        "json_parse": "failed",
        "config": "failed",
        "flags": "failed",
    }
    healthy = True
    error: Optional[str] = None

    resolved_binary = shutil.which(str(definition.runtime["binary"]))
    if resolved_binary is None:
        healthy = False
        error = f"binary not found: {definition.runtime['binary']}"
    else:
        checks["binary"] = resolved_binary

    config: Optional[dict[str, Any]] = None
    if error is None:
        try:
            config = load_agent_runtime_config(agent_name, registry=active_registry)
            checks["config"] = "ok"
        except AgentConfigError as exc:
            healthy = False
            error = str(exc)
            checks["config"] = "failed"

    if error is None and resolved_binary is not None:
        try:
            version_proc = subprocess.run(
                [resolved_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version_text = (version_proc.stdout or version_proc.stderr or "").strip()
            if version_proc.returncode != 0:
                healthy = False
                error = version_text or f"{definition.runtime['binary']} --version failed"
                checks["version"] = "failed"
            else:
                checks["version"] = version_text or "ok"
        except (OSError, subprocess.TimeoutExpired) as exc:
            healthy = False
            error = f"failed to run version check: {exc}"
            checks["version"] = "failed"

    if error is None and config is not None:
        dispatcher = Dispatcher()
        cmd = build_agent_command(
            agent_name,
            _HEALTH_PROMPT,
            config=config,
            registry=active_registry,
            binary_override=resolved_binary,
        )
        cli_flags = [token for token in cmd[1:] if token.startswith("-")]
        try:
            dispatcher.validate_full_spawn(
                agent_name,
                cli_flags,
                list(config.get("allowed_tools", [])),
            )
            checks["flags"] = "ok"
        except PermissionDenied as exc:
            healthy = False
            error = str(exc)
            checks["flags"] = "failed"

    if error is None and config is not None and resolved_binary is not None:
        cmd = build_agent_command(
            agent_name,
            _HEALTH_PROMPT,
            config=config,
            registry=active_registry,
            binary_override=resolved_binary,
        )
        try:
            probe = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_HEALTH_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            healthy = False
            error = f"prompt probe timed out after {_HEALTH_TIMEOUT_S}s"
            checks["prompt_test"] = "failed"
            checks["auth"] = "failed" if is_auth_error_text(str(exc)) else "ok"
        except OSError as exc:
            healthy = False
            error = f"failed to run prompt probe: {exc}"
            checks["prompt_test"] = "failed"
            checks["auth"] = "failed" if is_auth_error_text(str(exc)) else "ok"
        else:
            combined_text = "\n".join(
                part for part in ((probe.stdout or "").strip(), (probe.stderr or "").strip()) if part
            )
            if probe.returncode != 0:
                healthy = False
                error = combined_text or f"prompt probe failed with exit {probe.returncode}"
                checks["prompt_test"] = "failed"
                checks["auth"] = "failed" if is_auth_error_text(combined_text) else "ok"
            else:
                checks["prompt_test"] = "passed"
                try:
                    parsed = parse_agent_json_envelope(probe.stdout or "")
                    checks["json_parse"] = "ok"
                    if parsed["is_error"]:
                        healthy = False
                        detail = parsed["text"] or "agent returned is_error=true"
                        error = detail
                        checks["auth"] = "failed" if is_auth_error_text(detail) else "ok"
                    else:
                        checks["auth"] = "ok"
                except AgentRuntimeError as exc:
                    healthy = False
                    error = str(exc)
                    checks["json_parse"] = "failed"
                    checks["auth"] = "failed" if is_auth_error_text(combined_text) else "ok"

    result = {
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks,
    }
    if error:
        result["error"] = error
    return result


class AgentRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, AgentDefinition] = {}

    def register(
        self,
        name: str,
        *,
        description: str,
        config_schema: dict[str, Any],
        runtime: dict[str, Any],
        inputs: dict[str, Any],
        outputs: dict[str, str],
        examples: list[dict[str, Any]],
        config_path: Path,
    ) -> None:
        if name in self._definitions:
            raise ValueError(f"duplicate agent name '{name}'")
        self._definitions[name] = AgentDefinition(
            name=name,
            description=description,
            config_schema=deepcopy(config_schema),
            runtime=deepcopy(runtime),
            inputs=deepcopy(inputs),
            outputs=deepcopy(outputs),
            examples=deepcopy(examples),
            config_path=config_path,
        )

    def list_agents(self) -> list[str]:
        return sorted(self._definitions)

    def is_registered(self, name: Optional[str]) -> bool:
        return bool(name) and name in self._definitions

    def get_definition(self, name: str) -> AgentDefinition:
        definition = self._definitions.get(name)
        if definition is None:
            raise ValueError(f"agent not found: {name}")
        return definition

    def describe_agent(self, name: str) -> dict[str, Any]:
        definition = self.get_definition(name)
        return {
            "name": definition.name,
            "type": "agent",
            "description": definition.description,
            "inputs": deepcopy(definition.inputs),
            "outputs": deepcopy(definition.outputs),
            "examples": deepcopy(definition.examples),
            "config_schema": deepcopy(definition.config_schema),
            "runtime": deepcopy(definition.runtime),
        }

    def describe_all(self) -> list[dict[str, Any]]:
        return [self.describe_agent(name) for name in self.list_agents()]


_DEFAULT_REGISTRY: Optional[AgentRegistry] = None


def default_agent_registry() -> AgentRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = AgentRegistry()
        registry.register(
            "kazi",
            description="Claude Code CLI-based reasoning agent for scoped workflow execution.",
            config_schema={
                "model": {"type": "string", "required": False, "optional": True},
                "allowed_tools": {
                    "type": "list[string]",
                    "required": False,
                    "optional": True,
                    "default": [],
                },
                "timeout_s": {
                    "type": "int",
                    "required": False,
                    "optional": True,
                    "default": _DEFAULT_AGENT_TIMEOUT_S,
                },
            },
            runtime={
                "binary": "claude",
                "invocation": "claude -p <prompt> --output-format json",
                "supports_json_output": True,
            },
            inputs={
                "prompt": "string",
                "inputs": "dataflow inputs",
            },
            outputs={
                "result.text": "final generated text",
                "result.data.raw": "parsed Claude JSON envelope",
                "result.data.is_error": "true when the Claude envelope reports an error",
                "result.data.metadata": "Claude metadata block or extra envelope fields",
                "result.metrics.usage": "Claude usage block when present",
            },
            examples=[
                {
                    "label": "summarize",
                    "title": "Summarize",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "Summarize the provided inputs.",
                }
            ],
            config_path=_AGENTS_DIR / "kazi.yml",
        )
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
