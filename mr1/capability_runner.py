"""
Direct synchronous capability invocation.

A narrow read-only / safe-exec surface alongside workflow tasks.
Workflows remain the durable engine; this path is for one-shot checks only.
Direct calls never wait, retry, stream, schedule, or create workflow artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.scoped_agents import PersistentAgentStore
from mr1.tools import _read_file_pure
from mr1.watchers import (
    _evaluate_condition_script_pure,
    _evaluate_file_exists_pure,
    _evaluate_time_reached_pure,
)


@dataclass
class CapabilityResult:
    status: str  # "succeeded" | "failed"
    output: dict[str, Any]
    error: Optional[str]
    duration_ms: int
    capability: str


class CapabilityRunner:
    def __init__(
        self,
        *,
        capability_registry: Optional[CapabilityRegistry] = None,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
    ):
        self._registry = capability_registry or default_capability_registry()
        self._agents = scoped_agent_store or PersistentAgentStore()

    def run_capability(
        self,
        name: str,
        config: dict[str, Any],
        caller_agent_id: str,
        mode: str = "direct",
    ) -> CapabilityResult:
        # --- Preflight checks raise deterministic errors ---
        try:
            meta = self._registry.describe_capability(name)
        except ValueError:
            raise ValueError(f"capability not found: {name}")

        if not meta.get("direct_callable", False):
            raise ValueError(f"capability not callable in direct mode: {name}")

        caller_type = self._resolve_caller_type(caller_agent_id)
        if caller_type not in meta.get("callable_by", []):
            raise ValueError("access denied")

        # Effective timeout: min(config timeout, metadata ceiling)
        meta_timeout = int(meta.get("timeout_s", 30))
        config_timeout = config.get("timeout_s")
        if isinstance(config_timeout, int) and config_timeout > 0:
            effective_timeout = min(config_timeout, meta_timeout)
        else:
            effective_timeout = meta_timeout

        # --- Runtime dispatch ---
        started = time.monotonic()
        try:
            output, error = self._dispatch(name, config, effective_timeout)
            status = "failed" if error is not None else "succeeded"
        except Exception as exc:
            output = {}
            error = str(exc)
            status = "failed"
        duration_ms = int((time.monotonic() - started) * 1000)

        result = CapabilityResult(
            status=status,
            output=output,
            error=error,
            duration_ms=duration_ms,
            capability=name,
        )
        self._write_audit_log(caller_agent_id, name, config, result)
        return result

    def _resolve_caller_type(self, caller_agent_id: str) -> str:
        return "mr1" if self._agents.is_root_agent(caller_agent_id) else "mrn"

    def _dispatch(
        self,
        name: str,
        config: dict[str, Any],
        timeout_s: int,
    ) -> tuple[dict[str, Any], Optional[str]]:
        if name == "read_file":
            path_str = config.get("path", "")
            max_bytes = config.get("max_bytes", 65536)
            r = _read_file_pure(path_str, max_bytes)
            output = {**r["data"], "text": r["text"]}
            return output, r["error"]

        if name == "file_exists":
            path_str = config.get("path", "")
            r = _evaluate_file_exists_pure(path_str)
            output = {
                "exists": r["metadata"].get("exists", False),
                "state": r["state"],
                "message": r["message"],
                **r["metadata"],
            }
            return output, None

        if name == "time_reached":
            at_str = config.get("at", "")
            now = datetime.now(timezone.utc)
            r = _evaluate_time_reached_pure(at_str, now)
            output = {
                "reached": r["state"] == "satisfied",
                "at": r["metadata"].get("at", at_str),
                "now": r["metadata"].get("now", now.isoformat()),
                "state": r["state"],
                "message": r["message"],
            }
            return output, None

        if name == "condition_script":
            path_str = config.get("path", "")
            r = _evaluate_condition_script_pure(path_str, timeout_s)
            state = r["state"]
            if state in ("timed_out", "failed"):
                return {**r["metadata"], "state": state, "message": r["message"]}, r["message"]
            output = {
                "satisfied": state == "satisfied",
                "state": state,
                "message": r["message"],
                **r["metadata"],
            }
            return output, None

        raise ValueError(f"capability not found: {name}")

    def _write_audit_log(
        self,
        agent_id: str,
        name: str,
        config: dict[str, Any],
        result: CapabilityResult,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "capability": name,
            "config": config,
            "result": result.output,
            "result_status": result.status,
            "duration_ms": result.duration_ms,
            "error": result.error,
        }
        try:
            self._agents.append_capability_call_log(agent_id, record)
        except OSError:
            pass
