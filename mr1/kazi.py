"""
Kazi — Ephemeral Job Agent
===========================
Spawns, does one task, dies. Knows nothing about the broader system.

A Kazi receives a context package (dict) and nothing else. It:
  1. Validates the context package
  2. Loads its own agent config from agents/kazi.yml
  3. Builds the claude CLI command from the context
  4. Passes through dispatcher validation via spawner
  5. Spawns the subprocess and waits (with timeout)
  6. Captures stdout/stderr
  7. Logs the result to tasks/{task_id}/logs/ via logger
  8. Returns a KaziResult to whoever spawned it
  9. Dies — nothing persists in this module

The subprocess command is:
  claude -p "{instructions}" --allowedTools "{tools}" \
         --model {model} --output-format json

The dispatcher has already validated that kazi is permitted to use
these flags before the process is created.
"""

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from mr1.agents import (
    AgentRuntimeError,
    is_auth_error_text,
    parse_agent_json_envelope,
    validate_agent_runtime_config,
)
from mr1.core import Dispatcher, PermissionDenied, Logger, Spawner


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
_KAZI_CONFIG_PATH = _PKG_ROOT / "agents" / "kazi.yml"

# Default timeout for a kazi job (seconds). Individual jobs can override.
_DEFAULT_TIMEOUT_S = 300

# Stderr patterns that indicate the claude process hit its context limit.
_CONTEXT_EXCEEDED_PATTERNS = (
    "context window",
    "context length",
    "token limit",
    "maximum context",
    "too many tokens",
    "max_tokens",
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _emit_event(
    callback: Optional[Callable[[dict[str, Any]], None]],
    event_type: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    callback({"type": event_type, "timestamp": _now_iso(), **payload})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KaziResult:
    """
    Immutable result of a single kazi job.

    status is one of:
      completed         — clean exit, output captured
      failed            — non-zero exit for a reason other than context/timeout
      timeout           — process exceeded its time budget and was killed
      context_exceeded  — claude reported hitting the context window limit
      denied            — dispatcher rejected the command before spawn
      invalid           — context package was malformed
    """
    task_id: str
    status: str
    output: str
    error: Optional[str]
    duration_s: float
    pid: Optional[int]
    error_type: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "duration_s": round(self.duration_s, 2),
            "pid": self.pid,
            "error_type": self.error_type,
            "payload": dict(self.payload),
        }

    @property
    def ok(self) -> bool:
        return self.status == "completed"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load the kazi agent YAML definition."""
    with open(_KAZI_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _detect_context_exceeded(stderr: str) -> bool:
    """Check if stderr indicates the context window was exhausted."""
    lower = stderr.lower()
    return any(pattern in lower for pattern in _CONTEXT_EXCEEDED_PATTERNS)


def _extract_output(raw_stdout: str) -> str:
    """
    Parse the claude CLI JSON output envelope.

    With --output-format json the CLI returns a JSON envelope.
    Falls back to raw text only for non-JSON callers and tests.
    """
    try:
        return parse_agent_json_envelope(raw_stdout)["text"]
    except AgentRuntimeError:
        return raw_stdout.strip()


def _payload_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": parsed["text"],
        "text": parsed["text"],
        "data": {
            "raw": parsed["raw"],
            "is_error": parsed["is_error"],
            "metadata": parsed["metadata"],
        },
        "metrics": {
            "usage": parsed["usage"],
        },
    }


def _build_prompt(instructions: str, file_paths: list[str]) -> str:
    """
    Build the -p prompt string from the context package fields.

    The prompt is the ONLY thing the kazi sees — all relevant information
    must be baked in here.
    """
    parts = [instructions]
    if file_paths:
        listing = "\n".join(f"  - {p}" for p in file_paths)
        parts.append(f"RELEVANT FILES:\n{listing}")
    return "\n\n".join(parts)


def _fail(task_id: str, error: str) -> KaziResult:
    """Shorthand for returning an immediate failure without spawning."""
    return KaziResult(
        task_id=task_id,
        status="invalid",
        output="",
        error=error,
        duration_s=0.0,
        pid=None,
        error_type="cli_error",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    context: dict[str, Any],
    spawner: Optional[Spawner] = None,
    logger: Optional[Logger] = None,
    timeout: Optional[int] = None,
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> KaziResult:
    """
    Execute a single kazi job end-to-end.

    This is the only public entry point. Call it, get a KaziResult, done.

    Args:
        context: The context package. Required keys:
            task_id      (str)  — unique task identifier
            instructions (str)  — what the kazi should do

          Optional keys:
            allowed_tools (list[str]) — tools the kazi may use
                                        (defaults to kazi.yml config)
            working_dir   (str)       — cwd for the subprocess
            file_paths    (list[str]) — files the kazi should know about
            timeout       (int)       — per-job timeout override (seconds)

        spawner: Shared Spawner instance. Created fresh if None.
        logger:  Shared Logger instance. Created fresh if None.
        timeout: Fallback timeout if not set in the context package.

    Returns:
        KaziResult — always. Never raises, never hangs.
    """
    # ----- Validate context package -----

    task_id = context.get("task_id", "kazi-unknown")

    instructions = context.get("instructions")
    if not instructions or not isinstance(instructions, str):
        return _fail(task_id, "missing or invalid 'instructions' in context package")

    allowed_tools = context.get("allowed_tools") or []
    working_dir = context.get("working_dir")
    file_paths = context.get("file_paths") or []
    parent_task_id = context.get("parent_task_id", "mr1")
    lane = context.get("lane", "conversation")
    description = str(context.get("description") or instructions[:200])

    # Timeout priority: context > function arg > default.
    job_timeout = context.get("timeout") or timeout or _DEFAULT_TIMEOUT_S

    # ----- Load agent config -----

    config = _load_config()
    runtime_config = validate_agent_runtime_config(
        "kazi",
        {
            "model": config.get("model"),
            "allowed_tools": config.get("allowed_tools"),
            "timeout_s": config.get("timeout_s"),
        },
    )
    model = runtime_config.get("model")

    # If the caller didn't specify tools, use the config defaults.
    tools = allowed_tools if allowed_tools else runtime_config.get("allowed_tools", [])

    # ----- Infrastructure -----

    if logger is None:
        logger = Logger()
    if spawner is None:
        spawner = Spawner(
            dispatcher=Dispatcher(),
            logger=logger,
        )

    # ----- Build prompt -----

    prompt = _build_prompt(instructions, file_paths)

    # ----- Spawn through the permission-gated spawner -----

    start = time.monotonic()

    try:
        record = spawner.spawn(
            agent_type="kazi",
            task_id=task_id,
            prompt=prompt,
            model=model,
            tools=tools,
            extra_flags=[
                "--output-format", "json",
            ],
            cwd=working_dir,
        )
    except PermissionDenied as e:
        elapsed = time.monotonic() - start
        logger.log_denied(task_id, "kazi", str(e))
        _emit_event(
            event_callback,
            "task_failed",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="denied",
            lane=lane,
            description=description,
        )
        _emit_event(
            event_callback,
            "task_detached",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="denied",
            lane=lane,
            description=description,
        )
        return KaziResult(
            task_id=task_id,
            status="denied",
            output="",
            error=str(e),
            duration_s=elapsed,
            pid=None,
            error_type="cli_error",
        )

    pid = record.pid
    _emit_event(
        event_callback,
        "task_spawned",
        task_id=task_id,
        parent_task_id=parent_task_id,
        agent_type="kazi",
        status="running",
        pid=pid,
        lane=lane,
        description=description,
    )

    # ----- Wait with timeout -----
    # Use communicate() instead of wait() to avoid pipe-buffer deadlocks.
    # communicate() reads stdout/stderr while the process runs, then blocks
    # until exit or timeout.

    try:
        stdout_raw, stderr_raw = record.process.communicate(timeout=job_timeout)
    except subprocess.TimeoutExpired:
        # Kill the hung process.
        record.process.kill()

        # Drain pipes with a short grace period. If child processes inherited
        # the pipes and are still alive after the main process is killed,
        # don't hang forever — close the pipes and move on.
        try:
            stdout_raw, stderr_raw = record.process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            # Orphaned children are holding pipes open. Force close.
            if record.process.stdout:
                record.process.stdout.close()
            if record.process.stderr:
                record.process.stderr.close()
            record.process.wait()
            stdout_raw, stderr_raw = b"", b""

        elapsed = time.monotonic() - start
        logger.log(
            task_id, "kazi", "timeout", "error",
            metadata={
                "pid": pid,
                "timeout_s": job_timeout,
                "duration_s": round(elapsed, 2),
            },
        )
        _emit_event(
            event_callback,
            "task_failed",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="timeout",
            pid=pid,
            lane=lane,
            description=description,
        )
        _emit_event(
            event_callback,
            "task_detached",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="timeout",
            pid=pid,
            lane=lane,
            description=description,
        )
        return KaziResult(
            task_id=task_id,
            status="timeout",
            output=_extract_output(stdout_raw.decode("utf-8", errors="replace")),
            error=f"exceeded {job_timeout}s timeout",
            duration_s=elapsed,
            pid=pid,
            error_type="timeout",
        )

    elapsed = time.monotonic() - start

    # ----- Decode output -----

    stdout = stdout_raw.decode("utf-8", errors="replace")
    stderr = stderr_raw.decode("utf-8", errors="replace")
    returncode = record.process.returncode
    parsed: Optional[dict[str, Any]] = None
    parse_error: Optional[str] = None

    try:
        parsed = parse_agent_json_envelope(stdout)
    except AgentRuntimeError as exc:
        parse_error = str(exc)

    # ----- Classify and log result -----

    if parse_error is not None:
        logger.log(
            task_id, "kazi", "complete", "error",
            metadata={
                "pid": pid,
                "returncode": returncode,
                "stderr": stderr[:500],
                "duration_s": round(elapsed, 2),
                "error_type": "parse_error",
            },
        )
        _emit_event(
            event_callback,
            "task_failed",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="failed",
            pid=pid,
            lane=lane,
            description=description,
        )
        _emit_event(
            event_callback,
            "task_detached",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="failed",
            pid=pid,
            lane=lane,
            description=description,
        )
        return KaziResult(
            task_id=task_id,
            status="failed",
            output="",
            error=parse_error,
            duration_s=elapsed,
            pid=pid,
            error_type="parse_error",
        )

    assert parsed is not None
    payload = _payload_from_parsed(parsed)

    if parsed["is_error"]:
        error_type = "auth_error" if is_auth_error_text(parsed["text"]) else "cli_error"
        logger.log(
            task_id, "kazi", "complete", "error",
            metadata={
                "pid": pid,
                "returncode": returncode,
                "stderr": stderr[:500],
                "duration_s": round(elapsed, 2),
                "error_type": error_type,
            },
        )
        _emit_event(
            event_callback,
            "task_failed",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="failed",
            pid=pid,
            lane=lane,
            description=description,
        )
        _emit_event(
            event_callback,
            "task_detached",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="failed",
            pid=pid,
            lane=lane,
            description=description,
        )
        payload.update({
            "status": "failed",
            "error": parsed["text"] or "agent returned is_error=true",
            "pid": pid,
            "error_type": error_type,
        })
        return KaziResult(
            task_id=task_id,
            status="failed",
            output=parsed["text"],
            error=parsed["text"] or "agent returned is_error=true",
            duration_s=elapsed,
            pid=pid,
            error_type=error_type,
            payload=payload,
        )

    if returncode == 0:
        logger.log(
            task_id, "kazi", "complete", "ok",
            metadata={
                "pid": pid,
                "returncode": 0,
                "duration_s": round(elapsed, 2),
            },
        )
        _emit_event(
            event_callback,
            "task_completed",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="completed",
            pid=pid,
            lane=lane,
            description=description,
        )
        _emit_event(
            event_callback,
            "task_detached",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="completed",
            pid=pid,
            lane=lane,
            description=description,
        )
        return KaziResult(
            task_id=task_id,
            status="completed",
            output=parsed["text"],
            error=None,
            duration_s=elapsed,
            pid=pid,
            payload={
                **payload,
                "status": "succeeded",
                "pid": pid,
            },
        )

    # Non-zero exit — check for context window exhaustion.
    if _detect_context_exceeded(stderr):
        logger.log(
            task_id, "kazi", "context_exceeded", "error",
            metadata={
                "pid": pid,
                "returncode": returncode,
                "stderr": stderr[:500],
                "duration_s": round(elapsed, 2),
                "error_type": "cli_error",
            },
        )
        _emit_event(
            event_callback,
            "task_failed",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="context_exceeded",
            pid=pid,
            lane=lane,
            description=description,
        )
        _emit_event(
            event_callback,
            "task_detached",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type="kazi",
            status="context_exceeded",
            pid=pid,
            lane=lane,
            description=description,
        )
        return KaziResult(
            task_id=task_id,
            status="context_exceeded",
            output=parsed["text"],
            error=f"context window exceeded (exit {returncode})",
            duration_s=elapsed,
            pid=pid,
            error_type="cli_error",
            payload={
                **payload,
                "status": "failed",
                "error": f"context window exceeded (exit {returncode})",
                "pid": pid,
                "error_type": "cli_error",
            },
        )

    # Generic failure.
    logger.log(
        task_id, "kazi", "complete", "error",
        metadata={
            "pid": pid,
            "returncode": returncode,
            "stderr": stderr[:500],
            "duration_s": round(elapsed, 2),
            "error_type": "cli_error",
        },
    )
    _emit_event(
        event_callback,
        "task_failed",
        task_id=task_id,
        parent_task_id=parent_task_id,
        agent_type="kazi",
        status="failed",
        pid=pid,
        lane=lane,
        description=description,
    )
    _emit_event(
        event_callback,
        "task_detached",
        task_id=task_id,
        parent_task_id=parent_task_id,
        agent_type="kazi",
        status="failed",
        pid=pid,
        lane=lane,
        description=description,
    )
    return KaziResult(
        task_id=task_id,
        status="failed",
        output=parsed["text"],
        error=f"exit {returncode}: {stderr[:300]}",
        duration_s=elapsed,
        pid=pid,
        error_type="cli_error",
        payload={
            **payload,
            "status": "failed",
            "error": f"exit {returncode}: {stderr[:300]}",
            "pid": pid,
            "error_type": "cli_error",
        },
    )
