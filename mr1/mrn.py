"""
MRn — Parameterized Manager Agent
==================================
Handles complex multi-step tasks at any level of the agent hierarchy.
Spawned by MR(n-1), can delegate to MR(n+1) or Kazi workers.

Level n is passed at spawn time. Model and behavior vary by level:
  Level 2: Job orchestrator (haiku)
  Level 3: Task executor (haiku)
  Level 4+: Deeper executors (haiku)

A single MRn subprocess runs via:
  claude -p "{system_prompt + task_prompt}" --allowedTools "{tools}" \
         --model {model} --output-format json \
         --bare --dangerously-skip-permissions

If the MRn output contains a [DELEGATE] block, the wrapper parses it,
spawns the child agent, collects the result, then re-runs MRn with
updated context. This loops until MRn produces a final (non-delegating)
response or the delegation round limit is reached.
"""

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from mr1.core import Dispatcher, PermissionDenied, Logger, Spawner


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
_MRN_CONFIG_PATH = _PKG_ROOT / "agents" / "mrn.yml"

# Default timeout for an MRn job (seconds).
_DEFAULT_TIMEOUT_S = 600

# Stderr patterns that indicate context window exhaustion.
_CONTEXT_EXCEEDED_PATTERNS = (
    "context window",
    "context length",
    "token limit",
    "maximum context",
    "too many tokens",
    "max_tokens",
)

# Delegation pattern used by MRn agents.
_DELEGATE_PATTERN = re.compile(
    r"\[DELEGATE\]\s*(\{.*?\})\s*\[/DELEGATE\]",
    re.DOTALL,
)

# Maximum delegation rounds per job to prevent infinite loops.
_MAX_DELEGATION_ROUNDS = 5


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
class MRnResult:
    """
    Immutable result of an MRn job.

    status is one of:
      completed         — clean exit, output captured
      failed            — non-zero exit for a reason other than context/timeout
      timeout           — process exceeded its time budget and was killed
      context_exceeded  — claude reported hitting the context window limit
      denied            — dispatcher rejected the command before spawn
      invalid           — context package was malformed
    """
    task_id: str
    level: int
    status: str
    output: str
    error: Optional[str]
    duration_s: float
    pid: Optional[int]
    sub_tasks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "duration_s": round(self.duration_s, 2),
            "pid": self.pid,
            "sub_tasks": self.sub_tasks,
        }

    @property
    def ok(self) -> bool:
        return self.status == "completed"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path = _MRN_CONFIG_PATH) -> dict:
    """Load the MRn agent YAML definition."""
    with open(path) as f:
        return yaml.safe_load(f)


def _get_model_for_level(config: dict, level: int) -> str:
    """Look up the model assignment for the given level."""
    level_models = config.get("level_models", {})
    return level_models.get(level, config.get("default_model", "haiku"))


def _detect_context_exceeded(stderr: str) -> bool:
    """Check if stderr indicates the context window was exhausted."""
    lower = stderr.lower()
    return any(pattern in lower for pattern in _CONTEXT_EXCEEDED_PATTERNS)


def _extract_output(raw_stdout: str, label: str = "MRn") -> str:
    """
    Parse the claude CLI JSON output envelope.

    With --output-format json the CLI returns:
        {"result": "<text>", "is_error": false}

    Falls back to raw text if the envelope isn't present.
    """
    raw_stdout = raw_stdout.strip()
    if not raw_stdout:
        return ""

    try:
        envelope = json.loads(raw_stdout)
        if isinstance(envelope, dict) and "result" in envelope:
            if envelope.get("is_error"):
                return f"[{label} ERROR] {envelope['result']}"
            return envelope["result"]
    except json.JSONDecodeError:
        pass

    return raw_stdout


def _parse_response(raw: str) -> tuple[str, Optional[dict]]:
    """
    Split an MRn response into display text and an optional
    delegation directive.

    Returns:
        (display_text, directive_dict_or_None)
    """
    match = _DELEGATE_PATTERN.search(raw)
    if not match:
        return raw.strip(), None

    try:
        directive = json.loads(match.group(1))
    except json.JSONDecodeError:
        return raw.strip(), None

    if "agent" not in directive or "task" not in directive:
        return raw.strip(), None

    agent = directive["agent"]
    # Accept "kazi" or any "mrN" pattern.
    if agent != "kazi" and not (agent.startswith("mr") and agent[2:].isdigit()):
        return raw.strip(), None

    display = _DELEGATE_PATTERN.sub("", raw).strip()
    return display, directive


def _build_prompt(
    instructions: str,
    file_paths: list[str],
    accumulated_results: list[dict],
    system_prompt: str,
) -> str:
    """
    Build the -p prompt string for the MRn subprocess.

    The system prompt is prepended (MRn agents don't use
    --append-system-prompt).
    """
    parts = [system_prompt, "---", instructions]
    if file_paths:
        listing = "\n".join(f"  - {p}" for p in file_paths)
        parts.append(f"RELEVANT FILES:\n{listing}")
    if accumulated_results:
        results_lines = []
        for r in accumulated_results:
            results_lines.append(
                f"[Result from {r['agent']}]\n{r['result'][:1000]}"
            )
        parts.append(
            "PREVIOUS AGENT RESULTS:\n" + "\n\n".join(results_lines)
        )
    return "\n\n".join(parts)


def _build_system_prompt(level: int, height_limit: int) -> str:
    """Build the system instructions for an MRn agent at the given level."""
    next_level = level + 1
    can_spawn_next = next_level <= height_limit

    if can_spawn_next:
        delegation_text = (
            f"You can delegate sub-tasks:\n"
            f"- To MR{next_level} for complex sub-tasks requiring decomposition\n"
            f"- To Kazi for simple, scoped one-shot jobs\n\n"
            f"To delegate, include this block in your response:\n"
            f"[DELEGATE]\n"
            f'{{"agent": "mr{next_level}", "task": "description", "context": "context"}}\n'
            f"[/DELEGATE]\n\n"
            f"Or for a kazi:\n"
            f"[DELEGATE]\n"
            f'{{"agent": "kazi", "task": "description", "context": "context"}}\n'
            f"[/DELEGATE]"
        )
    else:
        delegation_text = (
            "You can delegate simple sub-tasks to Kazi workers:\n"
            "[DELEGATE]\n"
            '{"agent": "kazi", "task": "description", "context": "context"}\n'
            "[/DELEGATE]\n\n"
            "You CANNOT spawn manager agents — you are at the maximum hierarchy depth."
        )

    return (
        f"You are MR{level}, a level-{level} agent in the MR1 multi-agent system.\n\n"
        f"You receive tasks from MR{level - 1} and execute them thoroughly.\n\n"
        f"{delegation_text}\n\n"
        f"Rules:\n"
        f"- At most ONE delegation block per response.\n"
        f"- Handle the task directly when possible — only delegate when necessary.\n"
        f"- Be thorough and precise in your work."
    )


def _fail(task_id: str, level: int, error: str) -> MRnResult:
    """Shorthand for returning an immediate failure without spawning."""
    return MRnResult(
        task_id=task_id,
        level=level,
        status="invalid",
        output="",
        error=error,
        duration_s=0.0,
        pid=None,
        sub_tasks=[],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    context: dict[str, Any],
    level: int,
    spawner: Optional[Spawner] = None,
    logger: Optional[Logger] = None,
    timeout: Optional[int] = None,
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> MRnResult:
    """
    Execute an MRn job end-to-end.

    Args:
        context: The context package. Required keys:
            task_id      (str)  — unique task identifier
            instructions (str)  — what the agent should do

          Optional keys:
            allowed_tools (list[str]) — tools the agent may use
            working_dir   (str)       — cwd for the subprocess
            file_paths    (list[str]) — files the agent should know about
            timeout       (int)       — per-job timeout override (seconds)

        level:   The MRn level (2, 3, 4, ...).
        spawner: Shared Spawner instance. Created fresh if None.
        logger:  Shared Logger instance. Created fresh if None.
        timeout: Fallback timeout if not set in the context package.

    Returns:
        MRnResult — always. Never raises, never hangs.
    """
    task_id = context.get("task_id", f"mr{level}-unknown")
    agent_type = f"mr{level}"
    label = f"MR{level}"

    # ----- Validate context package -----

    instructions = context.get("instructions")
    if not instructions or not isinstance(instructions, str):
        return _fail(task_id, level, "missing or invalid 'instructions' in context package")

    allowed_tools = context.get("allowed_tools") or []
    working_dir = context.get("working_dir")
    file_paths = context.get("file_paths") or []
    job_timeout = context.get("timeout") or timeout or _DEFAULT_TIMEOUT_S
    parent_task_id = context.get("parent_task_id", "mr1")
    lane = context.get("lane", "conversation")
    description = str(context.get("description") or instructions[:200])

    # ----- Load agent config -----

    config = _load_config()
    model = _get_model_for_level(config, level)
    tools = allowed_tools if allowed_tools else config.get("allowed_tools", [])

    # ----- Infrastructure -----

    if logger is None:
        logger = Logger()
    if spawner is None:
        spawner = Spawner(
            dispatcher=Dispatcher(),
            logger=logger,
        )

    # ----- Validate agent type is allowed -----

    try:
        spawner._dispatcher.validate_agent(agent_type)
    except PermissionDenied as e:
        return _fail(task_id, level, str(e))

    height_limit = spawner._dispatcher.height_limit

    # ----- Build system prompt -----

    system_prompt = _build_system_prompt(level, height_limit)

    # ----- Delegation loop -----

    start = time.monotonic()
    sub_tasks: list[dict[str, Any]] = []
    accumulated_results: list[dict] = []
    last_pid: Optional[int] = None
    display_text = ""

    for round_num in range(_MAX_DELEGATION_ROUNDS):
        prompt = _build_prompt(instructions, file_paths, accumulated_results, system_prompt)

        # ----- Spawn through the permission-gated spawner -----

        try:
            record = spawner.spawn(
                agent_type=agent_type,
                task_id=f"{task_id}-r{round_num}",
                prompt=prompt,
                model=model,
                tools=tools,
                extra_flags=[
                    "--output-format", "json",
                    "--bare",
                    "--dangerously-skip-permissions",
                ],
                cwd=working_dir,
            )
        except PermissionDenied as e:
            elapsed = time.monotonic() - start
            logger.log_denied(task_id, agent_type, str(e))
            _emit_event(
                event_callback,
                "task_failed",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="denied",
                lane=lane,
                description=description,
            )
            _emit_event(
                event_callback,
                "task_detached",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="denied",
                lane=lane,
                description=description,
            )
            return MRnResult(
                task_id=task_id, level=level, status="denied",
                output="", error=str(e), duration_s=elapsed,
                pid=None, sub_tasks=sub_tasks,
            )

        last_pid = record.pid
        _emit_event(
            event_callback,
            "task_spawned",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type=agent_type,
            status="running",
            pid=last_pid,
            lane=lane,
            description=description,
        )

        # ----- Wait with timeout -----

        remaining_timeout = max(1, int(job_timeout - (time.monotonic() - start)))
        try:
            stdout_raw, stderr_raw = record.process.communicate(
                timeout=remaining_timeout,
            )
        except subprocess.TimeoutExpired:
            record.process.kill()
            try:
                stdout_raw, stderr_raw = record.process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                if record.process.stdout:
                    record.process.stdout.close()
                if record.process.stderr:
                    record.process.stderr.close()
                record.process.wait()
                stdout_raw, stderr_raw = b"", b""

            elapsed = time.monotonic() - start
            logger.log(
                task_id, agent_type, "timeout", "error",
                metadata={
                    "pid": last_pid,
                    "timeout_s": job_timeout,
                    "duration_s": round(elapsed, 2),
                },
            )
            _emit_event(
                event_callback,
                "task_failed",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="timeout",
                pid=last_pid,
                lane=lane,
                description=description,
            )
            _emit_event(
                event_callback,
                "task_detached",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="timeout",
                pid=last_pid,
                lane=lane,
                description=description,
            )
            return MRnResult(
                task_id=task_id, level=level, status="timeout",
                output=_extract_output(
                    stdout_raw.decode("utf-8", errors="replace"), label,
                ),
                error=f"exceeded {job_timeout}s timeout",
                duration_s=elapsed, pid=last_pid, sub_tasks=sub_tasks,
            )

        # ----- Decode output -----

        stdout = stdout_raw.decode("utf-8", errors="replace")
        stderr = stderr_raw.decode("utf-8", errors="replace")
        returncode = record.process.returncode

        # ----- Handle failures -----

        if returncode != 0:
            elapsed = time.monotonic() - start

            if _detect_context_exceeded(stderr):
                logger.log(
                    task_id, agent_type, "context_exceeded", "error",
                    metadata={
                        "pid": last_pid,
                        "returncode": returncode,
                        "stderr": stderr[:500],
                        "duration_s": round(elapsed, 2),
                    },
                )
                _emit_event(
                    event_callback,
                    "task_failed",
                    task_id=task_id,
                    parent_task_id=parent_task_id,
                    agent_type=agent_type,
                    status="context_exceeded",
                    pid=last_pid,
                    lane=lane,
                    description=description,
                )
                _emit_event(
                    event_callback,
                    "task_detached",
                    task_id=task_id,
                    parent_task_id=parent_task_id,
                    agent_type=agent_type,
                    status="context_exceeded",
                    pid=last_pid,
                    lane=lane,
                    description=description,
                )
                return MRnResult(
                    task_id=task_id, level=level, status="context_exceeded",
                    output=_extract_output(stdout, label),
                    error=f"context window exceeded (exit {returncode})",
                    duration_s=elapsed, pid=last_pid, sub_tasks=sub_tasks,
                )

            logger.log(
                task_id, agent_type, "complete", "error",
                metadata={
                    "pid": last_pid,
                    "returncode": returncode,
                    "stderr": stderr[:500],
                    "duration_s": round(elapsed, 2),
                },
            )
            _emit_event(
                event_callback,
                "task_failed",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="failed",
                pid=last_pid,
                lane=lane,
                description=description,
            )
            _emit_event(
                event_callback,
                "task_detached",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="failed",
                pid=last_pid,
                lane=lane,
                description=description,
            )
            return MRnResult(
                task_id=task_id, level=level, status="failed",
                output=_extract_output(stdout, label),
                error=f"exit {returncode}: {stderr[:300]}",
                duration_s=elapsed, pid=last_pid, sub_tasks=sub_tasks,
            )

        # ----- Parse for delegation -----

        output = _extract_output(stdout, label)
        display_text, directive = _parse_response(output)

        if directive is None:
            # No delegation — job is done.
            elapsed = time.monotonic() - start
            logger.log(
                task_id, agent_type, "complete", "ok",
                metadata={
                    "pid": last_pid,
                    "returncode": 0,
                    "duration_s": round(elapsed, 2),
                },
            )
            _emit_event(
                event_callback,
                "task_completed",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="completed",
                pid=last_pid,
                lane=lane,
                description=description,
            )
            _emit_event(
                event_callback,
                "task_detached",
                task_id=task_id,
                parent_task_id=parent_task_id,
                agent_type=agent_type,
                status="completed",
                pid=last_pid,
                lane=lane,
                description=description,
            )
            return MRnResult(
                task_id=task_id, level=level, status="completed",
                output=display_text, error=None,
                duration_s=elapsed, pid=last_pid, sub_tasks=sub_tasks,
            )

        # ----- Execute delegation -----

        child_agent = directive["agent"]
        child_task = directive["task"]
        child_context = directive.get("context", "")
        child_task_id = f"{task_id}-sub{round_num}"

        child_instructions = child_task
        if child_context:
            child_instructions += f"\n\nCONTEXT:\n{child_context}"

        child_context_pkg = {
            "task_id": child_task_id,
            "instructions": child_instructions,
            "allowed_tools": tools,
            "parent_task_id": task_id,
            "lane": lane,
            "description": child_task,
        }

        logger.log(
            child_task_id,
            agent_type,
            "delegate",
            "ok",
            metadata={
                "to": child_agent,
                "description": child_task[:200],
                "parent_task_id": task_id,
                "lane": lane,
            },
        )
        _emit_event(
            event_callback,
            "task_attached",
            task_id=child_task_id,
            parent_task_id=task_id,
            agent_type=child_agent,
            status="pending",
            lane=lane,
            description=child_task[:200],
        )

        if child_agent == "kazi":
            from mr1 import kazi

            child_result = kazi.run(
                context=child_context_pkg,
                spawner=spawner,
                logger=logger,
                event_callback=event_callback,
            )
            sub_tasks.append({
                "agent": "kazi",
                "task_id": child_task_id,
                "status": child_result.status,
                "output": child_result.output[:500],
            })
            accumulated_results.append({
                "agent": "kazi",
                "task": child_task,
                "result": child_result.output,
            })

        elif child_agent.startswith("mr"):
            child_level = int(child_agent[2:])

            # Validate height limit before recursive spawn.
            try:
                spawner._dispatcher.validate_spawn_level(level, child_agent)
            except PermissionDenied as e:
                logger.log_denied(child_task_id, child_agent, str(e))
                accumulated_results.append({
                    "agent": child_agent,
                    "task": child_task,
                    "result": f"[BLOCKED] {e}",
                })
                continue

            child_result = run(
                context=child_context_pkg,
                level=child_level,
                spawner=spawner,
                logger=logger,
                timeout=int(max(1, job_timeout - (time.monotonic() - start))),
                event_callback=event_callback,
            )
            sub_tasks.append({
                "agent": child_agent,
                "task_id": child_task_id,
                "status": child_result.status,
                "output": child_result.output[:500],
            })
            accumulated_results.append({
                "agent": child_agent,
                "task": child_task,
                "result": child_result.output,
            })

    # ----- Exhausted delegation rounds -----

    elapsed = time.monotonic() - start
    logger.log(
        task_id, agent_type, "complete", "ok",
        metadata={
            "pid": last_pid,
            "duration_s": round(elapsed, 2),
            "rounds": _MAX_DELEGATION_ROUNDS,
        },
    )
    _emit_event(
        event_callback,
        "task_completed",
        task_id=task_id,
        parent_task_id=parent_task_id,
        agent_type=agent_type,
        status="completed",
        pid=last_pid,
        lane=lane,
        description=description,
    )
    _emit_event(
        event_callback,
        "task_detached",
        task_id=task_id,
        parent_task_id=parent_task_id,
        agent_type=agent_type,
        status="completed",
        pid=last_pid,
        lane=lane,
        description=description,
    )
    return MRnResult(
        task_id=task_id, level=level, status="completed",
        output=display_text, error=None,
        duration_s=elapsed, pid=last_pid, sub_tasks=sub_tasks,
    )
