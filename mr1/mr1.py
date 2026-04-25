"""
MR1 — Persistent Orchestrator Agent
====================================
The only truly persistent agent in the MR1 system. Conversation state is
kept across turns by resuming a Claude Code session, while MR1 itself
persists local task state, memory, and delegation history on disk.

MR1 decides whether to:
  1. Answer directly from its own knowledge/memory
  2. Spawn an MR2 agent to manage a complex multi-step task
  3. Spawn a Kazi directly for a simple one-shot job

MR1 never restarts unless /memdltr explicitly triggers the
compression + restart cycle.

State is persisted to memory/active/mr1_state.json so MR1 can
resume context after restarts.
"""

import json
import re
import signal
import shlex
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

# ---------------------------------------------------------------------------
# Path setup — ensure mr1.core is importable when run as `python -m mr1.mr1`
# or `python mr1/mr1.py`.
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PKG_ROOT.parent))

from mr1.core import Dispatcher, PermissionDenied, Logger, Spawner
from mr1 import kazi, mrn
from mr1.kazi_runner import KaziAsyncRunner, MockRunner, Runner
from mr1.scheduler import Scheduler, WatcherTriggerError, WorkflowSpecError
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore
from mr1 import workflow_cli
from mr1.workflow_authoring import (
    PendingWorkflowDraft,
    WorkflowAuthoringService,
    workflow_to_spec,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_AGENTS_DIR = _PKG_ROOT / "agents"
_STATE_PATH = _PKG_ROOT / "memory" / "active" / "mr1_state.json"
_CONTEXT_PATH = _PKG_ROOT / "memory" / "active" / "mr1_context.md"
_MR1_CONFIG_PATH = _AGENTS_DIR / "mr1.yml"
_MRN_CONFIG_PATH = _AGENTS_DIR / "mrn.yml"
_KAZI_CONFIG_PATH = _AGENTS_DIR / "kazi.yml"

# Maximum number of decisions retained in state.
_MAX_DECISIONS = 50
_MAX_CONVERSATION = 80
_TERMINAL_TASK_STATUSES = {
    "completed",
    "failed",
    "timeout",
    "context_exceeded",
    "denied",
    "killed",
}

# Maximum delegation rounds per user turn.
_MAX_DELEGATION_ROUNDS = 5
_TEST_AGENT_MAX_HEIGHT = 5
_TEST_AGENT_PREFIX = "test-agent"


# ---------------------------------------------------------------------------
# Delegation protocol
# ---------------------------------------------------------------------------
# MR1's brain embeds a structured JSON directive between these markers when
# it decides to delegate work. mr1.py extracts it, strips it from the
# display text, and routes to the appropriate agent via the spawner.
#
# Format inside the markers must be valid JSON:
#   {"agent": "mr2"|"kazi", "task": "...", "context": "..."}
# ---------------------------------------------------------------------------
_DELEGATE_PATTERN = re.compile(
    r"\[DELEGATE\]\s*(\{.*?\})\s*\[/DELEGATE\]",
    re.DOTALL,
)

# Signal that MR1 has finished writing mr1_context.md during /memdltr.
_DUMP_COMPLETE_SIGNAL = "[MR1:DUMP_COMPLETE]"


# ---------------------------------------------------------------------------
# System prompt — injected via --append-system-prompt into MR1's own
# persistent claude process. This is the brain's behavioural contract.
# ---------------------------------------------------------------------------
_ORCHESTRATOR_PROMPT = """\
You are MR1, the top-level orchestrator of a multi-agent workflow system.

== ROLE ==
You are the user's interface and decision engine. For every message, decide the best execution path:

1. DIRECT ANSWER  
   Respond yourself when the user is:
   - asking questions
   - brainstorming / discussing
   - planning / reviewing
   - asking for explanations or comparisons
   - asking what to do next

2. WORKFLOW COMPILATION  
   Convert the request into a workflow when the user wants to:
   - automate a task
   - run multiple steps
   - execute a pipeline
   - monitor or wait for something
   - connect tools / files / agents together

3. DELEGATION (RARE)  
   Only delegate to a single agent when:
   - the task is clearly a one-shot execution
   - AND workflow overhead is unnecessary
   - AND it does not require structured dataflow

Prefer workflows over delegation for anything multi-step.

---

== CRITICAL ROUTING RULES ==

DO NOT create workflows for:
- brainstorming ("let’s think", "what would be good", "ideas")
- conceptual discussion
- architecture/design questions
- comparing approaches
- asking for recommendations
- reviewing system behavior

These MUST be handled as DIRECT ANSWER.

ONLY create workflows when the user clearly intends execution.

If unsure → DIRECT ANSWER.

---

== WORKFLOW SYSTEM ==

MR1 can construct and run workflows composed of:

- TOOLS (deterministic execution)
- WATCHERS (event/wait conditions)
- AGENTS (reasoning/generation)
- DATAFLOW (passing outputs between tasks)

Workflows must:
- be valid JSON
- follow the workflow schema exactly
- use capabilities and schema metadata
- never guess field formats

---

== WORKFLOW GENERATION RULES ==

When generating workflows:

- Use tools whenever possible (fast + deterministic)
- Use agents only for reasoning or summarization
- Use watchers for waiting or conditions
- Use inputs to pass data between tasks (never inline outputs into prompts)

Inputs MUST be objects:

"inputs": [
  {"name": "x", "from": "task_label.result.text"}
]

Never:

"inputs": ["task.result.text"]

---

== DELEGATION FORMAT (RARE) ==

Only use if NOT using workflows:

[DELEGATE]
{"agent": "kazi", "task": "clear actionable instruction", "context": "relevant context"}
[/DELEGATE]

Rules:
- At most ONE block
- No text after the block
- Prefer workflows over delegation

---

== BUILT-IN COMMANDS ==

The following are handled by the system:

/status
/tasks
/kill
/history
/memdltr
/workflows
/workflow
/jobs
/events
/watchers
/capabilities
/capability
/tools
/tool
/agents
/agent
/schema
/result
/inputs
/artifacts
/vizualize
/visualize-web
/test spawn agents <h>
/test kill agents

If the user sends one of these:
Respond EXACTLY with:
Handled by MR1 system.

---

== MEMORY DUMP PROTOCOL ==

If message starts with [SYSTEM:MEMDLTR]:

1. Write memory/active/mr1_context.md containing:
   - full conversation summary
   - active tasks + status
   - learned user preferences
   - key decisions and reasoning

2. End with EXACTLY:
[MR1:DUMP_COMPLETE]

---

== PERSONALITY ==

- Concise and direct
- No filler or fluff
- No apologies
- No hedging
- Lead with the answer or action
- If unsure → say so plainly

---

== FINAL DECISION LOGIC ==

Before responding, internally decide:

Is this:
- thinking → DIRECT ANSWER
- execution → WORKFLOW
- trivial one-shot → optional DELEGATE

Never mix modes.

Return only the chosen response.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_agent_config(path: Path) -> dict:
    """Load an agent YAML definition."""
    with open(path) as f:
        return yaml.safe_load(f)


def _generate_task_id() -> str:
    """Generate a unique, timestamp-prefixed task ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"task-{ts}-{short}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# MR1 Process — Claude session runner
# ---------------------------------------------------------------------------

class MR1Process:
    """
    Manages MR1's Claude Code session.

    Claude Code does not expose a stable long-lived interactive JSON mode
    for this workflow. Instead, each turn is executed with `claude --print`
    using stream-json I/O and the prior Claude session ID is resumed when
    available.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str,
        tools: list[str],
        session_id: Optional[str] = None,
    ):
        self._system_prompt = system_prompt
        self._model = model
        self._tools = tools
        self._session_id = session_id
        self._available = False

    def start(self) -> None:
        """Verify Claude Code is available for per-turn session use."""
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"claude CLI is unavailable: {detail}")
        self._available = True

    def send(self, message: str) -> str:
        """
        Execute a single Claude turn and return the final result text.
        """
        if not self.alive:
            return "[MR1 ERROR] Process is not running."

        result_text, error_text = self._invoke(message, resume=bool(self._session_id))
        if error_text and self._session_id:
            self._session_id = None
            result_text, error_text = self._invoke(message, resume=False)

        if error_text:
            return f"[MR1 ERROR] {error_text}"

        self._available = True
        return result_text

    def _invoke(self, message: str, resume: bool) -> tuple[str, Optional[str]]:
        try:
            payload = json.dumps(
                {"type": "user", "message": {"role": "user", "content": message}}
            )
        except TypeError as exc:
            return "", f"failed to encode input: {exc}"

        cmd = [
            "claude",
            "--print",
            "--verbose",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--replay-user-messages",
        ]
        if self._model:
            cmd.extend(["--model", self._model])
        if self._tools:
            cmd.extend(["--allowedTools", ",".join(self._tools)])
        if resume and self._session_id:
            cmd.extend(["--resume", self._session_id])
        else:
            cmd.extend(["--append-system-prompt", self._system_prompt])

        try:
            result = subprocess.run(
                cmd,
                input=payload + "\n",
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return "", "claude turn timed out"
        except OSError as exc:
            return "", f"could not run claude: {exc}"

        stdout = result.stdout or ""
        stderr = (result.stderr or "").strip()
        parsed_text = ""
        parsed_session_id = None
        parse_errors = 0

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if event.get("session_id"):
                parsed_session_id = event["session_id"]
            if event.get("type") == "result":
                parsed_text = event.get("result", "")
                parsed_session_id = event.get("session_id", parsed_session_id)

        if parsed_session_id:
            self._session_id = parsed_session_id

        if result.returncode != 0:
            detail = stderr or parsed_text.strip()
            if not detail and parse_errors:
                detail = "received malformed stream-json output"
            return "", detail or f"claude exited with code {result.returncode}"

        if not parsed_text:
            detail = stderr or "claude returned no result text"
            return "", detail

        return parsed_text, None

    def kill(self) -> None:
        """Forget the current Claude session handle."""
        self._session_id = None
        self._available = False

    @property
    def pid(self) -> Optional[int]:
        return None

    @property
    def alive(self) -> bool:
        return self._available

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id


# ---------------------------------------------------------------------------
# State Manager
# ---------------------------------------------------------------------------

class StateManager:
    """
    Manages MR1's persistent state at memory/active/mr1_state.json.

    Tracks the current session, active/completed tasks, running agent
    PIDs, and a rolling window of recent orchestration decisions.
    """

    def __init__(self, state_path: Path = _STATE_PATH):
        self._path = state_path
        self._state = self._load_or_init()

    def _load_or_init(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted — reinitialise.
        return {
            "session_id": uuid.uuid4().hex[:12],
            "started_at": _now_iso(),
            "claude_session_id": None,
            "tasks": {},
            "decisions": [],
            "agent_pids": [],
            "conversation": [],
            "pending_workflow": None,
        }

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2)
        tmp.rename(self._path)  # Atomic on POSIX.

    def set_claude_session_id(self, session_id: Optional[str]) -> None:
        self._state["claude_session_id"] = session_id
        self.save()

    # -- Tasks -------------------------------------------------------------

    def begin_task(
        self,
        task_id: str,
        agent_type: str,
        description: str,
        pid: Optional[int] = None,
        parent_task_id: str = "mr1",
        lane: str = "conversation",
    ) -> None:
        existing = self._state["tasks"].get(task_id, {})
        task = {
            "agent_type": agent_type,
            "status": existing.get("status", "running"),
            "pid": existing.get("pid"),
            "description": description[:300],
            "started_at": existing.get("started_at", _now_iso()),
            "parent_task_id": existing.get("parent_task_id", parent_task_id),
            "lane": existing.get("lane", lane),
        }
        if pid is not None:
            task["pid"] = pid
        if task["status"] not in _TERMINAL_TASK_STATUSES:
            task["status"] = "running"
        self._state["tasks"][task_id] = task
        self.save()

    def add_task(
        self,
        task_id: str,
        agent_type: str,
        description: str,
        pid: Optional[int],
    ) -> None:
        self.begin_task(task_id, agent_type, description, pid=pid)

    def update_task_pid(self, task_id: str, pid: int) -> None:
        if task_id not in self._state["tasks"]:
            return
        self._state["tasks"][task_id]["pid"] = pid
        self.save()

    def complete_task(self, task_id: str, status: str = "completed") -> None:
        if task_id in self._state["tasks"]:
            self._state["tasks"][task_id]["status"] = status
            self._state["tasks"][task_id]["finished_at"] = _now_iso()
            self.save()

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        task = self._state["tasks"].get(task_id)
        if task is None:
            return None
        return dict(task)

    @property
    def active_tasks(self) -> dict:
        return {
            tid: t
            for tid, t in self._state["tasks"].items()
            if t["status"] == "running"
        }

    # -- Decisions ---------------------------------------------------------

    def add_decision(
        self,
        user_input: str,
        action: str,
        task_id: Optional[str] = None,
    ) -> None:
        self._state["decisions"].append({
            "timestamp": _now_iso(),
            "input_summary": user_input[:200],
            "action": action,
            "task_id": task_id,
        })
        # Rolling window.
        if len(self._state["decisions"]) > _MAX_DECISIONS:
            self._state["decisions"] = self._state["decisions"][-_MAX_DECISIONS:]
        self.save()

    def add_conversation(
        self,
        role: str,
        text: str,
        kind: str = "message",
        task_id: Optional[str] = None,
        lane: str = "conversation",
    ) -> dict[str, Any]:
        entry = {
            "timestamp": _now_iso(),
            "role": role,
            "text": text[:3000],
            "kind": kind,
            "task_id": task_id,
            "lane": lane,
        }
        self._state.setdefault("conversation", []).append(entry)
        if len(self._state["conversation"]) > _MAX_CONVERSATION:
            self._state["conversation"] = self._state["conversation"][-_MAX_CONVERSATION:]
        self.save()
        return entry

    def set_pending_workflow(self, draft: Optional[dict[str, Any]]) -> None:
        self._state["pending_workflow"] = draft
        self.save()

    def clear_pending_workflow(self) -> None:
        self.set_pending_workflow(None)

    # -- Agent PIDs --------------------------------------------------------

    def add_agent_pid(self, pid: int) -> None:
        if pid not in self._state.get("agent_pids", []):
            self._state.setdefault("agent_pids", []).append(pid)
            self.save()

    def remove_agent_pid(self, pid: int) -> None:
        pids = self._state.get("agent_pids", [])
        if pid in pids:
            pids.remove(pid)
            self.save()

    # -- Accessors ---------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._state["session_id"]

    @property
    def conversation(self) -> list[dict[str, Any]]:
        return list(self._state.get("conversation", []))

    @property
    def claude_session_id(self) -> Optional[str]:
        return self._state.get("claude_session_id")

    @property
    def pending_workflow(self) -> Optional[dict[str, Any]]:
        value = self._state.get("pending_workflow")
        return dict(value) if isinstance(value, dict) else None

    def format_status(self) -> str:
        """Human-readable status block."""
        active = self.active_tasks
        recent = self._state["decisions"][-5:]
        lines = [
            f"Session:  {self._state['session_id']}",
            f"Started:  {self._state['started_at']}",
            f"Active tasks: {len(active)}",
        ]
        for tid, t in active.items():
            lines.append(
                f"  {tid}  [{t['agent_type']}]  pid={t['pid']}  {t['description'][:60]}"
            )
        if recent:
            lines.append("Recent decisions:")
            for d in recent:
                lines.append(
                    f"  {d['timestamp'][:19]}  {d['action']}"
                    + (f"  ({d['task_id']})" if d.get("task_id") else "")
                )
        return "\n".join(lines)

    def format_tasks(self) -> str:
        """Human-readable task list."""
        if not self._state["tasks"]:
            return "No tasks."
        lines = []
        for tid, t in self._state["tasks"].items():
            status_icon = {
                "running": "~",
                "completed": "+",
                "failed": "!",
                "killed": "x",
            }.get(t["status"], "?")
            lines.append(
                f"  [{status_icon}] {tid}  {t['agent_type']}  "
                f"{t['status']}  {t['description'][:50]}"
            )
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Compact state summary."""
        active = self.active_tasks
        if not active:
            return "No active tasks."
        parts = []
        for tid, t in active.items():
            parts.append(f"{tid} [{t['agent_type']}]: {t['description'][:80]}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# MR1 Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class TestAgentRecord:
    task_id: str
    parent_task_id: str
    agent_type: str
    description: str
    lane: str
    process: subprocess.Popen
    started_monotonic: float
    kill_requested: bool = False

class MR1:
    """
    The persistent orchestrator. Wires together:
      - A single persistent claude process (MR1Process)
      - Delegation (MR2/Kazi subprocesses via spawner)
      - State persistence (mr1_state.json)
    """

    def __init__(
        self,
        event_sink: Optional[Callable[[dict[str, Any]], None]] = None,
        *,
        workflow_store: Optional[WorkflowStore] = None,
        workflow_runner: Optional[Runner] = None,
        workflow_concurrency: int = 4,
        workflow_auto_tick: bool = True,
        workflow_compiler: Optional[Callable[[str, str], str]] = None,
        workflow_authoring_service: Optional[WorkflowAuthoringService] = None,
    ):
        self._dispatcher = Dispatcher()
        self._logger = Logger()
        self._spawner = Spawner(
            dispatcher=self._dispatcher,
            logger=self._logger,
        )
        self._state = StateManager()
        self._event_sink = event_sink

        # Load agent configs from YAML definitions.
        self._mr1_config = _load_agent_config(_MR1_CONFIG_PATH)
        self._mrn_config = _load_agent_config(_MRN_CONFIG_PATH)
        self._kazi_config = _load_agent_config(_KAZI_CONFIG_PATH)

        # The persistent claude process — created in start().
        self._process: Optional[MR1Process] = None
        self._web_viz_server = None
        self._test_agent_lock = threading.Lock()
        self._test_agents: dict[str, TestAgentRecord] = {}

        # Workflow scheduler (Phase 1). Lives inside this MR1 process.
        self._workflow_store = workflow_store or WorkflowStore()
        runner = workflow_runner or KaziAsyncRunner(
            self._workflow_store,
            dispatcher=self._dispatcher,
        )
        self._scheduler = Scheduler(
            self._workflow_store,
            runner,
            concurrency=workflow_concurrency,
            auto_tick=workflow_auto_tick,
            agent_id="MR1",
        )
        self._workflow_authoring = workflow_authoring_service or WorkflowAuthoringService(
            self._scheduler,
            self._workflow_store,
            compiler=workflow_compiler or self._run_workflow_compiler,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Prepare MR1 for turn-by-turn Claude session use."""
        memory_context = self._load_memory_context()
        system_prompt = self._build_system_prompt(memory_context)

        self._process = MR1Process(
            system_prompt=system_prompt,
            model=self._mr1_config["model"],
            tools=self._mr1_config["allowed_tools"],
            session_id=self._state.claude_session_id,
        )
        self._process.start()
        session_id = self._process.session_id
        self._state.set_claude_session_id(session_id if isinstance(session_id, str) else None)

    def _load_memory_context(self) -> str:
        """Read the memory context file if it exists."""
        if _CONTEXT_PATH.exists():
            try:
                return _CONTEXT_PATH.read_text(encoding="utf-8")
            except OSError:
                pass
        return ""

    def _build_system_prompt(self, memory_context: str) -> str:
        """
        Build the full system prompt from the orchestrator template
        and optional memory context from a previous session.
        """
        config_block = (
            f"Agent: {self._mr1_config['name']}\n"
            f"Model: {self._mr1_config['model']}\n"
            f"Lifetime: {self._mr1_config['lifetime']}\n"
            f"Memory access: {self._mr1_config['memory_access']}\n"
            f"Available tools: {', '.join(self._mr1_config['allowed_tools'])}\n"
        )
        prompt = f"{_ORCHESTRATOR_PROMPT}\n== AGENT CONFIG ==\n{config_block}"
        if memory_context:
            prompt += (
                f"\n== MEMORY CONTEXT (from previous session) ==\n"
                f"{memory_context}"
            )
        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, Optional[dict]]:
        """
        Split a brain response into display text and an optional
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
            # Malformed directive — treat as direct answer.
            return raw.strip(), None

        # Validate required fields.
        if "agent" not in directive or "task" not in directive:
            return raw.strip(), None

        if directive["agent"] not in ("mr2", "kazi"):
            return raw.strip(), None

        # Strip the directive block from the display text.
        display = _DELEGATE_PATTERN.sub("", raw).strip()
        return display, directive

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def _emit_event(self, event_type: str, **metadata: Any) -> None:
        if self._event_sink is None:
            return
        payload = {"type": event_type, "timestamp": _now_iso(), **metadata}
        self._event_sink(payload)

    def _register_spawned_task(
        self,
        task_id: str,
        agent_type: str,
        description: str,
        parent_task_id: str,
        pid: int,
        lane: str = "conversation",
    ) -> None:
        self._logger.log(
            task_id,
            agent_type,
            "delegate",
            "ok",
            metadata={
                "description": description,
                "parent_task_id": parent_task_id,
                "lane": lane,
            },
        )
        self._state.begin_task(
            task_id=task_id,
            agent_type=agent_type,
            description=description,
            pid=pid,
            parent_task_id=parent_task_id,
            lane=lane,
        )
        self._emit_event(
            "task_attached",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type=agent_type,
            description=description,
            lane=lane,
            status="running",
        )

        self._logger.log_spawn(task_id, agent_type, pid, ["python", "-m", "mr1.test_worker"])
        self._state.update_task_pid(task_id, pid)
        self._state.add_agent_pid(pid)
        self._emit_event(
            "task_spawned",
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_type=agent_type,
            description=description,
            lane=lane,
            pid=pid,
            status="running",
        )

    def _watch_test_agent(self, task_id: str) -> None:
        with self._test_agent_lock:
            record = self._test_agents.get(task_id)
        if record is None:
            return

        returncode = record.process.wait()
        duration_s = round(time.monotonic() - record.started_monotonic, 2)

        with self._test_agent_lock:
            record = self._test_agents.pop(task_id, None)
        if record is None:
            return

        self._state.remove_agent_pid(record.process.pid)
        self._logger.log_exit(task_id, record.agent_type, record.process.pid, returncode)

        if record.kill_requested:
            return

        status = "completed" if returncode == 0 else "failed"
        self._logger.log(
            task_id,
            record.agent_type,
            "complete",
            "ok" if returncode == 0 else "error",
            metadata={
                "duration_s": duration_s,
                "lane": record.lane,
            },
        )
        self._state.complete_task(task_id, status)
        self._emit_event(
            "task_completed" if status == "completed" else "task_failed",
            task_id=task_id,
            parent_task_id=record.parent_task_id,
            agent_type=record.agent_type,
            description=record.description,
            lane=record.lane,
            pid=record.process.pid,
            status=status,
        )

    def _record_conversation(
        self,
        role: str,
        text: str,
        kind: str = "message",
        task_id: Optional[str] = None,
    ) -> None:
        entry = self._state.add_conversation(role, text, kind=kind, task_id=task_id)
        self._emit_event(
            "conversation_turn",
            role=entry["role"],
            text=entry["text"],
            kind=entry["kind"],
            task_id=entry.get("task_id"),
            lane=entry["lane"],
        )

    def _send_to_brain(self, message: str) -> str:
        if self._process is None:
            return "[MR1 ERROR] Process is not running."
        response = self._process.send(message)
        session_id = self._process.session_id
        self._state.set_claude_session_id(session_id if isinstance(session_id, str) else None)
        return response

    def _run_workflow_compiler(self, system_prompt: str, message: str) -> str:
        proc = MR1Process(system_prompt, self._mr1_config["model"], [])
        proc.start()
        return proc.send(message)

    def _answer_directly(self, user_input: str) -> str:
        raw = self._send_to_brain(
            "Answer this request directly. Do not delegate to MR2 or Kazi.\n\n"
            f"User request:\n{user_input}"
        )
        text, _ = self._parse_response(raw)
        return text

    def _record_local_response(
        self,
        text: str,
        *,
        kind: str = "message",
    ) -> str:
        if text:
            self._record_conversation("mr1", text, kind=kind)
        return text

    def _handle_task_event(self, event: dict[str, Any]) -> None:
        task_id = event.get("task_id")
        if task_id:
            self._state.begin_task(
                task_id=task_id,
                agent_type=event.get("agent_type", "agent"),
                description=event.get("description", task_id),
                pid=event.get("pid"),
                parent_task_id=event.get("parent_task_id", "mr1"),
                lane=event.get("lane", "conversation"),
            )

            if event.get("pid"):
                self._state.update_task_pid(task_id, event["pid"])
                self._state.add_agent_pid(event["pid"])

            if event["type"] == "task_detached" and event.get("pid"):
                self._state.remove_agent_pid(event["pid"])
            if event["type"] in ("task_completed", "task_failed"):
                self._state.complete_task(task_id, event.get("status", "failed"))

        self._emit_event(event["type"], **{k: v for k, v in event.items() if k != "type"})

    def build_timeline_snapshot(self) -> dict[str, Any]:
        from mr1.viz import build_snapshot

        return build_snapshot(state_path=self._state._path, tasks_dir=_PKG_ROOT / "tasks")

    def _execute_delegation(self, directive: dict, user_input: str) -> str:
        """
        Spawn the delegated agent and block until it completes.
        Routes kazi jobs through kazi.run(), MR2 jobs through mrn.run().
        Returns the agent's output text.
        """
        agent_type = directive["agent"]
        task_description = directive["task"]
        context_text = directive.get("context", "")

        task_id = _generate_task_id()

        if agent_type == "kazi":
            return self._delegate_to_kazi(
                task_id, task_description, context_text, user_input,
            )

        if agent_type == "mr2":
            return self._delegate_to_mrn(
                task_id, 2, task_description, context_text, user_input,
            )

        return f"[ERROR] Unknown agent type: {agent_type}"

    def _delegate_to_kazi(
        self,
        task_id: str,
        task_description: str,
        context_text: str,
        user_input: str,
    ) -> str:
        """Route a job through kazi.run() with a proper context package."""
        instructions = task_description
        if context_text:
            instructions += f"\n\nCONTEXT:\n{context_text}"

        context_pkg = {
            "task_id": task_id,
            "instructions": instructions,
            "allowed_tools": self._kazi_config["allowed_tools"],
            "parent_task_id": "mr1",
            "lane": "conversation",
            "description": task_description,
        }

        self._logger.log(
            task_id, "mr1", "delegate", "ok",
            metadata={
                "to": "kazi",
                "description": task_description[:200],
                "parent_task_id": "mr1",
                "lane": "conversation",
            },
        )
        self._handle_task_event(
            {
                "type": "task_attached",
                "task_id": task_id,
                "parent_task_id": "mr1",
                "agent_type": "kazi",
                "description": task_description[:200],
                "lane": "conversation",
            }
        )

        result = kazi.run(
            context=context_pkg,
            spawner=self._spawner,
            logger=self._logger,
            event_callback=self._handle_task_event,
        )

        self._state.complete_task(task_id, result.status)
        self._state.add_decision(user_input, "spawn_kazi", task_id)

        if result.ok:
            return result.output
        return f"[KAZI {result.status.upper()}] {result.error or 'unknown error'}"

    def _delegate_to_mrn(
        self,
        task_id: str,
        level: int,
        task_description: str,
        context_text: str,
        user_input: str,
    ) -> str:
        """Spawn an MRn agent through mrn.run()."""
        instructions = task_description
        if context_text:
            instructions += f"\n\nCONTEXT:\n{context_text}"

        context_pkg = {
            "task_id": task_id,
            "instructions": instructions,
            "parent_task_id": "mr1",
            "lane": "conversation",
            "description": task_description,
        }

        self._logger.log(
            task_id, "mr1", "delegate", "ok",
            metadata={
                "to": f"mr{level}",
                "description": task_description[:200],
                "parent_task_id": "mr1",
                "lane": "conversation",
            },
        )
        self._handle_task_event(
            {
                "type": "task_attached",
                "task_id": task_id,
                "parent_task_id": "mr1",
                "agent_type": f"mr{level}",
                "description": task_description[:200],
                "lane": "conversation",
            }
        )

        try:
            result = mrn.run(
                context=context_pkg,
                level=level,
                spawner=self._spawner,
                logger=self._logger,
                event_callback=self._handle_task_event,
            )
        except PermissionDenied as e:
            self._logger.log_denied(task_id, f"mr{level}", str(e))
            self._state.add_decision(user_input, f"denied_mr{level}", task_id)
            return f"[BLOCKED] Permission denied for MR{level}: {e}"

        self._state.complete_task(task_id, result.status)
        self._state.add_decision(user_input, f"spawn_mr{level}", task_id)

        if result.ok:
            return result.output
        return f"[MR{level} {result.status.upper()}] {result.error or 'unknown error'}"

    # ------------------------------------------------------------------
    # Conversation step
    # ------------------------------------------------------------------

    def step(self, user_input: str, announce: bool = False) -> str:
        """
        Process one turn of conversation.

        Phase 5 is compiler-first for normal turns:
          1. Decide direct answer vs workflow authoring
          2. For workflow turns: compile, validate, preview, submit
          3. For direct answers: ask MR1 to answer without delegation
        """
        self._record_conversation("user", user_input)
        pending = self._workflow_authoring.coerce_pending_draft(
            self._state.pending_workflow
        )
        action = self._workflow_authoring.classify_request(
            user_input,
            pending_draft=pending,
        )

        if action == "direct_answer":
            self._state.add_decision(user_input, "direct_answer")
            return self._record_local_response(self._answer_directly(user_input))

        if action == "show_json_preview":
            if pending is None:
                return self._record_local_response("No pending workflow draft.")
            return self._record_local_response(
                json.dumps(pending.spec, indent=2),
                kind="workflow_json",
            )

        if action == "cancel_preview":
            self._state.clear_pending_workflow()
            self._state.add_decision(user_input, "cancel_workflow_preview")
            return self._record_local_response("Cancelled pending workflow draft.")

        if action == "confirm_preview":
            if pending is None:
                return self._record_local_response("No pending workflow draft.")
            result = self._workflow_authoring.submit(
                pending.spec,
                created_by=Provenance(type="agent", id="MR1"),
                target_workflow_id=pending.target_workflow_id,
            )
            self._state.clear_pending_workflow()
            self._state.add_decision(
                user_input,
                "submit_pending_workflow",
                result.workflow_id,
            )
            return self._record_local_response(result.message)

        mode = "modify" if action == "modify_workflow" else "create"
        target_workflow_id = self._workflow_authoring.extract_workflow_id(user_input)
        baseline_spec: Optional[dict[str, Any]] = None
        if pending is not None:
            baseline_spec = pending.spec
            target_workflow_id = pending.target_workflow_id or target_workflow_id
        elif target_workflow_id:
            workflow = self._workflow_store.load_workflow(target_workflow_id)
            if workflow is None:
                return self._record_local_response(
                    f"workflow not found: {target_workflow_id}"
                )
            baseline_spec = workflow_to_spec(workflow)

        if mode == "modify" and baseline_spec is None:
            return self._record_local_response(
                self._workflow_authoring.clarify_message(
                    "missing workflow target",
                    mode=mode,
                    target_workflow_id=target_workflow_id,
                )
            )

        try:
            spec = self._workflow_authoring.generate_spec(
                user_input,
                mode=mode,
                baseline_spec=baseline_spec,
            )
        except (RuntimeError, json.JSONDecodeError, WorkflowSpecError, ValueError) as exc:
            return self._record_local_response(
                self._workflow_authoring.clarify_message(
                    str(exc),
                    mode=mode,
                    target_workflow_id=target_workflow_id,
                )
            )

        validation = self._workflow_authoring.validate_and_maybe_fix(spec)
        if not validation.ok or validation.spec is None:
            return self._record_local_response(
                self._workflow_authoring.clarify_message(
                    validation.error or "workflow validation failed",
                    mode=mode,
                    target_workflow_id=target_workflow_id,
                )
            )

        preview_text, complexity = self._workflow_authoring.preview(validation.spec)
        if complexity == "simple":
            result = self._workflow_authoring.submit(
                validation.spec,
                created_by=Provenance(type="agent", id="MR1"),
                target_workflow_id=target_workflow_id,
            )
            self._state.clear_pending_workflow()
            self._state.add_decision(
                user_input,
                "modify_workflow" if mode == "modify" else "submit_workflow",
                result.workflow_id,
            )
            return self._record_local_response(result.message)

        draft = PendingWorkflowDraft(
            original_request=user_input,
            mode=mode,
            spec=validation.spec,
            target_workflow_id=target_workflow_id,
            preview_text=preview_text,
            complexity=complexity,
        )
        self._state.set_pending_workflow(draft.to_dict())
        self._state.add_decision(
            user_input,
            "preview_workflow_modification" if mode == "modify" else "preview_workflow",
            target_workflow_id,
        )
        return self._record_local_response(preview_text, kind="workflow_preview")

    # ------------------------------------------------------------------
    # Memory dump + restart (/memdltr)
    # ------------------------------------------------------------------

    def trigger_memdltr(self) -> str:
        """
        Trigger the full compression + restart cycle:
          1. Ask MR1 to dump context to mr1_context.md
          2. Wait for the dump completion signal
          3. Run mem_dltr.distill()
          4. Kill MR1 process
          5. Spawn a fresh MR1 process
        """
        response = self._send_to_brain(
            "[SYSTEM:MEMDLTR] Dump everything important about this conversation "
            "to memory/active/mr1_context.md. Include: full conversation summary, "
            "active tasks, user preferences, key decisions. After writing the file, "
            "end your response with exactly: [MR1:DUMP_COMPLETE]"
        )

        dump_confirmed = _DUMP_COMPLETE_SIGNAL in response
        self._emit_event(
            "system_event",
            lane="system",
            summary="memory distillation started",
            agent_type="mr1",
        )

        # Run distillation.
        from mr1.mini.mem_dltr import distill
        dltr_result = distill(logger=self._logger)

        # Kill and restart.
        self._process.kill()
        self._state.set_claude_session_id(None)
        self.start()

        status = "confirmed" if dump_confirmed else "unconfirmed"
        message = (
            f"Memory compressed (dump {status}). "
            f"Distilled: {dltr_result.forgotten} forgotten, "
            f"{dltr_result.dumped} dumped, {dltr_result.rag_chunks} RAG chunks. "
            f"MR1 restarted with fresh context."
        )
        self._emit_event(
            "system_event",
            lane="system",
            summary=message,
            agent_type="mem_dltr",
        )
        return message

    # ------------------------------------------------------------------
    # Built-in commands (handled locally, never sent to the brain)
    # ------------------------------------------------------------------

    def launch_visualizer(self) -> str:
        """Explain how to switch to the primary Ink-based MR1 interface."""
        return (
            "Timeline UI is now the primary MR1 interface. "
            "Exit this plain session and run `python main.py` or `npm run viz`. "
            "Use `python main.py --plain` to stay in the legacy loop."
        )

    def launch_web_visualizer(self) -> str:
        from mr1.web_viz import WebVizServer

        if self._web_viz_server is None:
            self._web_viz_server = WebVizServer(self)
        url = self._web_viz_server.start(open_browser=False)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        return (
            f"MR1 web visualizer running at {url}. "
            "It should open in your browser automatically. "
            "You can also launch it directly with `python main.py --web`."
        )

    def spawn_test_agents(self, height: int) -> str:
        if height < 0 or height > _TEST_AGENT_MAX_HEIGHT:
            return f"Height must be between 0 and {_TEST_AGENT_MAX_HEIGHT}."

        with self._test_agent_lock:
            active = [
                record for record in self._test_agents.values()
                if record.process.poll() is None
            ]
            if active:
                return "Synthetic test agents are already running. Use `/test kill agents` first."

        run_id = uuid.uuid4().hex[:8]
        spawned = 0
        current_level = ["mr1"]
        project_root = str(_PKG_ROOT.parent)

        for depth in range(height + 1):
            next_level: list[str] = []
            for index in range(2 ** depth):
                parent_task_id = current_level[index // 2] if depth > 0 else "mr1"
                task_id = f"{_TEST_AGENT_PREFIX}-{run_id}-d{depth}-n{index}"
                description = f"synthetic branch depth {depth} node {index}"
                duration_s = max(8, 26 - depth * 2 + (index % 3))
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "mr1.test_worker",
                        "--sleep",
                        str(duration_s),
                    ],
                    cwd=project_root,
                )
                record = TestAgentRecord(
                    task_id=task_id,
                    parent_task_id=parent_task_id,
                    agent_type="test_agent",
                    description=description,
                    lane="conversation",
                    process=process,
                    started_monotonic=time.monotonic(),
                )
                with self._test_agent_lock:
                    self._test_agents[task_id] = record
                self._register_spawned_task(
                    task_id=task_id,
                    agent_type=record.agent_type,
                    description=record.description,
                    parent_task_id=record.parent_task_id,
                    pid=process.pid,
                    lane=record.lane,
                )
                watcher = threading.Thread(
                    target=self._watch_test_agent,
                    args=(task_id,),
                    daemon=True,
                )
                watcher.start()
                spawned += 1
                next_level.extend([task_id, task_id])
            current_level = next_level

        self._state.add_decision(
            f"/test spawn agents {height}",
            f"spawn_test_agents_h{height}",
        )
        return f"Spawned {spawned} synthetic agents in a full binary tree of height {height}."

    def kill_test_agents(self) -> str:
        with self._test_agent_lock:
            records = [
                record for record in self._test_agents.values()
                if record.process.poll() is None
            ]

        if not records:
            return "No synthetic test agents are running."

        killed = 0
        for record in records:
            record.kill_requested = True
            self._logger.log_kill(record.task_id, record.agent_type, record.process.pid, "test_kill")
            self._state.complete_task(record.task_id, "killed")
            self._state.remove_agent_pid(record.process.pid)
            self._emit_event(
                "task_detached",
                task_id=record.task_id,
                parent_task_id=record.parent_task_id,
                agent_type=record.agent_type,
                description=record.description,
                lane=record.lane,
                pid=record.process.pid,
                status="killed",
            )
            try:
                record.process.terminate()
            except OSError:
                pass
            killed += 1

        self._state.add_decision("/test kill agents", "kill_test_agents")
        return f"Killed {killed} synthetic test agent(s)."

    def _handle_builtin(self, cmd: str) -> Optional[str]:
        """
        Handle slash commands locally. Returns the output string,
        or None if the input is not a built-in command.
        """
        cmd = cmd.strip()
        if cmd == "/status":
            return self._state.format_status()
        if cmd == "/tasks":
            return self._state.format_tasks()
        if cmd == "/kill":
            killed = self._spawner.kill_all("user_kill")
            synthetic_result = self.kill_test_agents()
            synthetic_killed = 0 if synthetic_result.startswith("No synthetic") else int(synthetic_result.split()[1])
            # Mark all running tasks as killed.
            for tid in list(self._state.active_tasks):
                self._state.complete_task(tid, "killed")
            total = killed + synthetic_killed
            return f"Terminated {total} agent(s)." if total else "No agents running."
        if cmd == "/history":
            recent = self._state._state["decisions"][-10:]
            if not recent:
                return "No recent decisions."
            lines = []
            for d in recent:
                lines.append(
                    f"  {d['timestamp'][:19]}  {d['action']}  "
                    f"{d.get('input_summary', '')[:60]}"
                )
            return "\n".join(lines)
        if cmd == "/memdltr":
            return self.trigger_memdltr()
        if cmd.startswith("/test spawn agents"):
            parts = cmd.split()
            if len(parts) != 4:
                return "Usage: /test spawn agents <height>"
            try:
                height = int(parts[3])
            except ValueError:
                return "Usage: /test spawn agents <height>"
            return self.spawn_test_agents(height)
        if cmd == "/test kill agents":
            return self.kill_test_agents()
        if cmd in ("/vizualize", "/visualize"):
            return self.launch_visualizer()
        if cmd in ("/visualize-web", "/vizualize-web"):
            return self.launch_web_visualizer()
        if cmd == "/workflows":
            return workflow_cli._format_workflows_table(
                self._scheduler.list_workflows()
            )
        if cmd == "/watchers":
            return workflow_cli._format_watchers(self._scheduler.list_workflows())
        if cmd == "/agents" or cmd.startswith("/agents "):
            return self._handle_agent_builtin(cmd)
        if cmd.startswith("/agent"):
            return self._handle_agent_builtin(cmd)
        if cmd == "/tools" or cmd.startswith("/tools "):
            return self._handle_capability_builtin(cmd)
        if cmd == "/capabilities" or cmd.startswith("/capabilities "):
            return self._handle_capability_builtin(cmd)
        if cmd.startswith("/capability"):
            return self._handle_capability_builtin(cmd)
        if cmd.startswith("/tool"):
            return self._handle_capability_builtin(cmd)
        if cmd == "/schema" or cmd.startswith("/schema "):
            return self._handle_schema_builtin(cmd)
        if cmd.startswith("/workflow "):
            rest = cmd[len("/workflow "):].strip()
            if rest.startswith("submit "):
                path_str = rest[len("submit "):].strip()
                return self._submit_workflow_from_path(path_str)
            if rest.startswith("rerun "):
                parts = rest.split(maxsplit=2)
                if len(parts) != 3:
                    return "Usage: /workflow rerun <workflow_id> <task>"
                try:
                    task_id = self._scheduler.rerun_task(parts[1], parts[2])
                except WorkflowSpecError as exc:
                    return str(exc)
                self._scheduler.tick()
                return f"rerun scheduled: {task_id}"
            if rest.startswith("cancel "):
                parts = rest.split(maxsplit=1)
                if len(parts) != 2:
                    return "Usage: /workflow cancel <workflow_id>"
                try:
                    cancelled = self._scheduler.cancel_workflow(parts[1])
                except WorkflowSpecError as exc:
                    return str(exc)
                self._scheduler.tick()
                return (
                    f"workflow cancelled: {parts[1]}"
                    if cancelled else f"workflow not found: {parts[1]}"
                )
            if rest.startswith("append "):
                parts = rest.split(maxsplit=2)
                if len(parts) != 3:
                    return "Usage: /workflow append <workflow_id> <path>"
                spec, error = workflow_cli._load_json_file(parts[2])
                if error:
                    return error
                try:
                    workflow_id = self._scheduler.append_workflow(parts[1], spec)
                except WorkflowSpecError as exc:
                    return str(exc)
                self._scheduler.tick()
                return f"workflow updated: {workflow_id}"
            if rest.startswith("insert "):
                parts = rest.split(maxsplit=3)
                if len(parts) != 4:
                    return "Usage: /workflow insert <workflow_id> <after_task> <path>"
                spec, error = workflow_cli._load_json_file(parts[3])
                if error:
                    return error
                try:
                    workflow_id = self._scheduler.insert_workflow(parts[1], parts[2], spec)
                except WorkflowSpecError as exc:
                    return str(exc)
                self._scheduler.tick()
                return f"workflow updated: {workflow_id}"
            if rest.startswith("replace "):
                try:
                    parts = shlex.split(rest)
                except ValueError:
                    return "Usage: /workflow replace [-r] <workflow_id> <task> <path>"
                rerun_after_replace = False
                if len(parts) > 1 and parts[1] == "-r":
                    rerun_after_replace = True
                    parts = [parts[0], *parts[2:]]
                if len(parts) != 4:
                    return "Usage: /workflow replace [-r] <workflow_id> <task> <path>"
                spec, error = workflow_cli._load_json_file(parts[3])
                if error:
                    return error
                try:
                    workflow_id = self._scheduler.replace_workflow(parts[1], parts[2], spec)
                except WorkflowSpecError as exc:
                    return str(exc)
                if rerun_after_replace:
                    self._scheduler.tick()
                    return f"workflow updated and rerun: {workflow_id}"
                return f"workflow updated: {workflow_id}"
            if rest.startswith("trigger "):
                parts = rest.split(maxsplit=3)
                if len(parts) < 3:
                    return "Usage: /workflow trigger <workflow_id> <label-or-task-id> [event_name]"
                wf_id = parts[1]
                label_or_task_id = parts[2]
                event_name = parts[3] if len(parts) > 3 else None
                try:
                    task_id = self._scheduler.trigger_watcher(
                        wf_id,
                        label_or_task_id,
                        event_name=event_name,
                    )
                except WatcherTriggerError as exc:
                    return str(exc)
                self._scheduler.tick()
                return f"triggered watcher: {task_id}"
            wf_id = rest
            wf = self._scheduler.get_workflow(wf_id)
            if wf is None:
                return f"workflow not found: {wf_id}"
            return workflow_cli._format_workflow_detail(wf)
        if cmd.startswith("/task "):
            rest = cmd[len("/task "):].strip()
            if rest.startswith("cancel "):
                task_id = rest[len("cancel "):].strip()
                if not task_id:
                    return "Usage: /task cancel <task_id>"
                try:
                    cancelled = self._scheduler.cancel_task(task_id)
                except WorkflowSpecError as exc:
                    return str(exc)
                self._scheduler.tick()
                return f"task cancelled: {cancelled}"
            task_id = rest
            wf, task = workflow_cli._find_workflow_for_task(
                self._workflow_store, task_id
            )
            if wf is None or task is None:
                return f"task not found: {task_id}"
            return workflow_cli._format_task_detail(wf, task)
        if cmd.startswith("/result "):
            task_id = cmd[len("/result "):].strip()
            wf, task = workflow_cli._find_workflow_for_task(
                self._workflow_store, task_id
            )
            if wf is None or task is None:
                return f"task not found: {task_id}"
            output = self._workflow_store.load_task_output(wf.workflow_id, task.task_id)
            return workflow_cli._format_result(task, output)
        if cmd.startswith("/inputs "):
            task_id = cmd[len("/inputs "):].strip()
            wf, task = workflow_cli._find_workflow_for_task(
                self._workflow_store, task_id
            )
            if wf is None or task is None:
                return f"task not found: {task_id}"
            inputs = self._workflow_store.load_task_inputs(wf.workflow_id, task.task_id)
            return workflow_cli._format_inputs(task, inputs)
        if cmd.startswith("/artifacts "):
            wf_id = cmd[len("/artifacts "):].strip()
            wf = self._scheduler.get_workflow(wf_id)
            if wf is None:
                return f"workflow not found: {wf_id}"
            return workflow_cli._format_artifacts(wf)
        if cmd == "/jobs":
            return workflow_cli._format_jobs(self._scheduler.list_workflows())
        if cmd.startswith("/events "):
            wf_id = cmd[len("/events "):].strip()
            if self._scheduler.get_workflow(wf_id) is None:
                return f"workflow not found: {wf_id}"
            events = self._workflow_store.load_events(wf_id, limit=50)
            return workflow_cli._format_events(events)
        if cmd == "/scheduler tick":
            self._scheduler.tick()
            return "scheduler ticked."
        return None

    def _handle_capability_builtin(self, cmd: str) -> str:
        try:
            parts = shlex.split(cmd)
        except ValueError:
            if cmd.startswith("/capability"):
                return "usage: /capability <name> [--json] [--example] [--brief]"
            if cmd.startswith("/tool"):
                return "usage: /tool <tool_type> [--json] [--example] [--brief]"
            if cmd.startswith("/capabilities"):
                return "usage: /capabilities [--json] [--brief]"
            return "usage: /tools [--json] [--brief]"

        command = parts[0]
        flags = {part for part in parts[1:] if part.startswith("--")}
        positionals = [part for part in parts[1:] if not part.startswith("--")]
        allowed_flags = {"--json", "--brief"}
        if command in {"/capability", "/tool"}:
            allowed_flags.add("--example")
        if any(flag not in allowed_flags for flag in flags):
            if command == "/capability":
                return "usage: /capability <name> [--json] [--example] [--brief]"
            if command == "/tool":
                return "usage: /tool <tool_type> [--json] [--example] [--brief]"
            if command == "/capabilities":
                return "usage: /capabilities [--json] [--brief]"
            return "usage: /tools [--json] [--brief]"
        if "--example" in flags and "--brief" in flags:
            return "invalid flag combination"

        if command == "/capabilities":
            if positionals:
                return "usage: /capabilities [--json] [--brief]"
            return workflow_cli._format_capabilities(
                json_output="--json" in flags,
                brief="--brief" in flags,
            )
        if command == "/capability":
            if len(positionals) != 1:
                return "usage: /capability <name> [--json] [--example] [--brief]"
            try:
                return workflow_cli._format_capability(
                    positionals[0],
                    json_output="--json" in flags,
                    example_only="--example" in flags,
                    brief="--brief" in flags,
                )
            except ValueError:
                return f"capability not found: {positionals[0]}"
        if command == "/tools":
            if positionals:
                return "usage: /tools [--json] [--brief]"
            return workflow_cli._format_tools(
                json_output="--json" in flags,
                brief="--brief" in flags,
            )
        if len(positionals) != 1:
            return "usage: /tool <tool_type> [--json] [--example] [--brief]"
        try:
            return workflow_cli._format_tool(
                positionals[0],
                json_output="--json" in flags,
                example_only="--example" in flags,
                brief="--brief" in flags,
            )
        except ValueError:
            return f"tool not found: {positionals[0]}"

    def _handle_agent_builtin(self, cmd: str) -> str:
        try:
            parts = shlex.split(cmd)
        except ValueError:
            if cmd.startswith("/agent"):
                return "usage: /agent <name> [health] [--json] [--brief]"
            return "usage: /agents [--json] [--brief]"

        command = parts[0]
        flags = {part for part in parts[1:] if part.startswith("--")}
        positionals = [part for part in parts[1:] if not part.startswith("--")]
        allowed_flags = {"--json", "--brief"}
        if any(flag not in allowed_flags for flag in flags):
            if command == "/agent":
                return "usage: /agent <name> [health] [--json] [--brief]"
            return "usage: /agents [--json] [--brief]"

        if command == "/agents":
            if positionals:
                return "usage: /agents [--json] [--brief]"
            return workflow_cli._format_agents(
                json_output="--json" in flags,
                brief="--brief" in flags,
            )

        if not positionals:
            return "usage: /agent <name> [health] [--json] [--brief]"
        agent_name = positionals[0]
        action = positionals[1] if len(positionals) > 1 else None
        if len(positionals) > 2 or (action is not None and action != "health"):
            return "usage: /agent <name> [health] [--json] [--brief]"
        try:
            if action == "health":
                return workflow_cli._format_agent_health(
                    agent_name,
                    json_output="--json" in flags,
                )
            return workflow_cli._format_agent(
                agent_name,
                json_output="--json" in flags,
                brief="--brief" in flags,
            )
        except ValueError:
            return f"agent not found: {agent_name}"

    def _handle_schema_builtin(self, cmd: str) -> str:
        usage = "usage: /schema [workflow|task|inputs|refs|task-kinds] [--json] [--brief]"
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return usage

        flags = {part for part in parts[1:] if part.startswith("--")}
        positionals = [part for part in parts[1:] if not part.startswith("--")]
        allowed_flags = {"--json", "--brief"}
        if any(flag not in allowed_flags for flag in flags):
            return usage
        if len(positionals) > 1:
            return usage
        try:
            return workflow_cli._format_schema(
                positionals[0] if positionals else None,
                json_output="--json" in flags,
                brief="--brief" in flags,
            )
        except ValueError as exc:
            return f"error: {exc}"

    def _submit_workflow_from_path(self, path_str: str) -> str:
        path = Path(path_str)
        if not path.exists():
            return f"spec file not found: {path}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                spec = json.load(f)
        except json.JSONDecodeError as exc:
            return f"invalid JSON: {exc}"
        try:
            wf_id = self._scheduler.submit_workflow(
                spec, Provenance(type="agent", id="MR1")
            )
        except WorkflowSpecError as exc:
            return f"invalid workflow: {exc}"
        return f"submitted: {wf_id}"

    def shutdown(self, reason: str = "user") -> int:
        killed = self._spawner.kill_all(reason)
        self.kill_test_agents()
        self._scheduler.shutdown(cancel_running=True)
        if self._web_viz_server is not None:
            self._web_viz_server.stop()
            self._web_viz_server = None
        if self._process:
            self._process.kill()
        self._state.set_claude_session_id(None)
        self._state.save()
        return killed

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start MR1 and enter the persistent conversation loop.
        Reads from stdin, writes to stdout.
        This IS the user's interface to MR1.
        """
        self.start()

        print("MR1 Orchestrator v0.2")
        print(f"Session: {self._state.session_id}")
        print(
            "Commands: /status  /tasks  /kill  /history  /memdltr  "
            "/workflows  /watchers  /capabilities  /capability <name>  "
            "/tools  /tool <type>  /agents  /agent <name>  /schema  /vizualize  /visualize-web  "
            "/test spawn agents <h>  /test kill agents"
        )
        print("Type 'exit' or Ctrl+C to quit.\n")

        def shutdown(killed_by: str = "user") -> None:
            killed = self.shutdown("shutdown")
            if killed:
                print(f"\n[mr1] Terminated {killed} running agent(s).")
            print("[mr1] Session saved. Goodbye.")
            sys.exit(0)

        signal.signal(signal.SIGINT, lambda *_: shutdown("sigint"))

        while True:
            try:
                user_input = input("\nyou > ").strip()
            except EOFError:
                shutdown("eof")

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                shutdown()

            # Check for built-in slash commands first.
            builtin_result = self._handle_builtin(user_input)
            if builtin_result is not None:
                print(f"\n{builtin_result}")
                continue

            # Normal conversation turn — goes through the persistent process.
            response = self.step(user_input, announce=True)
            print(f"\nmr1 > {response}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mr1 = MR1()
    mr1.run()


if __name__ == "__main__":
    main()
