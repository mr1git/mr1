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
from mr1.capability_policy import (
    CapabilityApprovalDecision,
    CapabilityApprovalStore,
)
from mr1.kazi_runner import KaziAsyncRunner, MockRunner, Runner
from mr1.messages import MessageStore
from mr1.mrn_loop import MRnStepRunner
from mr1.mrn_run import MRnRunPolicy, MRnRunRunner
from mr1.scheduler import Scheduler, WatcherTriggerError, WorkflowSpecError
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore
from mr1.event_log import EventLog
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore
from mr1 import workflow_cli
from mr1.workflow_authoring import (
    AuthoringResult,
    PendingWorkflowDraft,
    WorkflowAuthoringService,
    workflow_to_spec,
)
from mr1.workflow_compiler import WorkflowCompilerClient
from mr1.inbox_triage import InboxTriagePolicy, InboxTriageRunner


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
_GROUNDING_AGENT_LIMIT = 8
_GROUNDING_WORKFLOW_LIMIT = 8
_GROUNDING_MESSAGE_LIMIT = 8
_GROUNDING_APPROVAL_LIMIT = 8
_GROUNDING_EVENT_LIMIT = 10
_GROUNDING_MESSAGE_BODY_LIMIT = 4096
_GROUNDING_MESSAGE_TRUNCATION_SUFFIX = "...[truncated, use message_id for full]"
_AGENT_ID_PATTERN = re.compile(r"\bag-\d{8}T\d{6}-[0-9a-f]{6}\b")
_WORKFLOW_ID_PATTERN = re.compile(r"\bwf-\d{8}T\d{6}-[0-9a-f]{6}\b")
_TASK_ID_PATTERN = re.compile(r"\btk-\d{8}T\d{6}-[0-9a-f]{6}\b")
_MESSAGE_ID_PATTERN = re.compile(r"\bmsg-\d{8}T\d{6,}-[0-9a-f]{6}\b")
_REPLY_INTENT_PATTERN = re.compile(
    r"\b(reply|respond|clarify)(?:\s+to)?(?:\s+this)?(?:\s+message)?\b",
    re.IGNORECASE,
)
_WORKFLOW_INSPECTION_INTENT_PHRASES = (
    "check",
    "inspect",
    "summarize",
    "summary",
    "results",
    "result",
    "status",
    "what happened",
    "did it finish",
    "did the workflow finish",
    "why failed",
    "what failed",
    "show findings",
    "findings",
    "finish running",
)
_WORKFLOW_FINDINGS_INTENT_PHRASES = (
    "findings",
    "conclusion",
    "what did it find",
    "summarize findings",
    "answer/result of the workflow",
    "answer of the workflow",
    "result of the workflow",
    "what were the findings",
)
_WORKFLOW_AUTHORING_SUPPRESSION_PHRASES = (
    "don't create a workflow",
    "do not create a workflow",
    "not asking you to create a workflow",
    "this already ran",
    "check existing workflow",
    "inspect existing result",
)
_WORKFLOW_REFERENCE_ALIASES = (
    "that workflow",
    "the workflow",
    "this workflow",
)
_BULK_AGENT_ACTIONS = (
    "kill",
    "terminate",
    "resume",
    "run",
    "message",
)
_FINDINGS_PREFERRED_LABEL_TOKENS = (
    "synthesize",
    "summarize",
    "analyze",
    "final",
    "report",
    "answer",
)


def _has_workflow_pronoun_reference(normalized: str) -> bool:
    return bool(re.search(r"\bit\b", normalized))


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

_PERSISTENT_DELEGATION_MARKERS = (
    "create a child",
    "create child",
    "create an agent",
    "create agent",
    "delegate this domain",
    "delegate this area",
    "delegate this responsibility",
    "have an agent own",
    "have a child own",
    "have an mr2 own",
    "let that agent",
    "let the agent",
    "child responsible for",
    "agent responsible for",
)

_PERSISTENT_CHILD_TITLE_PATTERNS = (
    re.compile(r"\bchild(?:\s+of\s+yours)?(?:\s+agent)?[,\s]+(MR\d+)\b", re.IGNORECASE),
    re.compile(r"\bagent\s+(MR\d+)\b", re.IGNORECASE),
    re.compile(r"\bnamed\s+(MR\d+)\b", re.IGNORECASE),
)

_META_EXPLANATION_PATTERNS = (
    re.compile(r"\bin what situation(?:s)? (?:would you|you would)\b", re.IGNORECASE),
    re.compile(r"\bwhat situation(?:s)? would you use\b", re.IGNORECASE),
    re.compile(r"\bwhen would you\b", re.IGNORECASE),
    re.compile(r"\bwhen should (?:mr1|you)\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\btools?\s+vs\.?\s+workflows?\s+vs\.?\s+agents?\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) the difference between\b", re.IGNORECASE),
)

_PERSISTENT_DELEGATION_IMPERATIVE_PATTERNS = (
    re.compile(r"\bcreate (?:a|an) (?:child|agent)\b", re.IGNORECASE),
    re.compile(r"\bmake (?:mr\d+|the agent|that agent|an agent|a child)\b", re.IGNORECASE),
    re.compile(r"\bdelegate\b", re.IGNORECASE),
    re.compile(r"\bassign\b", re.IGNORECASE),
    re.compile(r"\bhave (?:an agent|a child|mr\d+|the agent|that agent)\b", re.IGNORECASE),
    re.compile(r"\blet (?:the agent|that agent)\b", re.IGNORECASE),
    re.compile(r"\b(?:can|could|would|will)\s+you\s+create\b", re.IGNORECASE),
    re.compile(r"\bi want you to create\b", re.IGNORECASE),
    re.compile(r"\bi want (?:an|a) (?:owner )?agent\b", re.IGNORECASE),
    re.compile(r"\bplease create\b", re.IGNORECASE),
    re.compile(r"\bset up (?:a|an) agent\b", re.IGNORECASE),
    re.compile(r"\bspin up (?:a|an) agent\b", re.IGNORECASE),
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

3. PERSISTENT DELEGATION / OWNERSHIP
   Create or reuse a persistent MR2-style child agent when the user wants:
   - a child responsible for an area or domain
   - delegation of ownership, not just execution
   - an agent to propose, create, review, or test within that domain
   - an agent that should decide when to create workflows

4. ONE-SHOT DELEGATION (RARE)  
   Delegate to a single worker only when:
   - the task is clearly a one-shot execution
   - AND workflow overhead is unnecessary
   - AND persistent ownership is unnecessary

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

If the user wants an agent to own an area, create a persistent MR2-style child instead of compiling a workflow directly.

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

Persistent ownership / orchestration example:

[DELEGATE]
{"agent": "mr2", "task": "Own tool creation for this area and decide when to create workflows", "context": "Keep responsibility for proposal, safety review, creation, and testing"}
[/DELEGATE]

One-shot worker example:

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


def _normalize_routing_text(value: str) -> str:
    """Normalize free-form user text for deterministic routing checks."""
    return " ".join(value.strip().lower().split())


def _generate_task_id() -> str:
    """Generate a unique, timestamp-prefixed task ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"task-{ts}-{short}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_text(value: Any, *, limit: int = 240) -> str:
    if not isinstance(value, str):
        return "-"
    normalized = " ".join(value.split())
    if not normalized:
        return "-"
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _truncate_grounding_message_body(
    text: str,
    *,
    limit: int = _GROUNDING_MESSAGE_BODY_LIMIT,
) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_GROUNDING_MESSAGE_TRUNCATION_SUFFIX))
    return text[:keep] + _GROUNDING_MESSAGE_TRUNCATION_SUFFIX


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
                    data = json.load(f)
                self._ensure_reference_defaults(data)
                return data
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted — reinitialise.
        data = {
            "session_id": uuid.uuid4().hex[:12],
            "started_at": _now_iso(),
            "claude_session_id": None,
            "tasks": {},
            "decisions": [],
            "agent_pids": [],
            "conversation": [],
            "pending_workflow": None,
        }
        self._ensure_reference_defaults(data)
        return data

    @staticmethod
    def _ensure_reference_defaults(state: dict[str, Any]) -> None:
        state.setdefault("last_created_agent_id", None)
        state.setdefault("last_referenced_agent_id", None)
        state.setdefault("last_created_workflow_id", None)
        state.setdefault("last_referenced_workflow_id", None)
        aliases = state.setdefault("reference_aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}
            state["reference_aliases"] = aliases
        agents = aliases.get("agents")
        workflows = aliases.get("workflows")
        aliases["agents"] = dict(agents) if isinstance(agents, dict) else {}
        aliases["workflows"] = dict(workflows) if isinstance(workflows, dict) else {}

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

    def set_reference_state(self, key: str, value: Optional[str]) -> None:
        self._state[key] = value
        self.save()

    def set_reference_alias(self, kind: str, alias: str, target_id: str) -> None:
        normalized_alias = alias.strip().lower()
        if not normalized_alias:
            return
        aliases = self._state.setdefault("reference_aliases", {})
        bucket = aliases.setdefault(kind, {})
        bucket[normalized_alias] = target_id
        self.save()

    @property
    def reference_aliases(self) -> dict[str, dict[str, str]]:
        aliases = self._state.get("reference_aliases", {})
        return {
            "agents": dict(aliases.get("agents", {})),
            "workflows": dict(aliases.get("workflows", {})),
        }

    @property
    def last_created_agent_id(self) -> Optional[str]:
        return self._state.get("last_created_agent_id")

    @property
    def last_referenced_agent_id(self) -> Optional[str]:
        return self._state.get("last_referenced_agent_id")

    @property
    def last_created_workflow_id(self) -> Optional[str]:
        return self._state.get("last_created_workflow_id")

    @property
    def last_referenced_workflow_id(self) -> Optional[str]:
        return self._state.get("last_referenced_workflow_id")

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
        scoped_agent_store: Optional[PersistentAgentStore] = None,
        message_store: Optional[MessageStore] = None,
        workflow_runner: Optional[Runner] = None,
        workflow_concurrency: int = 4,
        workflow_auto_tick: bool = True,
        workflow_compiler: Optional[Callable[[str, str], str]] = None,
        workflow_authoring_backend: str = "local",
        workflow_compiler_client: Optional[WorkflowCompilerClient] = None,
        workflow_authoring_service: Optional[WorkflowAuthoringService] = None,
        inbox_auto_triage: bool = True,
        inbox_triage_interval_s: float = 30.0,
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
        self._scoped_agents = scoped_agent_store or PersistentAgentStore(
            root=self._workflow_store.root.parent / "agents"
        )
        self._message_store = message_store or MessageStore(
            root=self._workflow_store.root.parent / "messages",
            scoped_agent_store=self._scoped_agents,
        )
        self._approval_store = CapabilityApprovalStore(
            self._workflow_store.root.parent / "capability_approvals"
        )
        self._event_log = EventLog(self._workflow_store.root.parent / "events")
        self._root_agent_id = self._scoped_agents.root_agent_id
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
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
            workspace_root=Path.cwd(),
        )
        self._workflow_authoring = workflow_authoring_service or WorkflowAuthoringService(
            self._scheduler,
            self._workflow_store,
            compiler=workflow_compiler if workflow_authoring_backend == "compiler_agent" else (workflow_compiler or self._run_workflow_compiler),
            authoring_backend=workflow_authoring_backend,
            workflow_compiler_client=workflow_compiler_client,
        )

        self._inbox_triage_interval_s = inbox_triage_interval_s
        self._inbox_stop = threading.Event()
        self._inbox_thread: Optional[threading.Thread] = None
        if inbox_auto_triage:
            self._inbox_thread = threading.Thread(
                target=self._run_inbox_loop,
                name="inbox-triage",
                daemon=True,
            )
            self._inbox_thread.start()

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

    def _turn_artifacts_dir(self) -> Path:
        path = self._state._path.parent / "mr1_turns"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _new_turn_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return f"turn-{timestamp}-{uuid.uuid4().hex[:6]}"

    def _write_turn_artifact(
        self,
        *,
        turn_id: str,
        user_input: str,
        route: str,
        runtime_grounding: dict[str, Any],
        resolved_references: dict[str, Any],
        ambiguities: list[dict[str, Any]],
        brain_prompt: Optional[str] = None,
        brain_response: Optional[str] = None,
        full_payload: Optional[str] = None,
    ) -> Path:
        payload = {
            "turn_id": turn_id,
            "timestamp": _now_iso(),
            "user_input": user_input,
            "route": route,
            "resolved_references": resolved_references,
            "ambiguities": ambiguities,
            "runtime_grounding": runtime_grounding,
            "brain_prompt": brain_prompt,
            "brain_response": brain_response,
            "full_payload": full_payload,
        }
        path = self._turn_artifacts_dir() / f"{turn_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp.replace(path)
        return path

    def _finalize_turn_response(
        self,
        text: str,
        *,
        turn_id: str,
        route: str,
        runtime_grounding: dict[str, Any],
        resolved_references: dict[str, Any],
        ambiguities: list[dict[str, Any]],
        kind: str = "message",
        brain_prompt: Optional[str] = None,
        brain_response: Optional[str] = None,
        full_payload: Optional[str] = None,
    ) -> str:
        self._write_turn_artifact(
            turn_id=turn_id,
            user_input=self._state.conversation[-1]["text"] if self._state.conversation else "",
            route=route,
            runtime_grounding=runtime_grounding,
            resolved_references=resolved_references,
            ambiguities=ambiguities,
            brain_prompt=brain_prompt,
            brain_response=brain_response,
            full_payload=full_payload,
        )
        return self._record_local_response(text, kind=kind)

    def _remember_created_agent(self, agent) -> None:
        self._state.set_reference_state("last_created_agent_id", agent.agent_id)
        self._state.set_reference_alias("agents", agent.title, agent.agent_id)

    def _remember_referenced_agent(self, agent) -> None:
        self._state.set_reference_state("last_referenced_agent_id", agent.agent_id)
        self._state.set_reference_alias("agents", agent.title, agent.agent_id)

    def _remember_created_workflow(self, workflow_id: str, *, title: Optional[str] = None) -> None:
        self._state.set_reference_state("last_created_workflow_id", workflow_id)
        self._state.set_reference_state("last_referenced_workflow_id", workflow_id)
        if title:
            self._state.set_reference_alias("workflows", title, workflow_id)

    def _remember_referenced_workflow(self, workflow_id: str, *, title: Optional[str] = None) -> None:
        self._state.set_reference_state("last_referenced_workflow_id", workflow_id)
        if title:
            self._state.set_reference_alias("workflows", title, workflow_id)

    def _prioritize_items(
        self,
        items: list[dict[str, Any]],
        *,
        id_field: str,
        pinned_ids: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        pinned_set = {item for item in pinned_ids if item}
        prioritized = [item for item in items if item.get(id_field) in pinned_set]
        remaining = [item for item in items if item.get(id_field) not in pinned_set]
        return (prioritized + remaining)[:limit]

    def _grounding_agents(
        self,
        *,
        limit: int = _GROUNDING_AGENT_LIMIT,
        pinned_agent_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for agent in self._scoped_agents.list_visible_agents(self._root_agent_id):
            unread_count = sum(
                1
                for message in self._message_store.list_inbox(agent.agent_id)
                if message.status == "unread"
            )
            payload.append({
                "agent_id": agent.agent_id,
                "title": agent.title,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "run_status": agent.run_status,
                "mission_summary": _compact_text(agent.mission, limit=240),
                "last_action_summary": _compact_text(json.dumps(agent.last_action, sort_keys=True) if agent.last_action else "", limit=240),
                "latest_run": dict(agent.last_run) if isinstance(agent.last_run, dict) else None,
                "unread_inbox_count": unread_count,
                "parent_agent_id": agent.parent_agent_id,
                "scope_roots_summary": [
                    _compact_text(path, limit=120)
                    for path in list(agent.scope_roots or [])[:5]
                ],
                "created_at": agent.created_at,
            })
        payload.sort(key=lambda item: (item["created_at"], item["agent_id"]), reverse=True)
        pinned = pinned_agent_ids or [
            self._state.last_referenced_agent_id,
            self._state.last_created_agent_id,
        ]
        return self._prioritize_items(payload, id_field="agent_id", pinned_ids=pinned, limit=limit)

    def _grounding_workflows(
        self,
        *,
        limit: int = _GROUNDING_WORKFLOW_LIMIT,
        pinned_workflow_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for workflow in self._workflow_store.list_workflows():
            tasks = list(workflow.tasks.values())
            tasks.sort(key=lambda item: item.created_at)
            payload.append({
                "workflow_id": workflow.workflow_id,
                "title": workflow.title,
                "status": workflow.status.value,
                "owner_agent_id": workflow.owner_agent_id,
                "created_at": workflow.created_at,
                "recent_task_status_summary": [
                    {
                        "label": task.label,
                        "status": task.status.value,
                        "summary": task.result_summary,
                    }
                    for task in tasks[:5]
                ],
                "memory_refs_used": list(workflow.metadata.get("memory_refs_used", []))
                if isinstance(workflow.metadata, dict) else [],
            })
        payload.sort(key=lambda item: (item["created_at"], item["workflow_id"]), reverse=True)
        pinned = pinned_workflow_ids or [
            self._state.last_referenced_workflow_id,
            self._state.last_created_workflow_id,
        ]
        return self._prioritize_items(payload, id_field="workflow_id", pinned_ids=pinned, limit=limit)

    def _grounding_messages(
        self,
        *,
        limit: int = _GROUNDING_MESSAGE_LIMIT,
        pinned_message_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        pinned_ids = [item for item in (pinned_message_ids or []) if item]
        messages = [
            message
            for message in self._message_store.list_inbox(self._root_agent_id)
            if message.status == "unread"
        ]
        seen = {message.message_id for message in messages}
        for message_id in pinned_ids:
            message = self._message_store.get_message(message_id)
            if message is None or message.message_id in seen:
                continue
            messages.append(message)
            seen.add(message.message_id)
        payload = [
            {
                "message_id": message.message_id,
                "from_agent_id": message.from_agent_id,
                "to_agent_id": message.to_agent_id,
                "kind": message.kind,
                "subject": message.subject,
                "status": message.status,
                "created_at": message.created_at,
                "body": _truncate_grounding_message_body(message.body),
            }
            for message in messages
        ]
        payload.sort(key=lambda item: (item["created_at"], item["message_id"]), reverse=True)
        return self._prioritize_items(payload, id_field="message_id", pinned_ids=pinned_ids, limit=limit)

    def _grounding_approvals(self, *, limit: int = _GROUNDING_APPROVAL_LIMIT) -> list[dict[str, Any]]:
        payload = []
        for approval in self._approval_store.list_requests():
            if approval.status != "pending":
                continue
            payload.append({
                "approval_request_id": approval.approval_request_id,
                "requesting_actor_id": approval.requesting_actor_id,
                "capability_name": approval.capability_name,
                "risk_score": approval.risk_score,
                "designated_approver_id": approval.designated_approver_id,
                "status": approval.status,
                "created_at": approval.created_at,
            })
        payload.sort(key=lambda item: (item["created_at"], item["approval_request_id"]), reverse=True)
        return payload[:limit]

    def _grounding_events(self, *, limit: int = _GROUNDING_EVENT_LIMIT) -> list[dict[str, Any]]:
        events = self._event_log.recent_activity(limit=limit)
        return [
            {
                "event_index": event.event_index,
                "event_type": event.event_type,
                "actor_id": event.actor_id,
                "target_id": event.target_id,
                "status": event.status,
                "summary": event.summary,
            }
            for event in events
        ]

    def build_runtime_grounding(
        self,
        *,
        resolved_references: Optional[dict[str, Any]] = None,
        ambiguities: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        resolved = dict(resolved_references or {})
        pinned_agent_ids = [
            item.get("id")
            for item in resolved.values()
            if item.get("kind") == "agent"
        ]
        pinned_workflow_ids = [
            item.get("id")
            for item in resolved.values()
            if item.get("kind") == "workflow"
        ]
        pinned_message_ids = [
            item.get("id")
            for item in resolved.values()
            if item.get("kind") == "message"
        ]
        return {
            "agents": self._grounding_agents(pinned_agent_ids=pinned_agent_ids),
            "workflows": self._grounding_workflows(pinned_workflow_ids=pinned_workflow_ids),
            "messages": self._grounding_messages(pinned_message_ids=pinned_message_ids),
            "approvals": self._grounding_approvals(),
            "events": self._grounding_events(),
            "resolved_references": resolved,
            "ambiguities": list(ambiguities or []),
        }

    def _format_runtime_grounding_block(self, runtime_grounding: dict[str, Any]) -> str:
        return "\n".join([
            "RUNTIME STATE IS SOURCE OF TRUTH.",
            "If this conflicts with remembered conversation context, trust runtime state.",
            'Resolve references such as "that MR2" using the runtime state first.',
            "",
            "=== RUNTIME GROUNDING ===",
            json.dumps(runtime_grounding, indent=2, sort_keys=True),
            "=== END RUNTIME GROUNDING ===",
        ])

    def _agent_reference_payload(self, agent) -> dict[str, Any]:
        return {
            "kind": "agent",
            "id": agent.agent_id,
            "title": agent.title,
        }

    def _workflow_reference_payload(self, workflow) -> dict[str, Any]:
        return {
            "kind": "workflow",
            "id": workflow.workflow_id,
            "title": workflow.title,
        }

    def _message_reference_payload(self, message) -> dict[str, Any]:
        return {
            "kind": "message",
            "id": message.message_id,
            "subject": message.subject,
            "from_agent_id": message.from_agent_id,
            "to_agent_id": message.to_agent_id,
        }

    @staticmethod
    def _is_mr_style_agent_title(title: str) -> bool:
        return bool(re.fullmatch(r"mr\d+", title, flags=re.IGNORECASE))

    @staticmethod
    def _bulk_agent_action(normalized: str) -> str:
        for action in _BULK_AGENT_ACTIONS:
            if re.search(rf"\b{action}\b", normalized):
                return action
        return "manage"

    def _agent_title_reference_kind(self, normalized: str, title: str) -> Optional[str]:
        escaped_title = re.escape(title)
        if not re.search(rf"\b{escaped_title}\b", normalized):
            return None

        bulk_patterns = (
            rf"\ball\s+{escaped_title}\s+agents\b",
            rf"\b(?:all\s+(?:of\s+)?)?(?:those|these|the)\s+{escaped_title}\s+agents\b",
            rf"\bevery\s+agent\s+named\s+{escaped_title}\b",
        )
        if any(re.search(pattern, normalized) for pattern in bulk_patterns):
            return "bulk"
        if (
            re.search(rf"\b{escaped_title}\s+agents\b", normalized)
            and any(re.search(rf"\b{action}\b", normalized) for action in _BULK_AGENT_ACTIONS)
        ):
            return "bulk"

        single_patterns = (
            rf"\bwhat\s+is\s+(?:that\s+|the\s+)?{escaped_title}\s+doing\b",
            rf"\b(?:run|kill|terminate|resume|message|ask)\s+(?:that\s+|the\s+)?{escaped_title}\b",
            rf"\b(?:status|inbox|approval)\s+(?:for|of)?\s*(?:that\s+|the\s+)?{escaped_title}\b",
            rf"\bworkflow\s+owner\s+(?:for|of)?\s*(?:that\s+|the\s+)?{escaped_title}\b",
            rf"\b(?:that|the)\s+{escaped_title}\s+(?:agent|child)\b",
            rf"\b(?:agent|child)\s+(?:named\s+)?{escaped_title}\b",
            rf"\b{escaped_title}\s+(?:agent|child)\b",
        )
        if any(re.search(pattern, normalized) for pattern in single_patterns):
            return "single"
        if self._is_mr_style_agent_title(title):
            return "single"
        return None

    def resolve_runtime_references(
        self,
        user_input: str,
        runtime_state: dict[str, Any],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        ambiguities: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        bulk_targets: list[dict[str, Any]] = []
        normalized = _normalize_routing_text(user_input)
        visible_agents = self._scoped_agents.list_visible_agents(self._root_agent_id)
        agents_by_id = {agent.agent_id: agent for agent in visible_agents}
        workflows = self._workflow_store.list_workflows()
        workflows_by_id = {workflow.workflow_id: workflow for workflow in workflows}

        for agent_id in sorted(set(_AGENT_ID_PATTERN.findall(user_input))):
            agent = agents_by_id.get(agent_id)
            if agent is None:
                missing.append({"reference": agent_id, "kind": "agent"})
                continue
            resolved[agent_id] = self._agent_reference_payload(agent)

        workflow_ids = set(re.findall(r"\bwf-\d{8}T\d{6}-[0-9a-f]{6}\b", user_input))
        explicit_workflow_id = self._workflow_authoring.extract_workflow_id(user_input)
        if explicit_workflow_id:
            workflow_ids.add(explicit_workflow_id)
        for workflow_id in sorted(workflow_ids):
            workflow = workflows_by_id.get(workflow_id)
            if workflow is None:
                missing.append({"reference": workflow_id, "kind": "workflow"})
                continue
            resolved[workflow_id] = self._workflow_reference_payload(workflow)

        for message_id in sorted(set(_MESSAGE_ID_PATTERN.findall(user_input))):
            message = self._message_store.get_message(message_id)
            if message is None:
                missing.append({"reference": message_id, "kind": "message"})
                continue
            resolved[message_id] = self._message_reference_payload(message)

        title_matches: dict[str, list[Any]] = {}
        for agent in visible_agents:
            title_matches.setdefault(agent.title.lower(), []).append(agent)

        for title, candidates in title_matches.items():
            plain_match = re.search(rf"\b{re.escape(title)}\b", normalized)
            that_match = re.search(rf"\bthat\s+{re.escape(title)}\b", normalized)
            if not plain_match and not that_match:
                continue
            reference_kind = self._agent_title_reference_kind(normalized, title)
            if reference_kind is None:
                continue
            if reference_kind == "bulk":
                bulk_targets.append({
                    "reference": f"{candidates[0].title} agents",
                    "kind": "agent",
                    "action": self._bulk_agent_action(normalized),
                    "candidates": [self._agent_reference_payload(item) for item in candidates],
                })
                continue
            if len(candidates) == 1:
                source = f"that {candidates[0].title}" if that_match else candidates[0].title
                resolved[source] = self._agent_reference_payload(candidates[0])
                continue
            pinned_id = None
            if that_match:
                pinned_id = self._state.last_referenced_agent_id or self._state.last_created_agent_id
            if pinned_id is not None:
                pinned_candidate = next((item for item in candidates if item.agent_id == pinned_id), None)
                if pinned_candidate is not None:
                    resolved[f"that {pinned_candidate.title}"] = self._agent_reference_payload(pinned_candidate)
                    continue
            ambiguities.append({
                "reference": f"that {candidates[0].title}" if that_match else candidates[0].title,
                "kind": "agent",
                "candidates": [self._agent_reference_payload(item) for item in candidates],
            })

        if "that agent" in normalized and "that agent" not in resolved:
            target_id = self._state.last_referenced_agent_id or self._state.last_created_agent_id
            if target_id and target_id in agents_by_id:
                resolved["that agent"] = self._agent_reference_payload(agents_by_id[target_id])
            else:
                missing.append({"reference": "that agent", "kind": "agent"})

        workflow_alias = "the workflow from earlier"
        if workflow_alias in normalized and workflow_alias not in resolved:
            workflow_id = self._state.last_referenced_workflow_id or self._state.last_created_workflow_id
            workflow = workflows_by_id.get(workflow_id or "")
            if workflow is not None:
                resolved[workflow_alias] = self._workflow_reference_payload(workflow)
            else:
                missing.append({"reference": workflow_alias, "kind": "workflow"})

        for alias, agent_id in self._state.reference_aliases.get("agents", {}).items():
            if (
                alias in normalized
                and alias not in resolved
                and agent_id in agents_by_id
                and self._agent_title_reference_kind(normalized, alias) == "single"
            ):
                resolved[alias] = self._agent_reference_payload(agents_by_id[agent_id])
        for alias, workflow_id in self._state.reference_aliases.get("workflows", {}).items():
            workflow = workflows_by_id.get(workflow_id)
            if alias in normalized and alias not in resolved and workflow is not None:
                resolved[alias] = self._workflow_reference_payload(workflow)

        return {
            "resolved_references": resolved,
            "ambiguities": ambiguities,
            "missing": missing,
            "bulk_targets": bulk_targets,
        }

    def _format_reference_ambiguity(self, ambiguities: list[dict[str, Any]]) -> str:
        first = ambiguities[0]
        candidates = ", ".join(
            f"{item.get('title') or item.get('id')} ({item['id']})"
            for item in first.get("candidates", [])
        )
        return f"Ambiguous reference '{first['reference']}'. Clarify one of: {candidates}"

    def _format_bulk_agent_target_clarification(
        self,
        bulk_targets: list[dict[str, Any]],
    ) -> str:
        first = bulk_targets[0]
        candidates = first.get("candidates", [])
        candidate_list = ", ".join(
            f"{item.get('title') or item.get('id')} ({item['id']})"
            for item in candidates
        )
        action = first.get("action") or "manage"
        if action in {"kill", "terminate"}:
            return (
                f"Bulk agent terminate request matches {len(candidates)} agent(s): {candidate_list}. "
                "Confirm the exact agent_ids you want me to terminate with /agent kill <ag-id>."
            )
        return (
            f"Bulk agent reference '{first.get('reference', 'agents')}' matches {len(candidates)} agent(s): "
            f"{candidate_list}. Clarify the exact agent_ids you want me to {action}."
        )

    def _resolved_workflow_target_id(self, resolved_references: dict[str, Any]) -> Optional[str]:
        for payload in resolved_references.values():
            if payload.get("kind") == "workflow":
                return payload.get("id")
        return None

    def _explicit_workflow_ids(
        self,
        user_input: str,
        resolved_references: dict[str, Any],
    ) -> list[str]:
        workflow_ids = {
            payload["id"]
            for payload in resolved_references.values()
            if payload.get("kind") == "workflow" and payload.get("id")
        }
        workflow_ids.update(_WORKFLOW_ID_PATTERN.findall(user_input))
        explicit_workflow_id = self._workflow_authoring.extract_workflow_id(user_input)
        if explicit_workflow_id:
            workflow_ids.add(explicit_workflow_id)
        return sorted(workflow_ids)

    def _explicit_task_ids(self, user_input: str) -> list[str]:
        return sorted(set(_TASK_ID_PATTERN.findall(user_input)))

    def _is_runtime_inspection_request(
        self,
        user_input: str,
        resolved_references: dict[str, Any],
    ) -> bool:
        normalized = _normalize_routing_text(user_input)
        if not normalized:
            return False
        has_intent = any(phrase in normalized for phrase in _WORKFLOW_INSPECTION_INTENT_PHRASES)
        has_intent = has_intent or bool(
            re.search(r"\bdid\b.*\bfinish(?:\s+running)?\b", normalized)
        )
        has_intent = has_intent or bool(
            re.search(r"\bwhy\b.*\bfail(?:ed)?\b", normalized)
        )
        suppresses_authoring = any(
            phrase in normalized for phrase in _WORKFLOW_AUTHORING_SUPPRESSION_PHRASES
        )
        if not has_intent and not suppresses_authoring:
            return False
        return bool(
            self._explicit_workflow_ids(user_input, resolved_references)
            or self._explicit_task_ids(user_input)
            or any(alias in normalized for alias in _WORKFLOW_REFERENCE_ALIASES)
            or (_has_workflow_pronoun_reference(normalized) and self._state.last_referenced_workflow_id)
        )

    def _is_workflow_findings_request(self, user_input: str) -> bool:
        normalized = _normalize_routing_text(user_input)
        if not normalized:
            return False
        return any(phrase in normalized for phrase in _WORKFLOW_FINDINGS_INTENT_PHRASES)

    def _approval_request_id_from_result(self, payload: Optional[dict[str, Any]]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        direct_value = payload.get("approval_request_id")
        if isinstance(direct_value, str) and direct_value:
            return direct_value
        data = payload.get("data")
        if isinstance(data, dict):
            nested_value = data.get("approval_request_id")
            if isinstance(nested_value, str) and nested_value:
                return nested_value
        return None

    def _task_summary_text(
        self,
        workflow_id: str,
        task_id: str,
    ) -> str:
        workflow = self._workflow_store.load_workflow(workflow_id)
        if workflow is None:
            return f"workflow not found: {workflow_id}"
        task = workflow.tasks.get(task_id)
        if task is None:
            return f"task not found: {task_id}"
        result_payload = self._workflow_store.read_result(workflow_id, task_id)
        output = self._workflow_store.load_task_output(workflow_id, task_id)
        approval_request_id = self._approval_request_id_from_result(result_payload)
        failure_type = None
        error_text = None
        summary_text = None
        if isinstance(result_payload, dict):
            failure_type = result_payload.get("failure_type") or result_payload.get("error_type")
            error_text = result_payload.get("error") or task.last_error
            summary_text = result_payload.get("summary")
        if output is not None:
            summary_text = output.summary or summary_text
        summary_text = summary_text or task.result_summary
        self._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)

        lines = [
            f"task_id:   {task.task_id}",
            f"label:     {task.label}",
            f"title:     {task.title}",
            f"workflow:  {workflow.workflow_id} ({workflow.title})",
            f"status:    {task.status.value}",
        ]
        if failure_type == "semantic_mismatch" or task.last_error_type == "semantic_mismatch":
            lines.append("semantic_validation: failed")
        if failure_type:
            lines.append(f"failure:   {failure_type}")
        if error_text:
            lines.append(f"error:     {_compact_text(error_text)}")
        if summary_text:
            lines.append(f"summary:   {_compact_text(summary_text)}")
        if approval_request_id:
            lines.append(f"approval:  {approval_request_id}")
        if task.log_stdout_path:
            lines.append(f"stdout:    {task.log_stdout_path}")
        if task.log_stderr_path:
            lines.append(f"stderr:    {task.log_stderr_path}")
        if task.result_path:
            lines.append(f"result:    {task.result_path}")
        if task.output_path:
            lines.append(f"output:    {task.output_path}")
        suggestions = []
        if task.status in {
            TaskStatus.FAILED,
            TaskStatus.TIMED_OUT,
            TaskStatus.CANCELLED,
            TaskStatus.BLOCKED,
        }:
            suggestions.append(f"/workflow rerun {workflow.workflow_id} {task.task_id}")
        if approval_request_id:
            suggestions.append(f"/approvals show {approval_request_id}")
        if suggestions:
            lines.extend(["", "suggested_next:"])
            lines.extend(f"- {item}" for item in suggestions)
        return "\n".join(lines)

    def _workflow_summary_text(self, workflow_id: str) -> str:
        workflow = self._workflow_store.load_workflow(workflow_id)
        if workflow is None:
            return f"workflow not found: {workflow_id}"
        self._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)
        task_lines: list[str] = []
        result_lines: list[str] = []
        failed_lines: list[str] = []
        approval_lines: list[str] = []
        semantic_lines: list[str] = []
        first_retry_task_id: Optional[str] = None
        first_approval_request_id: Optional[str] = None

        for label, task_id in workflow.label_to_task_id.items():
            task = workflow.tasks.get(task_id)
            if task is None:
                continue
            result_payload = self._workflow_store.read_result(workflow.workflow_id, task.task_id)
            output = self._workflow_store.load_task_output(workflow.workflow_id, task.task_id)
            summary_text = None
            if output is not None:
                summary_text = output.summary
            if summary_text is None and isinstance(result_payload, dict):
                summary_text = result_payload.get("summary")
            summary_text = summary_text or task.result_summary or task.last_error or task.blocked_reason
            task_line = f"- {label} ({task.task_id}): {task.status.value}"
            if summary_text:
                task_line += f" | {_compact_text(summary_text)}"
            task_lines.append(task_line)

            if task.is_terminal() and summary_text:
                result_lines.append(f"- {label}: {_compact_text(summary_text)}")

            if task.status in {
                TaskStatus.FAILED,
                TaskStatus.TIMED_OUT,
                TaskStatus.CANCELLED,
                TaskStatus.BLOCKED,
            }:
                failure_text = None
                if isinstance(result_payload, dict):
                    failure_text = result_payload.get("failure_type") or result_payload.get("error_type")
                failure_text = failure_text or task.last_error_type or task.blocked_reason or task.last_error
                failed_lines.append(
                    f"- {label} ({task.task_id}): {task.status.value}"
                    + (f" | {_compact_text(failure_text)}" if failure_text else "")
                )
                if failure_text == "semantic_mismatch" or task.last_error_type == "semantic_mismatch":
                    semantic_reason = task.last_error or summary_text or "task completed mechanically but did not satisfy intended postcondition"
                    semantic_lines.append(
                        f"- {label} ({task.task_id}): semantic_mismatch | {_compact_text(semantic_reason)}"
                    )
                if first_retry_task_id is None:
                    first_retry_task_id = task.task_id

            approval_request_id = self._approval_request_id_from_result(result_payload)
            if approval_request_id:
                approval_reason = None
                if isinstance(result_payload, dict):
                    decision_payload = result_payload.get("data", {}).get("decision")
                    if isinstance(decision_payload, dict):
                        approval_reason = decision_payload.get("reason")
                approval_lines.append(
                    f"- {label} ({task.task_id}): {approval_request_id}"
                    + (f" | {_compact_text(approval_reason)}" if approval_reason else "")
                )
                if first_approval_request_id is None:
                    first_approval_request_id = approval_request_id

        lines = [
            f"workflow_id: {workflow.workflow_id}",
            f"title:       {workflow.title}",
            f"status:      {workflow.status.value}",
            f"tasks:       {len(workflow.tasks)}",
        ]
        if workflow.finished_at:
            lines.append(f"finished:    {workflow.finished_at}")
        if semantic_lines:
            lines.extend(["", "semantic_validation: FAILED"])
            lines.extend(semantic_lines)
        lines.extend(["", "task_statuses:"])
        lines.extend(task_lines or ["- none"])
        if failed_lines:
            lines.extend(["", "failed_or_blocked:"])
            lines.extend(failed_lines)
        if result_lines:
            lines.extend(["", "result_summaries:"])
            lines.extend(result_lines[:8])
        if approval_lines:
            lines.extend(["", "approval_blockers:"])
            lines.extend(approval_lines)
        suggestions = [f"/workflow {workflow.workflow_id}", f"/events {workflow.workflow_id}"]
        if first_retry_task_id:
            suggestions.append(f"/workflow rerun {workflow.workflow_id} {first_retry_task_id}")
        if first_approval_request_id:
            suggestions.append(f"/approvals show {first_approval_request_id}")
        lines.extend(["", "suggested_next:"])
        lines.extend(f"- {item}" for item in suggestions)
        return "\n".join(lines)

    def _task_full_result_text(self, workflow_id: str, task_id: str) -> Optional[str]:
        output = self._workflow_store.load_task_output(workflow_id, task_id)
        if output is not None and output.text.strip():
            return output.text.strip()
        result_payload = self._workflow_store.read_result(workflow_id, task_id)
        if isinstance(result_payload, dict):
            text_value = result_payload.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value.strip()
            body_value = result_payload.get("body")
            if isinstance(body_value, str) and body_value.strip():
                return body_value.strip()
        return None

    def _findings_candidate_sort_key(
        self,
        workflow_id: str,
        task,
    ) -> tuple[int, int, str]:
        label_text = f"{task.label} {task.title}".lower()
        priority = 1
        for index, token in enumerate(_FINDINGS_PREFERRED_LABEL_TOKENS):
            if token in label_text:
                priority = 0
                return (priority, index, task.task_id)
        has_full_text = 0 if self._task_full_result_text(workflow_id, task.task_id) else 1
        return (priority, has_full_text, task.task_id)

    def _select_workflow_findings_task(self, workflow) -> Optional[Any]:
        succeeded_terminal = [
            task
            for task in workflow.tasks.values()
            if task.status == TaskStatus.SUCCEEDED and task.is_terminal()
        ]
        if not succeeded_terminal:
            return None
        candidates = sorted(
            succeeded_terminal,
            key=lambda task: self._findings_candidate_sort_key(workflow.workflow_id, task),
        )
        return candidates[0] if candidates else None

    def _workflow_findings_text(self, workflow_id: str) -> str:
        workflow = self._workflow_store.load_workflow(workflow_id)
        if workflow is None:
            return f"workflow not found: {workflow_id}"
        self._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)
        semantic_failures = [
            task
            for task in workflow.tasks.values()
            if task.last_error_type == "semantic_mismatch"
        ]
        if semantic_failures:
            lines = [
                "semantic_validation: FAILED",
                "This workflow completed mechanically but did not satisfy its intended postcondition.",
                "",
                f"workflow_id: {workflow.workflow_id}",
                f"title:       {workflow.title}",
                f"status:      {workflow.status.value}",
            ]
            if workflow.finished_at:
                lines.append(f"finished:    {workflow.finished_at}")
            lines.extend(["", "semantic_mismatches:"])
            for task in semantic_failures:
                lines.append(
                    f"- {task.label} ({task.task_id}): {_compact_text(task.last_error or task.result_summary or 'semantic_mismatch')}"
                )
            return "\n".join(lines)
        findings_task = self._select_workflow_findings_task(workflow)
        findings_text = None
        findings_summary = None
        findings_label = None
        if findings_task is not None:
            findings_text = self._task_full_result_text(workflow.workflow_id, findings_task.task_id)
            findings_summary = findings_task.result_summary
            findings_label = findings_task.label

        lines: list[str] = []
        if findings_text:
            lines.extend([
                "findings:",
                findings_text,
            ])
        elif findings_summary:
            lines.extend([
                "findings:",
                findings_summary,
            ])
        else:
            lines.append("findings: No final findings text is available for this workflow yet.")

        status_lines = [
            "",
            f"workflow_id: {workflow.workflow_id}",
            f"title:       {workflow.title}",
            f"status:      {workflow.status.value}",
        ]
        if findings_task is not None:
            status_lines.append(
                f"findings_task: {findings_label} ({findings_task.task_id})"
            )
        if workflow.finished_at:
            status_lines.append(f"finished:    {workflow.finished_at}")
        lines.extend(status_lines)
        return "\n".join(lines)

    def _maybe_route_runtime_inspection(
        self,
        user_input: str,
        *,
        turn_id: str,
        runtime_grounding: dict[str, Any],
        resolved_references: dict[str, Any],
        ambiguities: list[dict[str, Any]],
    ) -> Optional[str]:
        if not self._is_runtime_inspection_request(user_input, resolved_references):
            return None
        task_ids = self._explicit_task_ids(user_input)
        workflow_ids = self._explicit_workflow_ids(user_input, resolved_references)
        normalized = _normalize_routing_text(user_input)

        if len(task_ids) > 1:
            return self._finalize_turn_response(
                "Clarify which task you want me to inspect. Provide one task_id like tk-....",
                turn_id=turn_id,
                route="inspect_task_clarify",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )
        if len(workflow_ids) > 1:
            return self._finalize_turn_response(
                "Clarify which workflow you want me to inspect. Provide one workflow_id like wf-....",
                turn_id=turn_id,
                route="inspect_workflow_clarify",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )
        if task_ids:
            task_id = task_ids[0]
            workflow, task = workflow_cli._find_workflow_for_task(self._workflow_store, task_id)
            if workflow is None or task is None:
                return self._finalize_turn_response(
                    f"task not found: {task_id}",
                    turn_id=turn_id,
                    route="inspect_task",
                    runtime_grounding=runtime_grounding,
                    resolved_references=resolved_references,
                    ambiguities=ambiguities,
                )
            return self._finalize_turn_response(
                self._task_summary_text(workflow.workflow_id, task.task_id),
                turn_id=turn_id,
                route="inspect_task",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )
        if workflow_ids:
            if self._is_workflow_findings_request(user_input):
                return self._finalize_turn_response(
                    self._workflow_findings_text(workflow_ids[0]),
                    turn_id=turn_id,
                    route="inspect_workflow_findings",
                    runtime_grounding=runtime_grounding,
                    resolved_references=resolved_references,
                    ambiguities=ambiguities,
                )
            return self._finalize_turn_response(
                self._workflow_summary_text(workflow_ids[0]),
                turn_id=turn_id,
                route="inspect_workflow",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )
        if any(alias in normalized for alias in _WORKFLOW_REFERENCE_ALIASES):
            remembered_workflow_id = (
                self._state.last_referenced_workflow_id
                or self._state.last_created_workflow_id
            )
            if remembered_workflow_id:
                if self._is_workflow_findings_request(user_input):
                    return self._finalize_turn_response(
                        self._workflow_findings_text(remembered_workflow_id),
                        turn_id=turn_id,
                        route="inspect_workflow_findings",
                        runtime_grounding=runtime_grounding,
                        resolved_references=resolved_references,
                        ambiguities=ambiguities,
                    )
                return self._finalize_turn_response(
                    self._workflow_summary_text(remembered_workflow_id),
                    turn_id=turn_id,
                    route="inspect_workflow",
                    runtime_grounding=runtime_grounding,
                    resolved_references=resolved_references,
                    ambiguities=ambiguities,
                )
            return self._finalize_turn_response(
                "Which workflow do you mean? Provide a workflow_id like wf-....",
                turn_id=turn_id,
                route="inspect_workflow_clarify",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )
        if _has_workflow_pronoun_reference(normalized):
            remembered_workflow_id = (
                self._state.last_referenced_workflow_id
                or self._state.last_created_workflow_id
            )
            if remembered_workflow_id:
                if self._is_workflow_findings_request(user_input):
                    return self._finalize_turn_response(
                        self._workflow_findings_text(remembered_workflow_id),
                        turn_id=turn_id,
                        route="inspect_workflow_findings",
                        runtime_grounding=runtime_grounding,
                        resolved_references=resolved_references,
                        ambiguities=ambiguities,
                    )
                return self._finalize_turn_response(
                    self._workflow_summary_text(remembered_workflow_id),
                    turn_id=turn_id,
                    route="inspect_workflow",
                    runtime_grounding=runtime_grounding,
                    resolved_references=resolved_references,
                    ambiguities=ambiguities,
                )
        return None

    def _extract_message_reply_body(self, user_input: str, message_id: str) -> str:
        text = re.sub(re.escape(message_id), "", user_input, count=1)
        text = _REPLY_INTENT_PATTERN.sub("", text, count=1)
        text = re.sub(r"^\s*(?:and\s+)?provide\s+[A-Za-z0-9_-]+\s+with\s+", "", text, count=1, flags=re.IGNORECASE)
        return text.strip(" \n\t:,-")

    def _maybe_route_message_reply(
        self,
        user_input: str,
        *,
        turn_id: str,
        runtime_grounding: dict[str, Any],
        resolved_references: dict[str, Any],
        ambiguities: list[dict[str, Any]],
    ) -> Optional[str]:
        if not _REPLY_INTENT_PATTERN.search(user_input):
            return None
        message_payload = next(
            (item for item in resolved_references.values() if item.get("kind") == "message"),
            None,
        )
        if message_payload is None:
            return None
        message = self._message_store.get_message(message_payload["id"])
        if message is None:
            return None
        if message.to_agent_id != self._root_agent_id:
            return None
        child = self._scoped_agents.load_agent(message.from_agent_id)
        if child is None or not self._scoped_agents.is_visible(self._root_agent_id, child.agent_id):
            return None
        reply_body = self._extract_message_reply_body(user_input, message.message_id)
        if not reply_body:
            return self._finalize_turn_response(
                "I need the clarification text to send back to that child message.",
                turn_id=turn_id,
                route="message_reply",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )
        created = self._message_store.create_message(
            from_agent_id=self._root_agent_id,
            to_agent_id=child.agent_id,
            kind="request",
            subject=f"Re: {message.subject}",
            body=reply_body,
            workflow_id=message.workflow_id,
            task_id=message.task_id,
        )
        self._remember_referenced_agent(child)
        return self._finalize_turn_response(
            f"sent clarification to {child.agent_id} for {message.message_id}",
            turn_id=turn_id,
            route="message_reply",
            runtime_grounding=runtime_grounding,
            resolved_references=resolved_references,
            ambiguities=ambiguities,
            full_payload=json.dumps({
                "reply_to_message_id": message.message_id,
                "created_message_id": created.message_id,
                "to_agent_id": child.agent_id,
            }, indent=2, sort_keys=True),
        )

    def _maybe_route_bulk_agent_operation(
        self,
        *,
        turn_id: str,
        runtime_grounding: dict[str, Any],
        resolved_references: dict[str, Any],
        ambiguities: list[dict[str, Any]],
        bulk_targets: list[dict[str, Any]],
    ) -> Optional[str]:
        if not bulk_targets:
            return None
        return self._finalize_turn_response(
            self._format_bulk_agent_target_clarification(bulk_targets),
            turn_id=turn_id,
            route="clarify_bulk_agent_operation",
            runtime_grounding=runtime_grounding,
            resolved_references=resolved_references,
            ambiguities=ambiguities,
        )

    def _send_to_brain(self, message: str) -> str:
        if self._process is None:
            return "[MR1 ERROR] Process is not running."
        response = self._process.send(message)
        session_id = self._process.session_id
        self._state.set_claude_session_id(session_id if isinstance(session_id, str) else None)
        return response

    def _answer_directly_with_grounding(
        self,
        user_input: str,
        runtime_grounding: dict[str, Any],
    ) -> dict[str, str]:
        grounding_block = self._format_runtime_grounding_block(runtime_grounding)
        prompt = "\n\n".join([
            grounding_block,
            "Answer this request directly. Do not delegate to MR2 or Kazi.",
            f"User request:\n{user_input}",
        ])
        raw = self._send_to_brain(prompt)
        text, _ = self._parse_response(raw)
        return {
            "text": text,
            "brain_prompt": prompt,
            "brain_response": raw,
            "full_payload": prompt,
        }

    def _run_workflow_compiler(self, system_prompt: str, message: str) -> str:
        proc = MR1Process(system_prompt, self._mr1_config["model"], [])
        proc.start()
        return proc.send(message)

    def _format_authoring_preview(self, result: AuthoringResult) -> str:
        lines = [result.preview_text]
        if result.assumptions:
            lines.extend(["", "Assumptions:"])
            lines.extend(f"- {item}" for item in result.assumptions)
        if result.risks:
            lines.extend(["", "Risks:"])
            lines.extend(f"- {item}" for item in result.risks)
        if result.assumptions or result.risks or self._workflow_authoring._authoring_backend == "compiler_agent":
            lines.extend(["", f"Confidence: {result.confidence}"])
        return "\n".join(lines)

    def _is_meta_explanation_request(self, user_input: str) -> bool:
        normalized = _normalize_routing_text(user_input)
        if not normalized:
            return False
        if not any(
            token in normalized
            for token in (
                "tool",
                "tools",
                "workflow",
                "workflows",
                "agent",
                "agents",
                "child",
                "children",
                "delegate",
                "delegation",
                "owner",
                "ownership",
                "route",
                "routing",
                "mr1",
                "mr2",
            )
        ):
            return False
        return any(pattern.search(normalized) for pattern in _META_EXPLANATION_PATTERNS)

    def _is_persistent_delegation_request(self, user_input: str) -> bool:
        normalized = _normalize_routing_text(user_input)
        if not normalized:
            return False
        if self._is_meta_explanation_request(user_input):
            return False
        if not any(pattern.search(normalized) for pattern in _PERSISTENT_DELEGATION_IMPERATIVE_PATTERNS):
            return False
        if any(marker in normalized for marker in _PERSISTENT_DELEGATION_MARKERS):
            return True

        has_agent_target = any(token in normalized for token in ("mr2", "child", "agent"))
        if has_agent_target and any(
            phrase in normalized
            for phrase in ("responsible for", "owner agent", "ownership")
        ):
            return True
        if has_agent_target and re.search(r"\bown\b", normalized):
            return True
        if "delegate" in normalized and any(
            token in normalized
            for token in ("domain", "area", "ownership", "owner", "responsible")
        ):
            return True
        return False

    def _classify_turn_route(
        self,
        user_input: str,
        pending_draft: Optional[PendingWorkflowDraft],
    ) -> str:
        if self._is_meta_explanation_request(user_input):
            return "direct_answer"
        if pending_draft is None and self._is_persistent_delegation_request(user_input):
            return "persistent_delegation"
        return self._workflow_authoring.classify_request(
            user_input,
            pending_draft=pending_draft,
        )

    @staticmethod
    def _extract_requested_child_title(user_input: str) -> str:
        for pattern in _PERSISTENT_CHILD_TITLE_PATTERNS:
            match = pattern.search(user_input)
            if match:
                return match.group(1).upper()
        fallback_matches = re.findall(r"\b(MR\d+)\b", user_input, flags=re.IGNORECASE)
        for item in fallback_matches:
            normalized = item.upper()
            if normalized != "MR1":
                return normalized
        return "MR2"

    def _build_persistent_delegation_mission(self, user_input: str, *, agent_title: str) -> str:
        return "\n".join([
            f"You are a persistent {agent_title}-style child agent.",
            "Own the requested domain/responsibility instead of treating it as a one-shot execution.",
            "",
            "Parent request:",
            user_input.strip(),
            "",
            "Operating instructions:",
            "- Treat this as an ownership/delegation request.",
            "- You are already the delegated child created for this request; do not create another child agent unless the parent explicitly asks in a follow-up.",
            "- Prefer creating workflows for execution when appropriate.",
            "- Keep responsibility for proposal quality, safety review, creation, and testing.",
            "- Use workflow creation when execution should become structured work.",
            "- Escalate to MR1/user when clarification or confirmation is needed.",
        ])

    def _route_to_persistent_delegation(self, user_input: str) -> str:
        agent_title = self._extract_requested_child_title(user_input)
        try:
            agent = self._scoped_agents.create_child_agent(self._root_agent_id, agent_title)
            agent = self._scoped_agents.assign_mission(
                self._root_agent_id,
                agent.agent_id,
                self._build_persistent_delegation_mission(
                    user_input,
                    agent_title=agent.title,
                ),
            )
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        self._remember_created_agent(agent)
        self._remember_referenced_agent(agent)

        runner = MRnRunRunner(
            workflow_store=self._workflow_store,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
        )
        policy = MRnRunPolicy(
            max_steps=3,
            max_workflows_created=2,
            require_confirmation_for_workflows=True,
        )
        try:
            result = runner.run(
                agent.agent_id,
                policy,
                caller_agent_id=self._root_agent_id,
            )
        except (ValueError, AgentScopeError) as exc:
            return str(exc)

        self._state.add_decision(user_input, "spawn_persistent_mr2", agent.agent_id)
        return "\n".join([
            f"delegated to persistent agent: {agent.agent_id} ({agent.title})",
            workflow_cli._format_mrn_run_result(result),
        ])

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
          1. Decide direct answer vs persistent delegation vs workflow authoring
          2. For persistent delegation: create/run a scoped MRn child
          3. For workflow turns: compile, validate, preview, submit
          4. For direct answers: ask MR1 to answer without delegation
        """
        turn_id = self._new_turn_id()
        self._record_conversation("user", user_input)
        initial_grounding = self.build_runtime_grounding()
        resolution = self.resolve_runtime_references(user_input, initial_grounding)
        runtime_grounding = self.build_runtime_grounding(
            resolved_references=resolution["resolved_references"],
            ambiguities=resolution["ambiguities"],
        )
        resolved_references = runtime_grounding["resolved_references"]
        ambiguities = runtime_grounding["ambiguities"]
        bulk_targets = list(resolution.get("bulk_targets", []))

        for payload in resolved_references.values():
            if payload.get("kind") == "agent":
                agent = self._scoped_agents.load_agent(payload["id"])
                if agent is not None:
                    self._remember_referenced_agent(agent)
            if payload.get("kind") == "workflow":
                workflow = self._workflow_store.load_workflow(payload["id"])
                if workflow is not None:
                    self._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)

        reply_result = self._maybe_route_message_reply(
            user_input,
            turn_id=turn_id,
            runtime_grounding=runtime_grounding,
            resolved_references=resolved_references,
            ambiguities=ambiguities,
        )
        if reply_result is not None:
            return reply_result

        bulk_result = self._maybe_route_bulk_agent_operation(
            turn_id=turn_id,
            runtime_grounding=runtime_grounding,
            resolved_references=resolved_references,
            ambiguities=ambiguities,
            bulk_targets=bulk_targets,
        )
        if bulk_result is not None:
            return bulk_result

        if ambiguities:
            return self._finalize_turn_response(
                self._format_reference_ambiguity(ambiguities),
                turn_id=turn_id,
                route="clarify_reference",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )

        inspection_result = self._maybe_route_runtime_inspection(
            user_input,
            turn_id=turn_id,
            runtime_grounding=runtime_grounding,
            resolved_references=resolved_references,
            ambiguities=ambiguities,
        )
        if inspection_result is not None:
            return inspection_result

        pending = self._workflow_authoring.coerce_pending_draft(
            self._state.pending_workflow
        )
        action = self._classify_turn_route(
            user_input,
            pending,
        )

        if action == "direct_answer":
            self._state.add_decision(user_input, "direct_answer")
            answer = self._answer_directly_with_grounding(user_input, runtime_grounding)
            return self._finalize_turn_response(
                answer["text"],
                turn_id=turn_id,
                route="direct_answer",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
                brain_prompt=answer["brain_prompt"],
                brain_response=answer["brain_response"],
                full_payload=answer["full_payload"],
            )

        if action == "persistent_delegation":
            return self._finalize_turn_response(
                self._route_to_persistent_delegation(user_input),
                turn_id=turn_id,
                route="persistent_delegation",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )

        if action == "show_json_preview":
            if pending is None:
                return self._finalize_turn_response(
                    "No pending workflow draft.",
                    turn_id=turn_id,
                    route="show_json_preview",
                    runtime_grounding=runtime_grounding,
                    resolved_references=resolved_references,
                    ambiguities=ambiguities,
                )
            return self._finalize_turn_response(
                json.dumps(pending.spec, indent=2),
                turn_id=turn_id,
                route="show_json_preview",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
                kind="workflow_json",
            )

        if action == "cancel_preview":
            self._state.clear_pending_workflow()
            self._state.add_decision(user_input, "cancel_workflow_preview")
            return self._finalize_turn_response(
                "Cancelled pending workflow draft.",
                turn_id=turn_id,
                route="cancel_preview",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )

        if action == "confirm_preview":
            if pending is None:
                return self._finalize_turn_response(
                    "No pending workflow draft.",
                    turn_id=turn_id,
                    route="confirm_preview",
                    runtime_grounding=runtime_grounding,
                    resolved_references=resolved_references,
                    ambiguities=ambiguities,
                )
            result = self._workflow_authoring.submit(
                pending.spec,
                created_by=Provenance(type="agent", id="MR1"),
                caller_agent_id=self._root_agent_id,
                owner_agent_id=self._root_agent_id,
                target_workflow_id=pending.target_workflow_id,
                workflow_metadata={
                    "compiled_with_memory": bool(pending.compiled_with_memory),
                    "memory_refs_used": list(pending.memory_refs_used),
                    "memory_tools_used": list(pending.memory_tools_used),
                    "memory_context_summary": pending.memory_context_summary,
                },
            )
            self._state.clear_pending_workflow()
            self._state.add_decision(
                user_input,
                "submit_pending_workflow",
                result.workflow_id,
            )
            workflow = self._workflow_store.load_workflow(result.workflow_id)
            self._remember_created_workflow(
                result.workflow_id,
                title=workflow.title if workflow is not None else None,
            )
            return self._finalize_turn_response(
                result.message,
                turn_id=turn_id,
                route="confirm_preview",
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )

        mode = "modify" if action == "modify_workflow" else "create"
        target_workflow_id = (
            self._workflow_authoring.extract_workflow_id(user_input)
            or self._resolved_workflow_target_id(resolved_references)
        )
        baseline_spec: Optional[dict[str, Any]] = None
        if pending is not None:
            baseline_spec = pending.spec
            target_workflow_id = pending.target_workflow_id or target_workflow_id
        elif target_workflow_id:
            workflow = self._workflow_store.load_workflow(target_workflow_id)
            if workflow is None:
                return self._finalize_turn_response(
                    f"workflow not found: {target_workflow_id}",
                    turn_id=turn_id,
                    route=action,
                    runtime_grounding=runtime_grounding,
                    resolved_references=resolved_references,
                    ambiguities=ambiguities,
                )
            baseline_spec = workflow_to_spec(workflow)
            self._remember_referenced_workflow(workflow.workflow_id, title=workflow.title)

        if mode == "modify" and baseline_spec is None:
            return self._finalize_turn_response(
                self._workflow_authoring.clarify_message(
                    "missing workflow target",
                    mode=mode,
                    target_workflow_id=target_workflow_id,
                ),
                turn_id=turn_id,
                route=action,
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )

        try:
            authoring = self._workflow_authoring.author_request(
                user_input,
                caller_agent_id=self._root_agent_id,
                owner_agent_id=self._root_agent_id,
                mode=mode,
                baseline_spec=baseline_spec,
                target_workflow_id=target_workflow_id,
            )
        except (RuntimeError, json.JSONDecodeError, WorkflowSpecError, ValueError) as exc:
            return self._finalize_turn_response(
                self._workflow_authoring.clarify_message(
                    str(exc),
                    mode=mode,
                    target_workflow_id=target_workflow_id,
                ),
                turn_id=turn_id,
                route=action,
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )

        if authoring.complexity == "simple" and not authoring.needs_confirmation:
            result = self._workflow_authoring.submit(
                authoring.spec,
                created_by=Provenance(type="agent", id="MR1"),
                caller_agent_id=self._root_agent_id,
                owner_agent_id=self._root_agent_id,
                target_workflow_id=target_workflow_id,
                workflow_metadata=self._workflow_authoring.workflow_metadata_from_authoring(authoring),
            )
            self._state.clear_pending_workflow()
            self._state.add_decision(
                user_input,
                "modify_workflow" if mode == "modify" else "submit_workflow",
                result.workflow_id,
            )
            workflow = self._workflow_store.load_workflow(result.workflow_id)
            self._remember_created_workflow(
                result.workflow_id,
                title=workflow.title if workflow is not None else authoring.spec.get("title"),
            )
            return self._finalize_turn_response(
                result.message,
                turn_id=turn_id,
                route=action,
                runtime_grounding=runtime_grounding,
                resolved_references=resolved_references,
                ambiguities=ambiguities,
            )

        draft = PendingWorkflowDraft(
            original_request=user_input,
            mode=mode,
            spec=authoring.spec,
            target_workflow_id=target_workflow_id,
            preview_text=authoring.preview_text,
            assumptions=list(authoring.assumptions),
            risks=list(authoring.risks),
            needs_confirmation=authoring.needs_confirmation,
            confidence=authoring.confidence,
            complexity=authoring.complexity,
            compiled_with_memory=authoring.compiled_with_memory,
            memory_refs_used=list(authoring.memory_refs_used),
            memory_tools_used=list(authoring.memory_tools_used),
            memory_context_summary=authoring.memory_context_summary,
            memory_ref_warnings=list(authoring.memory_ref_warnings),
        )
        self._state.set_pending_workflow(draft.to_dict())
        self._state.add_decision(
            user_input,
            "preview_workflow_modification" if mode == "modify" else "preview_workflow",
            target_workflow_id,
        )
        return self._finalize_turn_response(
            self._format_authoring_preview(authoring),
            turn_id=turn_id,
            route=action,
            runtime_grounding=runtime_grounding,
            resolved_references=resolved_references,
            ambiguities=ambiguities,
            kind="workflow_preview",
        )

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

    def _running_test_agent_count(self) -> int:
        with self._test_agent_lock:
            return sum(
                1
                for record in self._test_agents.values()
                if record.process.poll() is None
            )

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
            running_synthetic = self._running_test_agent_count()
            killed = self._spawner.kill_all("user_kill")
            self.kill_test_agents()
            synthetic_killed = running_synthetic
            # Mark all running tasks as killed.
            for tid in list(self._state.active_tasks):
                self._state.complete_task(tid, "killed")
            total = killed + synthetic_killed
            if total == 0:
                return (
                    "No running spawned processes/test agents to kill. "
                    "Persistent agents are not affected. "
                    "Use /agent kill <ag-id> to terminate a persistent agent."
                )
            return (
                f"Killed {total} spawned process/test agent(s). "
                "Persistent MRn agents are not affected. "
                "Use /agent kill <ag-id> to terminate a persistent agent."
            )
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
        if cmd == "/inbox" or cmd.startswith("/inbox "):
            return self._handle_message_builtin(cmd)
        if cmd == "/outbox" or cmd.startswith("/outbox "):
            return self._handle_message_builtin(cmd)
        if cmd.startswith("/message"):
            return self._handle_message_builtin(cmd)
        if cmd == "/approvals" or cmd.startswith("/approvals "):
            return self._handle_approval_builtin(cmd)
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
            if positionals and positionals[0] == "call":
                if len(positionals) != 3 or flags:
                    return "usage: /capability call <name> <config-json>"
                capability_name = positionals[1]
                config_json = positionals[2]
                try:
                    config = json.loads(config_json)
                except json.JSONDecodeError:
                    return "error: invalid JSON config"
                if not isinstance(config, dict):
                    return "error: config must be a JSON object"
                from mr1.capability_runner import CapabilityRunner
                runner = CapabilityRunner(
                    scoped_agent_store=self._scoped_agents,
                    message_store=self._message_store,
                    workspace_root=self._workflow_store.root.parent,
                )
                try:
                    result = runner.run_capability(capability_name, config, self._root_agent_id)
                except ValueError as exc:
                    return f"error: {exc}"
                lines = [
                    f"capability:   {result.capability}",
                    f"status:       {result.status}",
                    f"duration_ms:  {result.duration_ms}",
                    "output:",
                    json.dumps(result.output, indent=2, sort_keys=True),
                ]
                if result.error is not None:
                    lines.append(f"error:        {result.error}")
                return "\n".join(lines)
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
        usage = (
            "usage: /agent <create <title>|kill <ag-id>|assign <ag-id> <mission-file>|"
            "step <ag-id>|run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]|"
            "<ag-id>|kazi [health]> [--json] [--brief]"
        )
        try:
            parts = shlex.split(cmd)
        except ValueError:
            if cmd.startswith("/agent"):
                return usage
            return "usage: /agents [--json] [--brief]"

        command = parts[0]
        if command == "/agent" and len(parts) > 1 and parts[1] == "run":
            return self._handle_agent_run_builtin(parts[2:], usage)
        flags = {part for part in parts[1:] if part.startswith("--")}
        positionals = [part for part in parts[1:] if not part.startswith("--")]
        allowed_flags = {"--json", "--brief"}
        if any(flag not in allowed_flags for flag in flags):
            if command == "/agent":
                return usage
            return "usage: /agents [--json] [--brief]"

        if command == "/agents":
            if positionals:
                return "usage: /agents [--json] [--brief]"
            return workflow_cli._format_agents(
                self._scoped_agents.list_visible_agents(self._root_agent_id),
                json_output="--json" in flags,
                brief="--brief" in flags,
            )

        if not positionals:
            return usage
        if positionals[0] == "create":
            title = " ".join(positionals[1:]).strip()
            if not title:
                return "usage: /agent create <title>"
            try:
                agent = self._scoped_agents.create_child_agent(self._root_agent_id, title)
            except ValueError as exc:
                return str(exc)
            return agent.agent_id
        if positionals[0] == "kill":
            if len(positionals) != 2:
                return "usage: /agent kill <ag-id>"
            try:
                agent = self._scoped_agents.terminate_agent(self._root_agent_id, positionals[1])
            except (ValueError, AgentScopeError) as exc:
                return str(exc)
            return agent.agent_id
        if positionals[0] == "assign":
            if len(positionals) != 3:
                return "usage: /agent assign <ag-id> <mission-file>"
            mission_path = Path(positionals[2])
            if not mission_path.exists():
                return f"error: mission file not found: {mission_path}"
            try:
                mission = mission_path.read_text(encoding="utf-8")
            except OSError:
                return f"error: mission file not found: {mission_path}"
            try:
                agent = self._scoped_agents.assign_mission(
                    self._root_agent_id,
                    positionals[1],
                    mission,
                )
            except (ValueError, AgentScopeError) as exc:
                return str(exc)
            return agent.agent_id
        if positionals[0] == "step":
            if len(positionals) != 2:
                return "usage: /agent step <ag-id>"
            runner = MRnStepRunner(
                workflow_store=self._workflow_store,
                scoped_agent_store=self._scoped_agents,
                message_store=self._message_store,
                workflow_authoring_service=self._workflow_authoring,
            )
            try:
                result = runner.step(positionals[1], caller_agent_id=self._root_agent_id)
            except (ValueError, AgentScopeError) as exc:
                return str(exc)
            return workflow_cli._format_mrn_step_result(result)

        agent_name = positionals[0]
        action = positionals[1] if len(positionals) > 1 else None
        if agent_name.startswith("ag-"):
            if len(positionals) != 1:
                return "usage: /agent <ag-id>"
            try:
                agent = self._scoped_agents.get_visible_agent(self._root_agent_id, agent_name)
            except (ValueError, AgentScopeError) as exc:
                return str(exc)
            return workflow_cli._format_agent(
                agent,
                reports=self._scoped_agents.list_reports(agent.agent_id),
                message_store=self._message_store,
                workflow_store=self._workflow_store,
                json_output="--json" in flags,
                brief="--brief" in flags,
            )

        if len(positionals) > 2 or (action is not None and action != "health"):
            return usage
        try:
            if action == "health":
                return workflow_cli._format_runtime_agent_health(
                    agent_name,
                    json_output="--json" in flags,
                )
            return workflow_cli._format_runtime_agent(
                agent_name,
                json_output="--json" in flags,
                brief="--brief" in flags,
            )
        except ValueError:
            return f"agent not found: {agent_name}"

    def _handle_agent_run_builtin(self, parts: list[str], usage: str) -> str:
        if not parts:
            return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
        agent_id = parts[0]
        if agent_id.startswith("--"):
            return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
        steps = 3
        max_workflows = 2
        require_confirmation_for_workflows = True
        index = 1
        while index < len(parts):
            token = parts[index]
            if token == "--steps":
                if index + 1 >= len(parts):
                    return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
                try:
                    steps = int(parts[index + 1])
                except ValueError:
                    return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
                index += 2
                continue
            if token == "--max-workflows":
                if index + 1 >= len(parts):
                    return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
                try:
                    max_workflows = int(parts[index + 1])
                except ValueError:
                    return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
                index += 2
                continue
            if token == "--no-confirm-workflows":
                require_confirmation_for_workflows = False
                index += 1
                continue
            return usage

        runner = MRnRunRunner(
            workflow_store=self._workflow_store,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
        )
        policy = MRnRunPolicy(
            max_steps=steps,
            max_workflows_created=max_workflows,
            require_confirmation_for_workflows=require_confirmation_for_workflows,
        )
        try:
            result = runner.run(agent_id, policy, caller_agent_id=self._root_agent_id)
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        return workflow_cli._format_mrn_run_result(result)

    def _handle_message_builtin(self, cmd: str) -> str:
        inbox_usage = "usage: /inbox [--archived] | /inbox triage [--max-actions N] [--max-messages N] | /inbox triage on | /inbox triage off | /inbox triage status"
        try:
            parts = shlex.split(cmd)
        except ValueError:
            if cmd.startswith("/message"):
                return "usage: /message <message_id> | /message read <message_id> | /message archive <message_id> | /message send <agent_id> <subject> <body-file>"
            if cmd.startswith("/outbox"):
                return "usage: /outbox"
            return inbox_usage

        command = parts[0]
        flags = {part for part in parts[1:] if part.startswith("--")}
        positionals = [part for part in parts[1:] if not part.startswith("--")]
        if command == "/inbox":
            if positionals and positionals[0] == "triage" and len(positionals) == 2:
                sub = positionals[1]
                if sub == "on":
                    if self._inbox_thread and self._inbox_thread.is_alive():
                        return "inbox auto-triage already running"
                    self._inbox_stop.clear()
                    self._inbox_thread = threading.Thread(
                        target=self._run_inbox_loop,
                        name="inbox-triage",
                        daemon=True,
                    )
                    self._inbox_thread.start()
                    return "inbox auto-triage started"
                if sub == "off":
                    self._inbox_stop.set()
                    if self._inbox_thread:
                        self._inbox_thread.join(timeout=2.0)
                        self._inbox_thread = None
                    return "inbox auto-triage stopped"
                if sub == "status":
                    running = bool(self._inbox_thread and self._inbox_thread.is_alive())
                    return f"inbox auto-triage: {'running' if running else 'stopped'} (interval: {self._inbox_triage_interval_s}s)"
                return inbox_usage
            if positionals and positionals[0] == "triage":
                if "--archived" in flags:
                    return inbox_usage
                max_messages = 10
                max_actions = 3
                index = 1
                while index < len(parts):
                    token = parts[index]
                    if token == "triage":
                        index += 1
                        continue
                    if token == "--max-actions":
                        if index + 1 >= len(parts):
                            return inbox_usage
                        try:
                            max_actions = int(parts[index + 1])
                        except ValueError:
                            return inbox_usage
                        index += 2
                        continue
                    if token == "--max-messages":
                        if index + 1 >= len(parts):
                            return inbox_usage
                        try:
                            max_messages = int(parts[index + 1])
                        except ValueError:
                            return inbox_usage
                        index += 2
                        continue
                    return inbox_usage
                try:
                    runner = self._make_inbox_triage_runner()
                    result = runner.run(
                        InboxTriagePolicy(
                            max_messages=max_messages,
                            max_actions=max_actions,
                        ),
                        caller_agent_id=self._root_agent_id,
                    )
                except ValueError as exc:
                    return f"error: {exc}"
                return workflow_cli._format_inbox_triage_result(result)
            if flags - {"--archived"} or positionals:
                return inbox_usage
            return workflow_cli._format_messages_table(
                self._message_store.list_inbox(
                    self._root_agent_id,
                    include_archived="--archived" in flags,
                ),
                mode="inbox",
            )
        if command == "/outbox":
            if flags:
                return "usage: /outbox"
            if positionals:
                return "usage: /outbox"
            return workflow_cli._format_messages_table(
                self._message_store.list_outbox(self._root_agent_id),
                mode="outbox",
            )

        if flags:
            return "usage: /message <message_id> | /message read <message_id> | /message archive <message_id> | /message send <agent_id> <subject> <body-file>"
        if not positionals:
            return "usage: /message <message_id> | /message read <message_id> | /message archive <message_id> | /message send <agent_id> <subject> <body-file>"
        if positionals[0] == "read":
            if len(positionals) != 2:
                return "usage: /message read <message_id>"
            try:
                workflow_cli._require_message(
                    self._message_store,
                    positionals[1],
                    self._root_agent_id,
                )
                message = self._message_store.mark_read(positionals[1])
                if message is None:
                    return f"message not found: {positionals[1]}"
                return message.message_id
            except (ValueError, AgentScopeError) as exc:
                return str(exc)
        if positionals[0] == "archive":
            if len(positionals) != 2:
                return "usage: /message archive <message_id>"
            try:
                workflow_cli._require_message(
                    self._message_store,
                    positionals[1],
                    self._root_agent_id,
                )
                message = self._message_store.archive_message(positionals[1])
                if message is None:
                    return f"message not found: {positionals[1]}"
                return message.message_id
            except (ValueError, AgentScopeError) as exc:
                return str(exc)
        if positionals[0] == "send":
            if len(positionals) != 4:
                return "usage: /message send <agent_id> <subject> <body-file>"
            body_path = Path(positionals[3])
            if not body_path.exists():
                return f"error: message body file not found: {body_path}"
            try:
                body = body_path.read_text(encoding="utf-8")
            except OSError:
                return f"error: message body file not found: {body_path}"
            if not self._message_store.can_agent_send_message(self._root_agent_id, positionals[1]):
                return "access denied: recipient not in agent scope"
            message = self._message_store.create_message(
                from_agent_id=self._root_agent_id,
                to_agent_id=positionals[1],
                kind="request",
                subject=positionals[2],
                body=body,
            )
            return message.message_id
        if len(positionals) != 1:
            return "usage: /message <message_id>"
        try:
            message = workflow_cli._require_message(
                self._message_store,
                positionals[0],
                self._root_agent_id,
            )
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        return workflow_cli._format_message_detail(message)

    def _handle_approval_builtin(self, cmd: str) -> str:
        usage = (
            "usage: /approvals list | /approvals show <approval_request_id> | "
            "/approvals approve <approval_request_id> [--grant-scope] [--reason TEXT] | "
            "/approvals deny <approval_request_id> [--reason TEXT]"
        )
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return usage
        if len(parts) < 2:
            return usage

        subcommand = parts[1]
        if subcommand == "list":
            if len(parts) != 2:
                return usage
            approvals = workflow_cli._visible_approvals(
                self._approval_store,
                self._scoped_agents,
                self._root_agent_id,
            )
            return workflow_cli._format_approvals_table(approvals)

        if subcommand == "show":
            if len(parts) != 3:
                return usage
            try:
                approval = workflow_cli._require_visible_approval(
                    self._approval_store,
                    parts[2],
                    self._scoped_agents,
                    self._root_agent_id,
                )
            except (ValueError, AgentScopeError) as exc:
                return str(exc)
            return workflow_cli._format_approval(approval)

        if subcommand not in {"approve", "deny"}:
            return usage
        if len(parts) < 3:
            return usage

        approval_request_id = parts[2]
        grant_scope = False
        reason = "approved" if subcommand == "approve" else "denied"
        index = 3
        while index < len(parts):
            token = parts[index]
            if token == "--grant-scope":
                if subcommand != "approve":
                    return usage
                grant_scope = True
                index += 1
                continue
            if token == "--reason":
                if index + 1 >= len(parts):
                    return usage
                reason = parts[index + 1]
                index += 2
                continue
            return usage

        try:
            workflow_cli._require_visible_approval(
                self._approval_store,
                approval_request_id,
                self._scoped_agents,
                self._root_agent_id,
            )
            decision = CapabilityApprovalDecision(
                approval_request_id=approval_request_id,
                decision="approved" if subcommand == "approve" else "denied",
                decided_by=self._root_agent_id,
                reason=reason,
                timestamp=time.time(),
                approval_scope="grant_scope" if grant_scope else "single_use",
            )
            updated = self._approval_store.apply_decision(
                approval_request_id,
                decision=decision,
                scoped_agent_store=self._scoped_agents,
            )
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        if (
            updated.requesting_actor_id != self._root_agent_id
            and self._message_store.can_agent_send_message(
                self._root_agent_id,
                updated.requesting_actor_id,
            )
        ):
            try:
                self._message_store.create_message(
                    from_agent_id=self._root_agent_id,
                    to_agent_id=updated.requesting_actor_id,
                    kind="request",
                    subject=f"Capability approval {updated.status}: {updated.capability_name}",
                    body=(
                        f"Approval request {updated.approval_request_id} is now {updated.status}. "
                        f"Reason: {reason}. "
                        "Retry or continue your capability step accordingly."
                    ),
                    workflow_id=updated.workflow_id,
                    task_id=updated.task_id,
                )
            except ValueError:
                pass
        return workflow_cli._format_approval_decision_result(
            updated,
            grant_scope=grant_scope,
        )

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
                spec,
                Provenance(type="agent", id="MR1"),
                caller_agent_id=self._root_agent_id,
                owner_agent_id=self._root_agent_id,
            )
        except WorkflowSpecError as exc:
            return f"invalid workflow: {exc}"
        return f"submitted: {wf_id}"

    def _make_inbox_triage_runner(self) -> "InboxTriageRunner":
        compiler_client = WorkflowCompilerClient(
            compiler=getattr(self._workflow_authoring, "_compiler", None) or self._run_workflow_compiler,
            scoped_agent_store=self._scoped_agents,
        )
        return InboxTriageRunner(
            workflow_store=self._workflow_store,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
            workflow_compiler_client=compiler_client,
            workflow_authoring_service=self._workflow_authoring,
            pending_workflow_state=self._state,
        )

    def _run_inbox_loop(self) -> None:
        while not self._inbox_stop.is_set():
            self._inbox_stop.wait(self._inbox_triage_interval_s)
            if self._inbox_stop.is_set():
                break
            try:
                unread = [
                    m for m in self._message_store.list_inbox(self._root_agent_id)
                    if m.status == "unread"
                ]
                if not unread:
                    continue
                runner = self._make_inbox_triage_runner()
                runner.run(
                    InboxTriagePolicy(max_messages=10, max_actions=5),
                    caller_agent_id=self._root_agent_id,
                )
            except Exception:
                pass

    def shutdown(self, reason: str = "user") -> int:
        self._inbox_stop.set()
        if self._inbox_thread is not None:
            self._inbox_thread.join(timeout=2.0)
        killed = self._spawner.kill_all(reason)
        self.kill_test_agents()
        self._scheduler.shutdown(cancel_running=True)
        if self._web_viz_server is not None:
            self._web_viz_server.stop()
            self._web_viz_server = None
        if self._process:
            self._process.kill()
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
            "/tools  /tool <type>  /agents  /agent <ag-id>  /agent create <title>  "
            "/agent step <ag-id>  /agent run <ag-id> --steps N  "
            "/inbox  /outbox  /message <id>  /approvals  /schema  /vizualize  /visualize-web  "
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
