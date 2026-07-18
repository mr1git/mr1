"""
Scripted brain + compiler for the quick fake-planner mode.

The important property of MR1's runtime, established by reading `MR1.step`,
is that the *orchestration decision* — respond / clarify / orchestrator /
run command / inspect / workflow — is made deterministically by
`build_route_advice` on the user's text plus runtime grounding. The brain
(`MR1Process`) is only invoked *inside* a chosen route, to author content:
a direct answer, an orchestrator's mission, or (via the compiler) a
workflow spec.

So a fake brain does not need to "decide" anything. It only needs to return
*valid, non-lying content* for whichever sub-step called it, so that the
whole harness — routing, creation, messaging, sampling, invariants,
reporting — can be exercised end to end without spending a token. That is
exactly what fake mode is for; behavioral quality is judged in real mode.

`FakeBrainProcess` is a drop-in for `MR1Process`. `make_fake_compiler`
returns a `Callable[[str, str], str]` that emits a schema-valid workflow
spec pointed at the disposable fixture repo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from mr1.worker import WorkerResult

# Marker text that only appears in the persistent-agent *design* prompt.
_AGENT_DESIGN_MARKER = "AGENT_TITLE:"

# Words that MR1 treats as an unverified-mutation claim in a direct answer
# (`_direct_response_claims_unverified_mutation`). A fake direct answer must
# avoid all of them or MR1 will replace the text with a refusal.
_MUTATION_WORDS = re.compile(
    r"\b(renam|paus|delet|remov|creat|spawn)",
    re.IGNORECASE,
)

# Keyword → persistent-agent title. Distinct responsibilities get distinct
# titles, which is also what keeps the no-duplicate-title invariant honest.
_TITLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\btest", re.IGNORECASE), "TestingSteward"),
    (re.compile(r"persist|storage|store\b|durab", re.IGNORECASE), "PersistenceWarden"),
    (re.compile(r"architec", re.IGNORECASE), "ArchitectureReviewer"),
    (re.compile(r"secur", re.IGNORECASE), "SecuritySentinel"),
    (re.compile(r"\bdoc", re.IGNORECASE), "DocsKeeper"),
    (re.compile(r"perf|latenc|benchmark", re.IGNORECASE), "PerformanceWatch"),
)


_FALLBACK_TITLES = (
    "DomainOwner", "ScopeSteward", "AreaCurator", "ResponsibilityLead", "OwnerAgent",
)


def _derive_agent_title(user_request: str) -> str:
    for pattern, title in _TITLE_RULES:
        if pattern.search(user_request):
            return title
    return "DomainOwner"


def _next_unique_title(base: str, used: set) -> str:
    """
    Guarantee a fresh title deterministically.

    The keyword-substring heuristic in `_derive_agent_title` can collide with a
    title already issued this session (e.g. a request that explicitly says
    "not the same thing as testing" still contains the substring "testing").
    Rather than parse negation, resolve the collision the same way a careful
    human would: pick a genuinely different, still-descriptive title.
    """
    if base.casefold() not in {t.casefold() for t in used}:
        return base
    for candidate in _FALLBACK_TITLES:
        if candidate.casefold() not in {t.casefold() for t in used}:
            return candidate
    return f"{base}{len(used) + 1}"


class FakeBrainProcess:
    """
    A stand-in for `MR1Process` with no LLM behind it.

    It inspects the prompt only to tell a *direct answer* apart from a
    *persistent-agent design* request, and returns valid content for each.
    Every call is recorded so a soak can assert on idle-tick brain usage.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: str = "fake",
        tools: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._tools = list(tools or [])
        self._session_id = session_id or "fake-session"
        self._available = False
        self.calls: list[str] = []
        self._used_titles: set = set()

    # -- MR1Process interface ------------------------------------------
    def start(self) -> None:
        self._available = True

    def send(self, message: str, *, retriable: bool = False) -> str:
        self.calls.append(message)
        if not self._available:
            return "[MR1 ERROR] Process is not running."
        if _AGENT_DESIGN_MARKER in message:
            return self._design_agent(message)
        return self._direct_answer(message)

    def kill(self) -> None:
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

    # -- content -------------------------------------------------------
    def _design_agent(self, prompt: str) -> str:
        request = _extract_user_request(prompt)
        title = _next_unique_title(_derive_agent_title(request), self._used_titles)
        self._used_titles.add(title)
        mission = (
            f"Own the responsibility described by: {request.strip()[:200]}. "
            "Operate read-only first: inspect the relevant modules, record "
            "findings, and report back before proposing any change. Escalate "
            "to the parent when a change would touch persistence or public "
            "behavior. First actions: read the target modules and summarize "
            "what is fragile."
        )
        return f"{_AGENT_DESIGN_MARKER} {title}\n{mission}"

    def _direct_answer(self, prompt: str) -> str:
        request = _extract_user_request(prompt)
        # Neutral, analytical, and careful to claim no runtime mutation.
        answer = (
            "Here is my read. The store module carries the most risk: its "
            "put path writes the whole file in place without a temporary "
            "file, so an interruption mid-write leaves a truncated store, and "
            "its load path catches every exception so a corrupt file looks "
            "empty. The bounded cache is the careful counterpart. On the "
            "question itself: "
            + request.strip()[:200]
            + " — I would start from the persistence paths and the missing "
            "store tests."
        )
        if _MUTATION_WORDS.search(answer):  # defensive; should not fire
            answer = (
                "The store module's persistence paths look the most fragile; "
                "the bounded cache is safer. I would start there."
            )
        return answer


def fake_mrn_reasoner(agent, system_prompt: str, prompt: str) -> str:
    """
    Drop-in for `mr1.mrn_loop.run_mrn_step_agent`.

    A freshly-created orchestrator runs its own first step via
    `MRnRunRunner`/`MRnStepRunner`, whose default reasoner spawns a *real*
    `claude` subprocess independent of MR1's own `_process` — so patching only
    MR1's brain is not enough to keep a fake-planner soak free of real LLM
    calls. This returns a valid MRn action JSON (see
    `mr1.orchestrator.loop.actions.parser`) matching the read-only-first,
    report-before-changing framing every fake mission is given.
    """
    report = (
        f"Initial read-only pass for {getattr(agent, 'title', 'this agent')}: reviewed the "
        "assigned scope, found the persistence paths to be the most fragile part, "
        "and made no changes. Awaiting further instruction before proposing any fix."
    )
    return json.dumps({
        "action": "write_report",
        "reason": "completed the initial read-only review as instructed",
        "report": report,
        "next_status": "reporting",
    })


def fake_worker_run(
    context: Dict[str, Any],
    spawner: Optional[Any] = None,
    logger: Optional[Any] = None,
    timeout: Optional[int] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> WorkerResult:
    """Drop-in for `mr1.worker.run` under the fake planner.

    A bounded-investigation turn (`root.py`'s `_route_bounded_investigation`)
    calls the real `worker.run()`, which spawns a real `claude` subprocess —
    patching only the brain (`FakeBrainProcess`) and the MRn reasoner
    (`fake_mrn_reasoner`, above) is not enough to keep a fake-planner run
    free of real LLM calls; this is the worker-spawn equivalent. Emits the
    same `task_spawned`/`task_completed`/`task_detached` event sequence the
    real function does, so `StateManager.tasks` (and therefore anything
    built on it) stays populated exactly as it would with a real worker.
    """
    task_id = context.get("task_id", "worker-unknown")
    instructions = str(context.get("instructions") or "")
    parent_task_id = context.get("parent_task_id", "mr1")
    lane = context.get("lane", "conversation")
    description = str(context.get("description") or instructions[:200])

    def _emit(event_type: str, status: str) -> None:
        if event_callback is None:
            return
        event_callback({
            "type": event_type,
            "task_id": task_id,
            "parent_task_id": parent_task_id,
            "agent_type": "worker",
            "status": status,
            "lane": lane,
            "description": description,
        })

    _emit("task_spawned", "running")
    output = (
        f"[FAKE WORKER] Reviewed the assigned scope for: {instructions.strip()[:150]!r}. "
        "Found the persistence paths to be the most fragile part; made no changes."
    )
    _emit("task_completed", "completed")
    _emit("task_detached", "completed")

    return WorkerResult(
        task_id=task_id,
        status="completed",
        output=output,
        error=None,
        duration_s=0.01,
        pid=None,
        payload={"summary": output, "text": output, "status": "succeeded"},
    )


def _extract_user_request(prompt: str) -> str:
    """Pull the user's request back out of an MR1 sub-step prompt."""
    marker = "User request:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


# --- fake workflow compiler ---------------------------------------------

_SHELL_HINTS = re.compile(
    r"\b(status|run|shell|command|pytest|git|execute)\b",
    re.IGNORECASE,
)


def make_fake_compiler(fixture_repo: Path) -> Callable[[str, str], str]:
    """
    Return a compiler ``(system_prompt, user_prompt) -> spec_json``.

    Emits a schema-valid single-task workflow against the fixture repo. If the
    request reads like "run"/"status"/"a command", it emits a `shell_command`
    tool task (which the capability gate will hold for consent — exercising the
    blocked/approval path); otherwise a read-only `read_file` task that submits
    and runs clean.
    """
    fixture_repo = Path(fixture_repo)
    readme = fixture_repo / "README.md"

    def compile(system_prompt: str, user_prompt: str) -> str:
        request = _extract_user_request(user_prompt) or user_prompt
        if _SHELL_HINTS.search(request):
            spec = {
                "title": "Check fixture repo status",
                "tasks": [
                    {
                        "label": "repo_status",
                        "title": "Repo status",
                        "task_kind": "tool",
                        "tool_type": "shell_command",
                        "tool_config": {
                            "argv": ["git", "status", "--short"],
                            "cwd": str(fixture_repo),
                        },
                    }
                ],
            }
        else:
            spec = {
                "title": "Inspect fixture repo readme",
                "tasks": [
                    {
                        "label": "read_readme",
                        "title": "Read readme",
                        "task_kind": "tool",
                        "tool_type": "read_file",
                        "tool_config": {"path": str(readme)},
                    }
                ],
            }
        return json.dumps(spec)

    return compile
