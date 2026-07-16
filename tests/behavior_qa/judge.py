"""
The second judge: an independent `claude` call that reviews one episodic
cluster's transcript against Marwan's stated philosophy and his own
documented behavioral feedback, and returns a structured verdict.

A deterministic checker can assert "an agent was created" or "no crash
occurred" — it cannot assert "this felt like a partner." That's why Category
B turns in `corpus.py` carry no hard contract: this module is the only
evaluator for them, and its judgment is only meaningful when it's a real
`claude` call (see `judge_cluster` / `FakeJudge` below for the zero-cost
harness-testing counterpart, which speaks to none of that).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from mr1.agents import AgentRuntimeError, parse_agent_json_envelope

DIMENSION_KEYS = (
    "initiative", "restraint", "ownership_recognition", "workflow_judgment",
    "reuse", "naturalness", "annoyance", "philosophy_alignment",
)
BEHAVIOR_VALUES = ("TOO_PASSIVE", "BALANCED", "TOO_AGGRESSIVE")
YES_MAYBE_NO = ("YES", "MAYBE", "NO")
SHOULD_HAVE_OPTIONS = (
    "discussed_only", "created_workflow", "created_owner",
    "reused_owner", "clarified", "escalated",
)

_DEFAULT_MODEL = "sonnet"
_DEFAULT_TIMEOUT_S = 180
_DEFAULT_MEMORY_DIR = Path(
    "~/.claude/projects/-Users-marwanabdelgawad-Documents-Projects-mr1/memory"
).expanduser()

_VISION = """\
You are judging MR1, a computational partner (not a workflow manager or a
command parser) for its user Marwan. The philosophy Marwan wants MR1 held to:

  User: "I talk." MR1: "I determine what should happen."

MR1 should feel like a chief of staff or a technical cofounder — someone who
decides structure, ownership, and whether a workflow, a persistent agent, a
clarifying question, or a plain conversational reply is the right response.
It should show healthy initiative without overacting: manufacturing agents,
workflows, or busywork out of a stray sentence is exactly as much a failure
as never picking up on something that clearly needed a standing owner.
"""

_DIMENSIONS_TEXT = """\
Score these 8 dimensions for the cluster as a whole, each 1 (bad) to 5 (great):

  initiative              — did MR1 appropriately take initiative?
  restraint               — did MR1 avoid creating unnecessary work?
  ownership_recognition   — did MR1 recognize long-term responsibilities?
  workflow_judgment       — did MR1 recognize bounded investigations?
  reuse                   — did MR1 reuse existing owners rather than duplicate them?
  naturalness             — did this feel like talking to a partner?
  annoyance               — would this behavior become frustrating if repeated?
                             (5 = very annoying, 1 = not at all)
  philosophy_alignment    — does this match the philosophy above?
"""

_SCHEMA_TEXT = """\
Respond with JSON ONLY — no prose before or after, no markdown code fences.
The object must have exactly this shape:

{
  "dimensions": {
    "initiative": <1-5>, "restraint": <1-5>, "ownership_recognition": <1-5>,
    "workflow_judgment": <1-5>, "reuse": <1-5>, "naturalness": <1-5>,
    "annoyance": <1-5>, "philosophy_alignment": <1-5>
  },
  "behavior": "TOO_PASSIVE" | "BALANCED" | "TOO_AGGRESSIVE",
  "should_have": [zero or more of "discussed_only", "created_workflow",
                  "created_owner", "reused_owner", "clarified", "escalated"],
  "marwan_approval": "YES" | "MAYBE" | "NO",
  "would_continue_using": "YES" | "MAYBE" | "NO",
  "would_marwan_have_done_this_himself": "YES" | "MAYBE" | "NO",
  "surprised_by_action": true | false,
  "partner_score": <integer 0-10>,
  "comments": {
    "intelligent": "<what felt intelligent, or empty string>",
    "annoying": "<what felt annoying, or empty string>",
    "unnatural": "<what felt unnatural, or empty string>"
  }
}

Four of these fields answer genuinely different questions, not restatements
of each other:
- marwan_approval: was *this* interaction handled well?
- would_continue_using: if MR1 behaved like this repeatedly, would Marwan
  keep relying on it as a partner? (the longer-horizon trust question)
- would_marwan_have_done_this_himself: does the specific call MR1 made
  (act / discuss / clarify / spawn / defer) match what Marwan would have
  decided himself in the moment, independent of how cleanly it was executed?
- surprised_by_action: would Marwan be caught off guard by what MR1 did here?
  A balanced, approved action can still be surprising (unexpected but
  welcome initiative); a technically fine one can be surprising in a bad way.
"""

JUDGE_SYSTEM_PROMPT = "\n".join([_VISION, _DIMENSIONS_TEXT, _SCHEMA_TEXT])


@dataclass
class JudgeVerdict:
    dimensions: Dict[str, Optional[int]] = field(default_factory=dict)
    behavior: Optional[str] = None
    should_have: List[str] = field(default_factory=list)
    marwan_approval: Optional[str] = None
    would_continue_using: Optional[str] = None
    would_marwan_have_done_this_himself: Optional[str] = None
    surprised_by_action: Optional[bool] = None
    partner_score: Optional[int] = None
    comments: Dict[str, str] = field(default_factory=dict)
    judge_error: Optional[str] = None

    @classmethod
    def degraded(cls, error: str) -> "JudgeVerdict":
        return cls(
            dimensions={k: None for k in DIMENSION_KEYS},
            comments={"intelligent": "", "annoying": "", "unnatural": ""},
            judge_error=error,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "behavior": self.behavior,
            "should_have": self.should_have,
            "marwan_approval": self.marwan_approval,
            "would_continue_using": self.would_continue_using,
            "would_marwan_have_done_this_himself": self.would_marwan_have_done_this_himself,
            "surprised_by_action": self.surprised_by_action,
            "partner_score": self.partner_score,
            "comments": self.comments,
            "judge_error": self.judge_error,
        }


# ---------------------------------------------------------------------
# Marwan *behavioral* preference replay
# ---------------------------------------------------------------------

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _split_frontmatter(text: str) -> tuple:
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    frontmatter_text, body = parts[1], parts[2]
    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return None, body
    if not isinstance(frontmatter, dict):
        return None, body
    metadata = frontmatter.get("metadata")
    mem_type = metadata.get("type") if isinstance(metadata, dict) else None
    if mem_type is None:
        # Some memory files carry `type:` flat at the frontmatter's top level
        # rather than nested under `metadata:` — accept either shape.
        mem_type = frontmatter.get("type")
    return mem_type, body


def load_marwan_preferences(memory_dir: Optional[Path] = None) -> List[str]:
    """Behavioral precedent only: every memory of type `feedback`, in index
    order. `user`/`project`/`reference` memories are facts about Marwan or
    the project, not corrections/confirmations about how he wants MR1 to
    behave, so they're deliberately excluded. Never raises — a missing or
    unreadable memory store just yields no replayed preferences."""
    root = Path(memory_dir) if memory_dir is not None else _DEFAULT_MEMORY_DIR
    index_path = root / "MEMORY.md"
    if not index_path.exists():
        return []
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except OSError:
        return []

    preferences: List[str] = []
    for match in _LINK_RE.finditer(index_text):
        rel_path = match.group(1).strip()
        mem_path = (root / rel_path).resolve()
        try:
            mem_path.relative_to(root.resolve())
        except ValueError:
            continue  # refuse to follow a link outside the memory dir
        if not mem_path.exists():
            continue
        try:
            text = mem_path.read_text(encoding="utf-8")
        except OSError:
            continue
        mem_type, body = _split_frontmatter(text)
        if mem_type != "feedback":
            continue
        body = body.strip()
        if body:
            preferences.append(body)
    return preferences


# ---------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------

def build_judge_prompt(cluster, turn_records: List[Any], preferences: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"Cluster: {cluster.name} (category {cluster.category})")
    lines.append(f"Topic: {cluster.topic}")
    lines.append(f"What this cluster is testing: {cluster.goal}")
    if cluster.resumes_from:
        lines.append(
            f"This conversation continues cluster {cluster.resumes_from!r} — "
            "real prior state (agents/workflows created earlier) is present "
            "in this runtime, simulating the same computer, a later "
            "conversation, the same thread."
        )
    lines.append("")
    lines.append("Transcript (Marwan's turns, MR1's real responses and real "
                  "runtime actions):")
    for r in turn_records:
        lines.append(f"[{r.index}] Marwan: {r.text}")
        for se in r.soft_expectations:
            lines.append(f"    (soft expectation, not a rule: {se})")
        response = (r.response_text or "")[:600]
        lines.append(f"    MR1 responded: {response!r}")
        advice = r.route_advice or {}
        lines.append(
            f"    routing: advised={advice.get('route')!r} "
            f"confidence={advice.get('confidence')!r} final_route={r.route!r} "
            f"final_action={r.final_action!r}"
        )
        if advice.get("reason"):
            lines.append(f"    advisor reason: {advice.get('reason')!r}")
        if r.override_reason:
            lines.append(f"    override reason: {r.override_reason!r}")
        if r.created_agents:
            titles = [a.get("title") for a in r.created_agents]
            lines.append(f"    created persistent agent(s): {titles}")
        if r.created_workflows:
            lines.append(f"    created workflow(s): {len(r.created_workflows)}")
        if r.created_messages:
            lines.append(f"    sent message(s): {len(r.created_messages)}")
        if r.approval_ids:
            lines.append(f"    raised approval(s)/escalation(s): {r.approval_ids}")
        if not r.ok:
            lines.append(f"    turn raised an exception: {r.errors}")
        lines.append("")

    if preferences:
        lines.append(
            "Known behavioral preferences from prior sessions (things "
            "Marwan has actually said about how MR1 should behave):"
        )
        for p in preferences:
            lines.append(f"- {p}")
        lines.append("")

    lines.append(_SCHEMA_TEXT)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# the subprocess call + parsing
# ---------------------------------------------------------------------

def call_judge(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> str:
    """A harness-level evaluator call, outside MR1's own governed-spawn path
    (no tools, no Dispatcher gate needed — this is the test harness itself
    invoking Claude as a reviewer, exactly like the harness's own
    `--planner real` subprocess use already sits outside that gate)."""
    cmd = [
        "claude", "-p", user_prompt,
        "--model", model,
        "--append-system-prompt", system_prompt,
        "--output-format", "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"judge timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise ValueError(f"failed to run judge: {exc}") from exc

    raw_output = proc.stdout or ""
    if proc.returncode != 0:
        detail = (proc.stderr or raw_output).strip() or f"exit {proc.returncode}"
        raise ValueError(f"judge process failed: {detail}")
    try:
        parsed = parse_agent_json_envelope(raw_output)
    except AgentRuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if parsed["is_error"]:
        raise ValueError(parsed["text"] or "judge returned an error")
    return parsed["text"]


_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(text: str) -> JudgeVerdict:
    """Parse the judge's JSON answer, tolerating a stray prose wrapper."""
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _BRACE_RE.search(text)
        if not match:
            raise ValueError(f"judge output is not JSON: {text[:200]!r}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("judge output JSON must be an object")
    return _verdict_from_dict(data)


def _verdict_from_dict(data: Dict[str, Any]) -> JudgeVerdict:
    dims_raw = data.get("dimensions") or {}
    if not isinstance(dims_raw, dict):
        raise ValueError(f"invalid dimensions: {dims_raw!r}")
    dimensions: Dict[str, Optional[int]] = {}
    for key in DIMENSION_KEYS:
        value = dims_raw.get(key)
        if not isinstance(value, (int, float)) or not (1 <= value <= 5):
            raise ValueError(f"invalid dimension {key}: {value!r}")
        dimensions[key] = int(value)

    behavior = data.get("behavior")
    if behavior not in BEHAVIOR_VALUES:
        raise ValueError(f"invalid behavior: {behavior!r}")

    for field_name in (
        "marwan_approval", "would_continue_using",
        "would_marwan_have_done_this_himself",
    ):
        if data.get(field_name) not in YES_MAYBE_NO:
            raise ValueError(f"invalid {field_name}: {data.get(field_name)!r}")

    should_have = data.get("should_have")
    if not isinstance(should_have, list) or any(
        s not in SHOULD_HAVE_OPTIONS for s in should_have
    ):
        raise ValueError(f"invalid should_have: {should_have!r}")

    surprised = data.get("surprised_by_action")
    if not isinstance(surprised, bool):
        raise ValueError(f"invalid surprised_by_action: {surprised!r}")

    partner_score = data.get("partner_score")
    if not isinstance(partner_score, (int, float)) or not (0 <= partner_score <= 10):
        raise ValueError(f"invalid partner_score: {partner_score!r}")

    comments_raw = data.get("comments") or {}
    comments = {
        "intelligent": str((comments_raw or {}).get("intelligent", "")),
        "annoying": str((comments_raw or {}).get("annoying", "")),
        "unnatural": str((comments_raw or {}).get("unnatural", "")),
    }

    return JudgeVerdict(
        dimensions=dimensions,
        behavior=behavior,
        should_have=list(should_have),
        marwan_approval=data.get("marwan_approval"),
        would_continue_using=data.get("would_continue_using"),
        would_marwan_have_done_this_himself=data.get("would_marwan_have_done_this_himself"),
        surprised_by_action=surprised,
        partner_score=int(partner_score),
        comments=comments,
    )


def judge_cluster(
    cluster,
    turn_records: List[Any],
    *,
    model: str = _DEFAULT_MODEL,
    preferences: Optional[List[str]] = None,
) -> JudgeVerdict:
    """Never raises: any failure (timeout, bad JSON, missing/invalid keys)
    degrades this one cluster's verdict rather than the whole run."""
    try:
        user_prompt = build_judge_prompt(cluster, turn_records, preferences or [])
        raw = call_judge(JUDGE_SYSTEM_PROMPT, user_prompt, model=model)
        return parse_judge_output(raw)
    except Exception as exc:  # noqa: BLE001 — degrade, never propagate
        return JudgeVerdict.degraded(f"{exc.__class__.__name__}: {exc}")


class FakeJudge:
    """Deterministic canned verdict for harness-scaffold unit tests and the
    zero-cost `--planner fake` CLI dry run. Never spends a token, and never
    stands in for a real behavioral judgment."""

    def __call__(
        self,
        cluster,
        turn_records: List[Any],
        *,
        model: str = _DEFAULT_MODEL,
        preferences: Optional[List[str]] = None,
    ) -> JudgeVerdict:
        return JudgeVerdict(
            dimensions={k: 3 for k in DIMENSION_KEYS},
            behavior="BALANCED",
            should_have=["discussed_only"],
            marwan_approval="YES",
            would_continue_using="YES",
            would_marwan_have_done_this_himself="YES",
            surprised_by_action=False,
            partner_score=7,
            comments={"intelligent": "(fake judge)", "annoying": "", "unnatural": ""},
        )
