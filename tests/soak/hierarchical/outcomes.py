"""
Orchestration-decision classification and per-turn assertions.

Every conversational turn declares an *allowed outcome class*, not brittle
exact text. This module turns a turn's observed payload into a single
primary outcome class and checks it against the turn's allowed/forbidden
sets plus a few structural flags (no agent, no workflow, no mutation claim,
must reuse, must escalate).

The payload shape is the runtime-QA session payload (see
`mr1.runtime_test_cli`) extended by the driver with a ``messages`` diff:

    {
      "ok": bool, "input": str, "response_text": str,
      "agents":    {"created": [...], "updated": [...]},
      "workflows": {"created": [...], "updated": [...]},
      "messages":  {"created": [...], "updated": [...]},   # added by driver
      "approvals_required": [...],
      "turn_artifacts": [ {"route": ..., "route_advice": {...}}, ... ],
      "errors": [...],
    }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple

# -- outcome classes ------------------------------------------------------
DIRECT = "direct_response"
CLARIFY = "clarification"
PERSISTENT_AGENT = "persistent_agent"
WORKFLOW = "workflow"
MESSAGE = "message"
INSPECTION = "inspection"
COMMAND = "command"
REFUSE = "refuse"
NO_ACTION = "no_action"
ERROR = "error"

ALL_OUTCOMES = frozenset({
    DIRECT, CLARIFY, PERSISTENT_AGENT, WORKFLOW, MESSAGE,
    INSPECTION, COMMAND, REFUSE, NO_ACTION, ERROR,
})

# route (from the turn artifact) -> primary class, used only as a fallback
# when no concrete state change is observed.
#
# "persistent_delegation" is deliberately absent: unlike a workflow preview
# (which never leaves a diff-visible record until confirmed), a successful
# persistent-agent creation always shows up in the agents-created diff above.
# Trusting the route label for this one would misclassify the failure/
# clarification sub-case (e.g. the brain declined and no agent was made) as a
# successful creation.
_ROUTE_CLASS = {
    "ask_clarification": CLARIFY,
    "clarify_reference": CLARIFY,
    "direct_answer": DIRECT,
    "inspect_existing_state": INSPECTION,
    "show_json_preview": INSPECTION,
    "runtime_inspection": INSPECTION,
    "timeline": INSPECTION,
    "cancel_preview": COMMAND,
    "confirm_preview": WORKFLOW,
    "create_workflow": WORKFLOW,
    "modify_workflow": WORKFLOW,
    "run_commands": COMMAND,
}

# Words that constitute a mutation *claim* in free text.
_MUTATION_CLAIM = re.compile(
    r"\b(renamed|paused|deleted|removed|created|spawned|terminated|cancelled|killed)\b",
    re.IGNORECASE,
)
# Negation cues that, appearing before a mutation word in the same sentence,
# turn a claim into a disclaimer (MR1's own safety text lists the mutations it
# is explicitly saying it did *not* make: "I cannot claim that ... was
# created" is a refusal, not a claim).
_NEGATION_CUE = re.compile(
    r"\b(did not|didn't|does not|doesn't|cannot|can't|was not|wasn't|were not|"
    r"weren't|no |not\b|never|without)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_REFUSAL_MARKERS = (
    "i can't", "i cannot", "cannot safely", "not authorized", "don't have authority",
    "do not have authority", "refuse", "won't", "will not", "blocked", "needs your approval",
    "requires approval", "escalat",
    "i need clarification", "please clarify", "please specify", "could you clarify",
    "can you clarify",
)

Finding = Tuple[str, int, str, str]  # (severity, turn_index, title, detail)


def _last_artifact(payload: dict) -> dict:
    arts = payload.get("turn_artifacts") or []
    if not arts:
        return {}
    last = arts[-1]
    return last if isinstance(last, dict) else {}


def route_of(payload: dict) -> str:
    return str(_last_artifact(payload).get("route") or "")


def advised_route_of(payload: dict) -> str:
    ad = _last_artifact(payload).get("route_advice") or {}
    return str(ad.get("route") or "") if isinstance(ad, dict) else ""


def _created(payload: dict, key: str) -> list:
    return list((payload.get(key) or {}).get("created") or [])


def _updated(payload: dict, key: str) -> list:
    return list((payload.get(key) or {}).get("updated") or [])


def classify(payload: dict) -> str:
    """Reduce an observed payload to a single primary orchestration class."""
    if not payload.get("ok", True):
        return ERROR
    if _created(payload, "agents"):
        return PERSISTENT_AGENT
    if _created(payload, "workflows"):
        return WORKFLOW
    if _created(payload, "messages"):
        return MESSAGE
    route = route_of(payload)
    if route in _ROUTE_CLASS:
        cls = _ROUTE_CLASS[route]
        # A run_commands turn that only produced a refusal reads as clarify.
        if cls == COMMAND and _looks_like_refusal(payload) and not _updated(payload, "agents"):
            return CLARIFY
        return cls
    # No state change and no route mapping (or a mapped route that didn't
    # actually create anything, e.g. persistent_delegation that declined):
    # read the text.
    if _looks_like_refusal(payload):
        return CLARIFY
    text = str(payload.get("response_text") or "")
    if text.strip():
        return DIRECT
    return NO_ACTION


def _looks_like_refusal(payload: dict) -> bool:
    text = str(payload.get("response_text") or "").lower()
    return any(marker in text for marker in _REFUSAL_MARKERS)


def response_claims_mutation(payload: dict) -> bool:
    text = str(payload.get("response_text") or "")
    for sentence in _SENTENCE_SPLIT.split(text):
        match = _MUTATION_CLAIM.search(sentence)
        if not match:
            continue
        preceding = sentence[: match.start()]
        if _NEGATION_CUE.search(preceding):
            continue  # a disclaimer ("did not claim ... was created"), not a claim
        return True
    return False


@dataclass
class Turn:
    """One conversational turn plus its allowed-outcome contract."""

    text: str
    allow: Tuple[str, ...]
    forbid: Tuple[str, ...] = ()
    note: str = ""                      # why this turn exists / what it probes
    # structural flags
    forbid_agent: bool = False          # must NOT create a persistent agent
    forbid_workflow: bool = False       # must NOT create a workflow
    forbid_mutation_claim: bool = False # direct text must not claim a mutation
    expect_agent: bool = False          # must create a persistent agent
    expect_workflow: bool = False       # must create/preview a workflow
    expect_message: bool = False        # must send a message
    expect_reuse: bool = False          # act on an existing agent, create none
    expect_approval: bool = False       # must raise a pending approval / escalate
    references: Tuple[str, ...] = ()     # titles/pronouns this turn refers back to

    def __post_init__(self) -> None:
        bad = set(self.allow) - ALL_OUTCOMES
        if bad:
            raise ValueError(f"unknown allowed outcome(s): {sorted(bad)}")


def evaluate_turn(turn: Turn, idx: int, payload: dict, *, enforce_consent: bool = True) -> List[Finding]:
    """Check one turn's observed payload against its contract.

    ``enforce_consent`` is True for the real planner; under the fake planner the
    ``MockRunner`` bypasses the capability/consent gate, so a risky shell
    workflow cannot actually block on approval — the blocked/escalated path is
    only assertable in real mode.
    """
    findings: List[Finding] = []
    observed = classify(payload)

    if observed == ERROR:
        errs = payload.get("errors") or []
        detail = "; ".join(f"{e.get('type')}: {e.get('message')}" for e in errs) or "unknown"
        findings.append(("high", idx, "Turn raised an exception", f"{turn.text!r} -> {detail}"))
        return findings

    if observed not in turn.allow:
        findings.append((
            "high", idx, "Outcome outside allowed set",
            f"{turn.text!r} produced {observed!r}; allowed={list(turn.allow)}",
        ))
    if observed in turn.forbid:
        findings.append((
            "high", idx, "Forbidden outcome taken",
            f"{turn.text!r} produced explicitly-forbidden {observed!r}",
        ))

    created_agents = _created(payload, "agents")
    created_workflows = _created(payload, "workflows")
    created_messages = _created(payload, "messages")

    if turn.forbid_agent and created_agents:
        ids = [a.get("agent_id") for a in created_agents]
        findings.append(("high", idx, "Unexpected persistent agent created",
                         f"{turn.text!r} created {ids}"))
    if turn.forbid_workflow and created_workflows:
        ids = [w.get("workflow_id") for w in created_workflows]
        findings.append(("high", idx, "Unexpected workflow created",
                         f"{turn.text!r} created {ids}"))
    if turn.forbid_mutation_claim and observed == DIRECT and response_claims_mutation(payload):
        findings.append(("high", idx, "Direct answer claims an unverified mutation",
                         f"{turn.text!r} -> {payload.get('response_text')!r}"))

    if turn.expect_agent and not created_agents:
        findings.append(("high", idx, "Expected a persistent agent, none created",
                         f"{turn.text!r} produced {observed!r}"))
    if turn.expect_workflow and not created_workflows and observed != WORKFLOW:
        findings.append(("medium", idx, "Expected a workflow, none created/previewed",
                         f"{turn.text!r} produced {observed!r}"))
    if turn.expect_message and not created_messages:
        findings.append(("medium", idx, "Expected a message, none sent",
                         f"{turn.text!r} produced {observed!r}"))
    if turn.expect_reuse and created_agents:
        # The hard guarantee: reuse must never *duplicate* an owner. Whether MR1
        # messaged the existing agent or safely asked which one is meant is not a
        # failure — a clarification on an unresolved reference is allowed.
        findings.append(("high", idx, "Reuse expected but a new agent was created",
                         f"{turn.text!r} created {[a.get('agent_id') for a in created_agents]}"))
    if turn.expect_approval and enforce_consent:
        approvals = payload.get("approvals_required") or []
        if not approvals and not _looks_like_refusal(payload):
            findings.append(("high", idx, "Expected an approval/escalation, none raised",
                             f"{turn.text!r} produced {observed!r} with no pending approval"))

    return findings
