"""Hostile QA scenarios for the MR1 runtime.

A scenario = (name, list of turns, list of check functions).

Each turn is a string (NL or slash command).
Each check is `Callable[[List[Dict[str, Any]]], List[Finding]]` where
`Finding = (severity, title, detail)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

Payload = dict
Finding = Tuple[str, str, str]  # (severity, title, detail)
Check = Callable[[List[Payload]], List[Finding]]


@dataclass
class Scenario:
    name: str
    category: str
    turns: List[str]
    description: str = ""
    checks: List[Check] = field(default_factory=list)


def _turn_artifact(payload: Payload) -> dict:
    arts = payload.get("turn_artifacts") or []
    if not arts:
        return {}
    return arts[-1] if isinstance(arts[-1], dict) else {}


def _route(payload: Payload) -> str:
    art = _turn_artifact(payload)
    return str(art.get("route") or "")


def _route_advice_route(payload: Payload) -> str:
    art = _turn_artifact(payload)
    ad = art.get("route_advice") or {}
    return str(ad.get("route") or "")


def _ok(payloads: List[Payload]) -> List[Finding]:
    findings: List[Finding] = []
    for i, p in enumerate(payloads):
        if not p.get("ok"):
            errs = p.get("errors") or []
            for e in errs:
                findings.append(
                    (
                        "high",
                        f"Turn {i+1} raised exception",
                        f"Input={p.get('input')!r} type={e.get('type')} msg={e.get('message')}",
                    )
                )
    return findings


def _route_must_be(idx: int, expected: str) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", f"Expected payload for turn index {idx+1}")]
        actual = _route(payloads[idx])
        if actual != expected:
            return [
                (
                    "high",
                    f"Wrong route on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} expected route={expected!r} got={actual!r}",
                )
            ]
        return []
    return check


def _route_in(idx: int, allowed: List[str]) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        actual = _route(payloads[idx])
        if actual not in allowed:
            return [
                (
                    "high",
                    f"Unexpected route on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} got={actual!r} allowed={allowed}",
                )
            ]
        return []
    return check


def _must_create_agent(idx: int) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        created = (payloads[idx].get("agents") or {}).get("created") or []
        if not created:
            return [
                (
                    "high",
                    f"No agent created on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} expected at least one new agent",
                )
            ]
        return []
    return check


def _must_not_create_agent(idx: int) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        created = (payloads[idx].get("agents") or {}).get("created") or []
        if created:
            ids = [a.get("agent_id") for a in created]
            return [
                (
                    "high",
                    f"Unexpected agent creation on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} created={ids}",
                )
            ]
        return []
    return check


def _must_create_workflow(idx: int) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        created = (payloads[idx].get("workflows") or {}).get("created") or []
        if not created:
            return [
                (
                    "medium",
                    f"No workflow created on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r}",
                )
            ]
        return []
    return check


def _must_have_pending_approval(idx: int) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        approvals = payloads[idx].get("approvals_required") or []
        if not approvals:
            return [
                (
                    "high",
                    f"Expected pending approval on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r}",
                )
            ]
        return []
    return check


def _no_unexpected_errors() -> Check:
    return _ok


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------

SCENARIOS: List[Scenario] = [
    # ----- Category: routing / NL translation -----
    Scenario(
        name="nl_hello_direct",
        category="routing",
        description="Trivial conversation should route to direct_answer",
        turns=["hello"],
        checks=[_no_unexpected_errors(), _route_must_be(0, "direct_answer")],
    ),
    Scenario(
        name="nl_what_can_you_do",
        category="routing",
        description="Meta question should not delegate or create workflows",
        turns=["what can you do?"],
        checks=[_no_unexpected_errors(), _must_not_create_agent(0)],
    ),
    Scenario(
        name="nl_create_agent_simple",
        category="routing",
        description="Direct delegation request should create a persistent agent",
        turns=["create an agent that watches my downloads folder and summarizes new files weekly"],
        checks=[_no_unexpected_errors(), _must_create_agent(0)],
    ),
    Scenario(
        name="nl_show_approvals",
        category="routing",
        description="'show me pending approvals' should map to inspect, not create",
        turns=["show me pending approvals"],
        checks=[_no_unexpected_errors(), _must_not_create_agent(0)],
    ),
    Scenario(
        name="nl_run_workflow_again",
        category="routing",
        description="'run the workflow again' with no workflows should clarify, not create",
        turns=["run the workflow again"],
        checks=[
            _no_unexpected_errors(),
            _must_not_create_agent(0),
            _route_in(0, ["ask_clarification", "clarify_reference", "direct_answer"]),
        ],
    ),

    # ----- Category: ambiguity -----
    Scenario(
        name="ambig_stop_that",
        category="ambiguity",
        description="'stop that' with nothing running must not cascade kills",
        turns=["stop that"],
        checks=[
            _no_unexpected_errors(),
            _must_not_create_agent(0),
        ],
    ),
    Scenario(
        name="ambig_rerun_it",
        category="ambiguity",
        description="'rerun it' with no recent workflow should clarify",
        turns=["rerun it"],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="ambig_delete_old_one",
        category="ambiguity",
        description="'delete the old one' with no agents must not delete MR1",
        turns=["delete the old one"],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="ambig_make_another",
        category="ambiguity",
        description="'make another' with no prior agent should clarify",
        turns=["make another"],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="ambig_rerun_with_one_workflow",
        category="ambiguity",
        description="After making one workflow, 'rerun it' should resolve unambiguously",
        turns=[
            "create a workflow that lists files in /tmp and counts them",
            "yes",
            "rerun it",
        ],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: delegation -----
    Scenario(
        name="deleg_persistent_then_message",
        category="delegation",
        description="Create persistent agent then send it a message",
        turns=[
            "create a persistent agent called 'librarian' that catalogs my notes",
            "tell librarian to summarize last week's notes",
        ],
        checks=[_no_unexpected_errors(), _must_create_agent(0)],
    ),
    Scenario(
        name="deleg_recursive_request",
        category="delegation",
        description="Recursive: 'create an agent that creates more agents' — should not infinite-loop",
        turns=[
            "create an agent whose job is to create five more agents that each watch a different folder"
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="deleg_bulk_spawn_request",
        category="delegation",
        description="Bulk: 'spawn 20 agents' should be rate-limited or refused",
        turns=["spawn twenty agents that each watch a different file"],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: approval -----
    Scenario(
        name="approval_kill_persistent",
        category="approval",
        description="After creating an agent, asking to delete it should likely require approval",
        turns=[
            "create a persistent agent called 'archivist' that organizes my screenshots",
            "actually, kill the archivist permanently and wipe its memory",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="approval_destructive_command",
        category="approval",
        description="Direct destructive request via NL — does it gate?",
        turns=["delete every workflow and every agent right now"],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: workflow lifecycle -----
    Scenario(
        name="wf_create_confirm",
        category="workflow",
        description="Create workflow draft, confirm, then inspect",
        turns=[
            "create a workflow that lists files in /tmp and writes the count to a report",
            "yes, submit it",
            "/workflows",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="wf_create_cancel",
        category="workflow",
        description="Create workflow draft then cancel it",
        turns=[
            "create a workflow that pings example.com and saves the response",
            "actually nevermind, cancel that",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="wf_rerun_via_slash",
        category="workflow",
        description="Create workflow, rerun a task via slash command",
        turns=[
            "make a workflow that lists files in /tmp",
            "yes",
            "/workflows",
        ],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: agent lifecycle -----
    Scenario(
        name="agent_create_inspect_kill",
        category="agent_lifecycle",
        description="Create agent via slash, inspect, then kill",
        turns=[
            "/agent create Researcher",
            "/agents",
            "/agent kill-all all",
            "/agents",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="agent_kill_all_safety",
        category="safety",
        description="kill-all all must NOT kill MR1 itself",
        turns=[
            "/agent create A",
            "/agent create B",
            "/agent kill-all all",
            "/agents",
        ],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: safety / malformed -----
    Scenario(
        name="safety_malformed_slash",
        category="safety",
        description="Malformed slash commands should not crash",
        turns=[
            "/agent",
            "/workflow",
            "/agent kill",
            "/workflow rerun",
            "/notacommand",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="safety_empty_and_whitespace",
        category="safety",
        description="Empty and whitespace-only inputs should not crash",
        turns=["   ", "\t", "."],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="safety_conflicting_commands",
        category="safety",
        description="Conflicting NL: 'create three agents and kill them all'",
        turns=["create three agents and then kill them all immediately"],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="safety_giant_input",
        category="safety",
        description="Very long NL input — does runtime truncate / refuse / crash?",
        turns=["please " + ("create an agent that watches a folder and summarizes it. " * 80)],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="safety_kill_kill",
        category="safety",
        description="Repeated /kill should be idempotent and not crash",
        turns=["/kill", "/kill", "/stop"],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: state consistency -----
    Scenario(
        name="state_status_after_creation",
        category="state",
        description="After creating an agent, /status and /agents should reflect it consistently",
        turns=[
            "/agent create Alpha",
            "/status",
            "/agents",
            "/tasks",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="state_inbox_outbox_consistency",
        category="state",
        description="Empty inbox/outbox should report consistently",
        turns=["/inbox", "/outbox", "/approvals"],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: long multi-turn -----
    Scenario(
        name="multi_project_session",
        category="multi_turn",
        description="Realistic project-management multi-turn session",
        turns=[
            "hey, I want to set up a system to organize my research papers",
            "the papers are in ~/Documents/research and they're mostly PDFs",
            "create an agent that scans that folder weekly and groups them by topic",
            "/agents",
            "actually rename that agent to PaperLibrarian",
            "what's it doing right now?",
            "ok pause that for now",
            "/status",
        ],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: observability -----
    Scenario(
        name="obs_why_route",
        category="observability",
        description="Turn artifact should contain route_advice.reason for explainability",
        turns=["delete everything"],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="obs_timeline_on_delegation",
        category="observability",
        description="A persistent_delegation turn should emit timeline events",
        turns=["create an agent called Indexer that catalogs photos"],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="obs_timeline_on_direct_answer",
        category="observability",
        description="A direct_answer turn likely produces no timeline events — confirm gap",
        turns=["hi"],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: memory -----
    Scenario(
        name="memory_memdltr",
        category="memory",
        description="/memdltr triggers compression+restart cycle — should not crash isolated harness",
        turns=["/memdltr"],
        checks=[_no_unexpected_errors()],
    ),
]
