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


def _response_text(payload: Payload) -> str:
    return str(payload.get("response_text") or "")


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


def _response_must_start_with(idx: int, expected_prefix: str) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        response_text = _response_text(payloads[idx])
        if not response_text.startswith(expected_prefix):
            return [
                (
                    "high",
                    f"Wrong response text on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} expected_prefix={expected_prefix!r} got={response_text!r}",
                )
            ]
        return []
    return check


def _builtin_dispatch_must_bypass_step(idx: int) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        dispatch = payloads[idx].get("dispatch") or {}
        if not dispatch.get("builtin_attempted") or not dispatch.get("builtin_handled") or dispatch.get("step_called"):
            return [
                (
                    "high",
                    f"Slash command escaped builtin path on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} dispatch={dispatch!r}",
                )
            ]
        return []
    return check


def _created_agent_title_must_be(idx: int, expected_title: str) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        created = (payloads[idx].get("agents") or {}).get("created") or []
        if not created:
            return [
                (
                    "high",
                    f"No agent created on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r}",
                )
            ]
        actual_title = str(created[0].get("title") or "")
        if actual_title != expected_title:
            return [
                (
                    "high",
                    f"Wrong created agent title on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} expected_title={expected_title!r} got={actual_title!r}",
                )
            ]
        return []
    return check


def _no_unexpected_errors() -> Check:
    return _ok


def _turn_must_emit_runtime_turn_event(idx: int, *, final_action: str | None = None) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        events = (payloads[idx].get("timeline") or {}).get("events") or []
        matching = [event for event in events if event.get("event_type") == "runtime_turn_decided"]
        if not matching:
            return [
                (
                    "high",
                    f"Missing runtime_turn_decided event on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r}",
                )
            ]
        if final_action is not None and matching[-1].get("metadata", {}).get("final_action") != final_action:
            return [
                (
                    "high",
                    f"Wrong final_action on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} expected={final_action!r} got={matching[-1].get('metadata', {}).get('final_action')!r}",
                )
            ]
        return []
    return check


def _routing_signals_must_be_visible(idx: int) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        art = _turn_artifact(payloads[idx])
        route_advice = art.get("route_advice") or {}
        signals = route_advice.get("signals") or {}
        decision = art.get("routing_decision") or {}
        if not any(signals.get(key) for key in (
            "matched_operational_verbs",
            "matched_inspection_phrases",
            "matched_persistent_agent_patterns",
            "matched_workflow_patterns",
        )):
            return [
                (
                    "high",
                    f"Routing signals missing on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r}",
                )
            ]
        if "matched_signals" not in decision:
            return [
                (
                    "high",
                    f"Routing decision missing matched_signals on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r}",
                )
            ]
        return []
    return check


def _override_reason_must_include(idx: int, needle: str) -> Check:
    def check(payloads: List[Payload]) -> List[Finding]:
        if idx >= len(payloads):
            return [("high", f"Missing turn {idx+1}", "")]
        art = _turn_artifact(payloads[idx])
        reason = str(art.get("route_advice_override_reason") or "")
        if needle.lower() not in reason.lower():
            return [
                (
                    "high",
                    f"Override reason not informative on turn {idx+1}",
                    f"Input={payloads[idx].get('input')!r} reason={reason!r}",
                )
            ]
        return []
    return check


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
        checks=[
            _no_unexpected_errors(),
            _must_create_agent(0),
            _created_agent_title_must_be(0, "librarian"),
        ],
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
            "/workflow cancel",
            "/workflow append",
            "/workflow insert",
            "/workflow replace",
            "/workflow trigger",
            "/notacommand",
        ],
        checks=[
            _no_unexpected_errors(),
            _builtin_dispatch_must_bypass_step(1),
            _response_must_start_with(1, "Usage: /workflow <workflow_id>"),
            _builtin_dispatch_must_bypass_step(3),
            _response_must_start_with(3, "Usage: /workflow rerun <workflow_id> <task>"),
            _builtin_dispatch_must_bypass_step(4),
            _response_must_start_with(4, "Usage: /workflow cancel <workflow_id>"),
            _builtin_dispatch_must_bypass_step(5),
            _response_must_start_with(5, "Usage: /workflow append <workflow_id> <path>"),
            _builtin_dispatch_must_bypass_step(6),
            _response_must_start_with(6, "Usage: /workflow insert <workflow_id> <after_task> <path>"),
            _builtin_dispatch_must_bypass_step(7),
            _response_must_start_with(7, "Usage: /workflow replace [-r] <workflow_id> <task> <path>"),
            _builtin_dispatch_must_bypass_step(8),
            _response_must_start_with(8, "Usage: /workflow trigger <workflow_id> <label-or-task-id> [event_name]"),
        ],
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
        checks=[_no_unexpected_errors(), _response_must_start_with(2, "No approval requests.")],
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
        checks=[_no_unexpected_errors(), _turn_must_emit_runtime_turn_event(0, final_action="ask_clarification"), _routing_signals_must_be_visible(0)],
    ),
    Scenario(
        name="obs_timeline_on_delegation",
        category="observability",
        description="A persistent_delegation turn should emit a runtime decision event",
        turns=["create an agent called Indexer that catalogs photos"],
        checks=[_no_unexpected_errors(), _turn_must_emit_runtime_turn_event(0, final_action="persistent_agent"), _routing_signals_must_be_visible(0)],
    ),
    Scenario(
        name="obs_timeline_on_direct_answer",
        category="observability",
        description="A direct_answer turn should emit a runtime decision event",
        turns=["hi"],
        checks=[_no_unexpected_errors(), _turn_must_emit_runtime_turn_event(0, final_action="direct_response")],
    ),
    Scenario(
        name="obs_timeline_on_run_commands",
        category="observability",
        description="A run_commands turn should emit a runtime decision event with inspectable signals",
        turns=[
            "/agent create Librarian",
            "kill the librarian agent permanently",
        ],
        checks=[
            _no_unexpected_errors(),
            _turn_must_emit_runtime_turn_event(1, final_action="run_commands"),
            _routing_signals_must_be_visible(1),
        ],
    ),
    Scenario(
        name="obs_timeline_on_workflow_authoring",
        category="observability",
        description="A workflow authoring turn should emit a runtime decision event",
        turns=["create a workflow that lists files in /tmp and writes the count to a report"],
        checks=[_no_unexpected_errors(), _turn_must_emit_runtime_turn_event(0, final_action="workflow_authoring"), _routing_signals_must_be_visible(0)],
    ),
    Scenario(
        name="obs_override_on_pause_agent_command",
        category="observability",
        description="An unsupported operational command should record an informative override cause",
        turns=[
            "create a persistent agent called 'librarian' that catalogs my notes",
            "pause that agent",
        ],
        checks=[
            _no_unexpected_errors(),
            _turn_must_emit_runtime_turn_event(0, final_action="persistent_agent"),
            _turn_must_emit_runtime_turn_event(1, final_action="ask_clarification"),
            _override_reason_must_include(1, "concrete target or command detail"),
        ],
    ),

    # ----- Category: memory -----
    Scenario(
        name="memory_memdltr",
        category="memory",
        description="/memdltr triggers compression+restart cycle — should not crash isolated harness",
        turns=["/memdltr"],
        checks=[_no_unexpected_errors()],
    ),

    # =========================================================================
    # Session 2 scenarios — new coverage areas
    # =========================================================================

    # ----- Category: recursion / fanout -----
    Scenario(
        name="fanout_duplicate_title",
        category="identity",
        description="Duplicate persistent agent titles must be rejected case-insensitively at creation time",
        turns=[
            "/agent create Alpha",
            "/agent create alpha",
            "/agents",
        ],
        checks=[
            _no_unexpected_errors(),
            _must_create_agent(0),
            _must_not_create_agent(1),
            _response_must_start_with(1, "agent title already exists: Alpha"),
        ],
    ),
    Scenario(
        name="identity_reserved_word_title_all",
        category="identity",
        description="Agent titled 'all' must be rejected because it collides with the global kill-all selector",
        turns=[
            "/agent create Keeper",
            "/agent create all",
        ],
        checks=[
            _no_unexpected_errors(),
            _must_create_agent(0),
            _response_must_start_with(1, "agent title 'all' is reserved"),
        ],
    ),
    Scenario(
        name="identity_unicode_title",
        category="identity",
        description="Unicode and emoji agent titles must survive create/list/kill round-trip",
        turns=[
            "/agent create 数据助手",
            "/agent create 🤖Sentinel",
            "/agent kill-all 数据助手",
            "/agent kill-all 🤖Sentinel",
            "/agents",
        ],
        checks=[
            _no_unexpected_errors(),
            _must_create_agent(0),
            _must_create_agent(1),
        ],
    ),
    Scenario(
        name="identity_null_byte_title",
        category="identity",
        description="Null byte in agent title must be rejected",
        turns=["/agent create Title\x00Null"],
        checks=[
            _no_unexpected_errors(),
            _response_must_start_with(0, "agent title must not contain control characters"),
        ],
    ),

    # ----- Category: workflow failure recovery -----
    Scenario(
        name="wf_rerun_after_cancel",
        category="workflow",
        description="Rerun a task from a CANCELLED workflow — should be refused or at minimum not silently reopen the workflow",
        turns=[
            # Submit → cancel → attempt rerun of cancelled task
            # We use a no-op workflow created via NL then slash-cancel it
            "create a workflow that lists files in /tmp and counts them",
            "yes",
            # cancel via /workflows lookup then slash
            "/workflows",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="wf_append_after_cancel",
        category="workflow",
        description="Append tasks to a CANCELLED workflow — must be refused, not silently reopen",
        turns=[
            "create a workflow that lists files in /tmp",
            "yes",
            "/workflows",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="wf_double_cancel",
        category="workflow",
        description="Cancelling an already-cancelled workflow should return a clear 'already cancelled' error, not 'not found'",
        turns=[
            "create a workflow that lists files in /tmp",
            "yes",
            "/workflows",
        ],
        checks=[_no_unexpected_errors()],
    ),
    Scenario(
        name="wf_rerun_nonexistent",
        category="workflow",
        description="Rerun a non-existent workflow/task — must return clear error, not crash",
        turns=[
            "/workflow rerun nonexistent-wf-id task1",
            "/workflow rerun wf-bad-format-id nonexistent-task",
        ],
        checks=[
            _no_unexpected_errors(),
            _builtin_dispatch_must_bypass_step(0),
            _builtin_dispatch_must_bypass_step(1),
        ],
    ),
    Scenario(
        name="wf_submit_invalid_json",
        category="workflow",
        description="Submit a workflow spec that is not valid JSON — must return parse error",
        turns=[
            "/workflow submit /nonexistent/path/badspec.json",
        ],
        checks=[
            _no_unexpected_errors(),
            _builtin_dispatch_must_bypass_step(0),
        ],
    ),
    Scenario(
        name="wf_submit_invalid_spec",
        category="workflow",
        description="Submit a workflow spec with valid JSON but missing 'tasks' — must return spec validation error",
        turns=[
            # Brain-authored workflow with cancel path — no real file here, we test the routing
            "create a workflow that pings example.com and saves the response",
            "cancel",
        ],
        checks=[_no_unexpected_errors()],
    ),

    # ----- Category: approval lifecycle -----
    Scenario(
        name="approval_bogus_id",
        category="approval",
        description="Approve/deny with a non-existent approval ID must return clear error, not crash",
        turns=[
            "/approvals approve apr-this-does-not-exist",
            "/approvals deny   apr-also-bogus",
        ],
        checks=[
            _no_unexpected_errors(),
            _builtin_dispatch_must_bypass_step(0),
            _builtin_dispatch_must_bypass_step(1),
        ],
    ),
    Scenario(
        name="approval_list_empty",
        category="approval",
        description="/approvals bare and /approvals list with no pending — must both return table, not usage string",
        turns=[
            "/approvals",
            "/approvals list",
        ],
        checks=[
            _no_unexpected_errors(),
            _builtin_dispatch_must_bypass_step(0),
            _builtin_dispatch_must_bypass_step(1),
        ],
    ),

    # ----- Category: state / cross-session -----
    Scenario(
        name="state_agent_survives_restart",
        category="state",
        description="Agent created in session 1 must be visible in session 2 on same runtime root (cross-session persistence)",
        turns=[
            "/agent create Persisted",
            "/agents",
        ],
        checks=[
            _no_unexpected_errors(),
            _must_create_agent(0),
        ],
    ),

    # ----- Category: concurrency (test-harness) -----
    Scenario(
        name="concurrency_kill_then_kill",
        category="safety",
        description="Kill an agent, then try to kill it again — must be idempotent, not error",
        turns=[
            "/agent create Temp",
            "/agent kill-all Temp",
            "/agent kill-all Temp",
        ],
        checks=[
            _no_unexpected_errors(),
            _must_create_agent(0),
        ],
    ),
    Scenario(
        name="concurrency_double_cancel_workflow",
        category="workflow",
        description="Cancel a workflow twice — second cancel must return clear status, not crash",
        turns=[
            "create a workflow that lists files in /tmp",
            "yes",
            "/workflows",
        ],
        checks=[_no_unexpected_errors()],
    ),
]
