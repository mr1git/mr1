import pytest

from mr1.routing_advisor import build_route_advice


def _runtime_grounding(*, agents=None, workflows=None, approvals=None):
    return {
        "agents": list(agents or []),
        "workflows": list(workflows or []),
        "messages": [],
        "approvals": list(approvals or []),
        "events": [],
        "resolved_references": {},
        "ambiguities": [],
    }


def test_message_reply_routes_to_run_commands():
    advice = build_route_advice(
        "msg-20260503T044113313444-5228b5 respond with Continue with the approved implementation.",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "run_commands"


def test_approve_routes_to_run_commands():
    advice = build_route_advice(
        "approve cap_approval_test",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "run_commands"


def test_existing_workflow_findings_route_to_inspection():
    advice = build_route_advice(
        "did workflow wf-20260503T044113-ed0ba7 finish? summarize findings",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "inspect_existing_state"


def test_failed_task_route_to_inspection():
    advice = build_route_advice(
        "why did task tk-20260503T044113-ed0ba7 fail?",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "inspect_existing_state"


def test_recent_agent_block_question_routes_to_inspection():
    advice = build_route_advice(
        "why did the agent we just created get blocked after i ran it?",
        runtime_grounding=_runtime_grounding(agents=[{"agent_id": "ag-1"}]),
    )

    assert advice.route == "inspect_existing_state"


def test_named_persistent_agent_routes_correctly():
    advice = build_route_advice(
        "create a persistent agent named Sage to own self-evolution",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "persistent_agent"


def test_self_evolving_aspect_routes_to_persistent_agent():
    advice = build_route_advice(
        "self-evolving aspect not handled by MR1 directly",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "persistent_agent"


def test_create_workflow_routes_correctly():
    advice = build_route_advice(
        "create a workflow to read README and summarize",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "create_workflow"


def test_create_workflow_with_lists_verb_routes_correctly():
    advice = build_route_advice(
        "create a workflow that lists files in /tmp and writes the count to a report",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "create_workflow"


@pytest.mark.parametrize(
    "phrase",
    [
        "yes",
        "yes, submit it",
        "ok",
        "okay",
        "submit",
        "do it",
        "go ahead",
        "proceed",
    ],
)
def test_pending_workflow_confirmation_phrases_route_to_workflow_handling(phrase):
    advice = build_route_advice(
        phrase,
        runtime_grounding=_runtime_grounding(),
        pending_state={"mode": "create"},
    )

    assert advice.route == "create_workflow"


@pytest.mark.parametrize(
    "phrase",
    [
        "no",
        "cancel",
        "stop",
        "nevermind",
        "never mind",
        "abort",
        "actually nevermind, cancel that",
    ],
)
def test_pending_workflow_cancellation_phrases_route_to_workflow_handling(phrase):
    advice = build_route_advice(
        phrase,
        runtime_grounding=_runtime_grounding(),
        pending_state={"mode": "create"},
    )

    assert advice.route == "create_workflow"


def test_meta_question_routes_direct_response():
    advice = build_route_advice(
        "describe when you would use tools vs workflows vs agents",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_research_group_language_stays_direct_response():
    advice = build_route_advice(
        "I am part of a research group",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_bulk_kill_routes_to_run_commands():
    advice = build_route_advice(
        "kill all research agents",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "run_commands"


def test_how_would_you_kill_agents_is_direct_response():
    advice = build_route_advice(
        "how would you kill all research agents",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_describe_agent_replies_is_direct_response():
    advice = build_route_advice(
        "describe how agent replies should work",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_conversational_with_agent_word_routes_direct_response():
    advice = build_route_advice(
        "what do you think about this agent design?",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_planning_discussion_routes_direct_response():
    advice = build_route_advice(
        "let's plan the agent architecture",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_execution_tokens_without_operational_intent_routes_direct_response():
    advice = build_route_advice(
        "I want to understand how the MR2 delegation workflow works",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_create_descriptive_agent_routes_to_persistent_agent():
    advice = build_route_advice(
        "create a new self evolution agent named evangelion",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "persistent_agent"


def test_why_did_you_create_question_routes_direct_response():
    advice = build_route_advice(
        "why did u create a workflow instead of a persistent agent walk me thru ur thought process",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "direct_response"


def test_confirmation_without_pending_workflow_clarifies():
    advice = build_route_advice(
        "yes, submit it",
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "ask_clarification"


@pytest.mark.parametrize(
    "user_text",
    [
        "create three agents and then kill them all",
        "create an agent then delete it",
        "spawn five workers and immediately stop them",
    ],
)
def test_conflicting_create_then_destroy_routes_to_clarification(user_text):
    advice = build_route_advice(
        user_text,
        runtime_grounding=_runtime_grounding(),
    )

    assert advice.route == "ask_clarification"


@pytest.mark.parametrize(
    ("user_text", "expected_route"),
    [
        ("delete every workflow and every agent right now", "ask_clarification"),
        ("delete the old one", "ask_clarification"),
        ("delete everything", "ask_clarification"),
        ("show me pending approvals", "inspect_existing_state"),
        ("list all workflows", "inspect_existing_state"),
        ("list approvals", "inspect_existing_state"),
        ("spawn twenty agents that each watch a different file", "persistent_agent"),
        ("spawn an agent for me", "persistent_agent"),
        ("kill the archivist permanently", "run_commands"),
        ("kill the archivist agent permanently", "run_commands"),
        ("tell librarian to summarize last week's notes", "run_commands"),
        ("tell the librarian agent to summarize last week's notes", "run_commands"),
        ("rerun it", "ask_clarification"),
        ("stop that", "ask_clarification"),
        ("rename Librarian to PaperLibrarian", "run_commands"),
        ("pause that agent", "run_commands"),
        ("create three agents and then kill them all", "ask_clarification"),
        ("can you show me what's running?", "inspect_existing_state"),
        ("what workflows are running?", "inspect_existing_state"),
    ],
)
def test_runtime_qa_routing_phrases_route_as_expected(user_text, expected_route):
    advice = build_route_advice(
        user_text,
        runtime_grounding=_runtime_grounding(
            agents=[{"agent_id": "ag-1", "title": "Archivist"}, {"agent_id": "ag-2", "title": "Librarian"}],
            workflows=[{"workflow_id": "wf-1", "title": "Example", "status": "running"}],
            approvals=[{"approval_request_id": "cap_approval_test"}],
        ),
    )

    assert advice.route == expected_route


def test_route_advice_exposes_operational_and_inspection_signals():
    advice = build_route_advice(
        "show me pending approvals",
        runtime_grounding=_runtime_grounding(
            approvals=[{"approval_request_id": "cap_approval_test"}],
        ),
    )

    assert advice.signals["matched_inspection_phrases"]
    assert "show" in advice.signals["matched_inspection_phrases"]
    assert advice.signals["matched_operational_verbs"] == []


def test_route_advice_exposes_persistent_and_workflow_signals():
    persistent = build_route_advice(
        "spawn an agent for me",
        runtime_grounding=_runtime_grounding(),
    )
    workflow = build_route_advice(
        "create a workflow that lists files in /tmp and writes the count to a report",
        runtime_grounding=_runtime_grounding(),
    )

    assert "spawn/make/add/start agent imperative" in persistent.signals["matched_persistent_agent_patterns"]
    assert any(
        item.startswith("workflow create marker:")
        for item in workflow.signals["matched_workflow_patterns"]
    )
    assert "workflow action word:create" in workflow.signals["matched_workflow_patterns"]
