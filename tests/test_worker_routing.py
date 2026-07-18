"""
Regression coverage for the bounded-investigation → worker-spawn route.

MR1's root `step()` previously had no live path that ever spawned a worker:
`_delegate_to_worker` existed fully-built (including the `"spawn_worker"`
decision-log action and `task_attached`/`task_completed` events) but was
never called. These tests exercise the new `bounded_investigation` route
that makes it reachable, end to end, without spawning a real `claude`
subprocess (the underlying `subprocess.Popen` call is mocked, same recipe
as `tests/test_worker.py`).
"""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from mr1.mr1 import MR1, MR1Process, StateManager
from mr1.orchestrator.root import _BOUNDED_INVESTIGATION_WORKER_TIMEOUT_S
from mr1.worker_runner import MockRunner
from mr1.workflow_store import WorkflowStore
from mr1.routing_advisor import build_route_advice


class TestBoundedInvestigationRouting:
    """Pure routing_advisor coverage — no MR1 instance needed."""

    @pytest.mark.parametrize("text", [
        "Investigate this repository.",
        "Take a look through this repo and tell me what seems fragile.",
        "Figure out why these tests are flaky.",
        "Take a look through this repo.",
        "See if anything here worries you.",
        "Can you check whether the scheduler is dropping ticks?",
    ])
    def test_bounded_investigation_prompts_route_correctly(self, text):
        advice = build_route_advice(text)
        assert advice.route == "bounded_investigation"

    def test_ownership_language_still_wins_over_investigation_wording(self):
        advice = build_route_advice(
            "Keep an eye on this codebase — make an agent to own that."
        )
        assert advice.route == "orchestrator_ownership"

    def test_followup_question_stays_direct_response(self):
        advice = build_route_advice("What did you find?")
        assert advice.route == "direct_response"

    def test_operational_commands_are_not_shadowed(self):
        assert build_route_advice("Pause the runtime for now.").route == "run_commands"
        assert build_route_advice("Please stop that agent.").route == "run_commands"
        assert build_route_advice(
            "Delete every workflow and every agent right now."
        ).route == "ask_clarification"


@pytest.fixture
def mr1_instance(tmp_path):
    instance = MR1(
        workflow_store=WorkflowStore(root=tmp_path / "workflows"),
        workflow_runner=MockRunner(),
        workflow_auto_tick=False,
        inbox_auto_triage=False,
    )
    instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
    instance._process = MagicMock(spec=MR1Process)
    instance._process.alive = True
    instance._process.send.return_value = (
        "MR1 investigated the repo as delegated and here is a grounded summary of "
        "what the worker found."
    )
    return instance


def _mock_popen(output_text: str, *, is_error: bool = False, returncode: int = 0):
    payload = json.dumps({"result": output_text, "is_error": is_error})
    proc = MagicMock()
    proc.pid = 4242
    proc.communicate.return_value = (payload.encode(), b"")
    proc.returncode = returncode
    return proc


def _mock_popen_timeout(partial_text: str = ""):
    """Simulates a worker subprocess that hangs past its time budget: the
    first `communicate()` raises TimeoutExpired (worker.py's main wait),
    the second (the post-kill drain) returns whatever partial bytes were
    flushed before the kill."""
    proc = MagicMock()
    proc.pid = 4242
    proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="claude", timeout=_BOUNDED_INVESTIGATION_WORKER_TIMEOUT_S),
        (partial_text.encode(), b""),
    ]
    return proc


class TestBoundedInvestigationEndToEnd:
    @patch("mr1.core.spawner.subprocess.Popen")
    def test_worker_spawn_is_observable_in_decisions_and_events(self, mock_popen, mr1_instance):
        mock_popen.return_value = _mock_popen("found 3 TODOs and one flaky test fixture")

        response = mr1_instance.step("Take a look through this repo and tell me what seems fragile.")

        assert response == mr1_instance._process.send.return_value

        decisions = mr1_instance._state.conversation  # sanity: conversation recorded
        assert decisions  # user + mr1 turns recorded

        # The authoritative detection source: StateManager's decision log.
        # (`_handle_task_event`'s task_attached/task_completed events only
        # reach `self._event_log` when an external `event_sink` is wired in
        # — true of every delegation type today, not something this route
        # changes — so the decision log is the source of truth here.)
        spawn_decisions = [
            d for d in mr1_instance._state._state["decisions"]
            if d["action"] == "spawn_worker"
        ]
        assert len(spawn_decisions) == 1

        # No AgentRecord was persisted — a worker is ephemeral, never an owner.
        assert mr1_instance._scoped_agents.list_agents() == [
            mr1_instance._scoped_agents.require_agent(mr1_instance._root_agent_id)
        ]

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_worker_failure_is_not_papered_over(self, mock_popen, mr1_instance):
        mock_popen.return_value = _mock_popen("", is_error=False, returncode=1)

        mr1_instance.step("Investigate this repository.")

        # The synthesis prompt sent to the brain must carry the worker's
        # real (failed) status, not a silently-invented success.
        sent_prompt = mr1_instance._process.send.call_args[0][0]
        assert "[WORKER" in sent_prompt
        assert "do not invent findings" in sent_prompt.lower()

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_response_is_the_brain_synthesis_not_raw_worker_output(self, mock_popen, mr1_instance):
        mock_popen.return_value = _mock_popen("raw worker text nobody should see verbatim")

        response = mr1_instance.step("Figure out why these tests are flaky.")

        assert response == mr1_instance._process.send.return_value
        assert "raw worker text nobody should see verbatim" not in response


class TestBoundedWorkerExecutionEnvelope:
    """Problem #2 follow-up: workers must be bounded (short timeout,
    scope/traversal guardrails) and must surface partial results honestly
    instead of hanging or silently discarding whatever they did manage to
    produce before being cut off."""

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_worker_is_given_a_short_bounded_timeout_not_the_300s_default(
        self, mock_popen, mr1_instance,
    ):
        mock_popen.return_value = _mock_popen("quick findings")

        mr1_instance.step("Take a look through this repo.")

        # worker.py's job_timeout is passed straight through to
        # subprocess.communicate(timeout=...) — assert it's the short,
        # investigation-specific bound, not the 300s worker-wide default.
        assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == (
            _BOUNDED_INVESTIGATION_WORKER_TIMEOUT_S
        )

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_timed_out_worker_preserves_partial_output_for_synthesis(
        self, mock_popen, mr1_instance,
    ):
        mock_popen.return_value = _mock_popen_timeout("partial: found 2 of an unknown total")

        mr1_instance.step("Take a look through this repo.")

        sent_prompt = mr1_instance._process.send.call_args[0][0]
        assert "[WORKER TIMEOUT]" in sent_prompt
        assert "Partial output captured before timeout" in sent_prompt
        assert "partial: found 2 of an unknown total" in sent_prompt
        # Distinguishes a partial timeout from a hard failure in the
        # synthesis instructions given to the brain.
        assert "partial" in sent_prompt.lower()

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_timed_out_worker_with_no_partial_output_says_so_plainly(
        self, mock_popen, mr1_instance,
    ):
        mock_popen.return_value = _mock_popen_timeout("")

        mr1_instance.step("Take a look through this repo.")

        sent_prompt = mr1_instance._process.send.call_args[0][0]
        assert "[WORKER TIMEOUT]" in sent_prompt
        # The *dynamic* partial-output section (built only when output is
        # non-empty) must be absent — not to be confused with the static
        # instruction text describing what to do *if* it were present.
        assert "Partial output captured before timeout:" not in sent_prompt
        assert "didn't finish in time" in sent_prompt

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_worker_instructions_include_traversal_scope_guardrail(
        self, mock_popen, mr1_instance,
    ):
        mock_popen.return_value = _mock_popen("findings")

        mr1_instance.step("Take a look through this repo.")

        spawned_cmd = mock_popen.call_args[0][0]
        prompt_arg = spawned_cmd[spawned_cmd.index("-p") + 1]
        assert ".venv" in prompt_arg
        assert "bounded" in prompt_arg.lower() or "focused pass" in prompt_arg.lower()

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_synthesis_prompt_asks_for_brief_disclosure(self, mock_popen, mr1_instance):
        mock_popen.return_value = _mock_popen("some findings")

        mr1_instance.step("Take a look through this repo.")

        sent_prompt = mr1_instance._process.send.call_args[0][0]
        assert "worker" in sent_prompt.lower()
        assert "bounded" in sent_prompt.lower()
