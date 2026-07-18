"""
Ontology naming invariant (docs/architecture/AGENT_ONTOLOGY.md §6):

    mr_level is metadata, never a substitute for a title. A title is never
    defaulted to "MR2", "MR3", or any other level-shaped string. Where a
    title is genuinely unset, the fallback is the literal string
    "Unnamed agent".

`MR1._default_child_title` rotates through a fixed set of names
(Sentinel, Darwin, Architect, Curator, Sage) when the caller/brain didn't
supply one, but its own last-resort fallback (once that rotation is
exhausted) must not regress to a literal "MR<n>" string.
"""

import re

import pytest

from mr1.mr1 import MR1
from mr1.orchestrator.state import StateManager
from mr1.worker_runner import MockRunner
from mr1.workflow_store import WorkflowStore

_GENERIC_TITLE = re.compile(r"^MR\d+$", re.IGNORECASE)


@pytest.fixture
def mr1_instance(tmp_path):
    instance = MR1(
        workflow_store=WorkflowStore(root=tmp_path / "workflows"),
        workflow_runner=MockRunner(),
        workflow_auto_tick=False,
        inbox_auto_triage=False,
    )
    instance._state = StateManager(state_path=tmp_path / "mr1_state.json")
    return instance


class TestDefaultChildTitleNamingInvariant:
    def test_rotation_names_are_never_generic(self, mr1_instance):
        for _ in range(5):
            title = mr1_instance._default_child_title()
            assert not _GENERIC_TITLE.match(title), f"generic title leaked: {title!r}"
            mr1_instance._scoped_agents.create_child_agent(
                mr1_instance._root_agent_id, title,
            )

    def test_exhausted_rotation_does_not_fall_back_to_mr_level(self, mr1_instance):
        """Create enough owners to exhaust the fixed rotation, then assert
        the next fallback title still isn't a level-shaped string."""
        for _ in range(5):
            title = mr1_instance._default_child_title()
            mr1_instance._scoped_agents.create_child_agent(
                mr1_instance._root_agent_id, title,
            )

        sixth_title = mr1_instance._default_child_title()
        assert not _GENERIC_TITLE.match(sixth_title), (
            f"rotation-exhausted fallback produced a level-shaped title: {sixth_title!r}"
        )
        assert sixth_title == "Unnamed agent"

    def test_no_persisted_agent_ever_carries_a_generic_title(self, mr1_instance):
        # 5 rotation names + 1 rotation-exhausted fallback; a 7th call would
        # collide on the now-reserved "Unnamed agent" title, which is a
        # separate (pre-existing, unrelated) uniqueness concern.
        for _ in range(6):
            title = mr1_instance._default_child_title()
            agent = mr1_instance._scoped_agents.create_child_agent(
                mr1_instance._root_agent_id, title,
            )
            assert not _GENERIC_TITLE.match(agent.title), (
                f"persisted agent {agent.agent_id} has generic title {agent.title!r}"
            )
