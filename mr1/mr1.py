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

Facade
------
The `MR1` class itself, its module-level constants, and `TestAgentRecord`
now live in `mr1.orchestrator.root`. This module remains as a facade so
that:
  - `from mr1.mr1 import MR1, StateManager, MR1Process, ...` keeps working
    for the 14+ test files and production callers that depend on it.
  - `monkeypatch.setattr(mr1_module, "StateManager", ...)` continues to
    affect `MR1.__init__` (via `_resolve_state_manager()` in `root`).
  - `@patch("mr1.mr1.MRnRunRunner.run")` / `MRnStepRunner.step` /
    `MR1.start` still resolve, because the same class objects are
    re-exported here.
"""

import sys
from pathlib import Path

# Ensure mr1.core is importable when run as `python -m mr1.mr1` or
# `python mr1/mr1.py`. Kept on the facade so the entry-point path
# preserves prior behavior for direct invocation.
_PKG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PKG_ROOT.parent))

# Identity helpers — imported by tests as `from mr1.mr1 import _load_agent_config, _generate_task_id`.
from mr1.orchestrator.identity import (
    _compact_text as _compact_text,
    _generate_task_id as _generate_task_id,
    _load_agent_config as _load_agent_config,
    _now_iso as _now_iso,
)

# Orchestrator prompt — imported by tests as `from mr1.mr1 import _ORCHESTRATOR_PROMPT`.
from mr1.orchestrator.prompts import _ORCHESTRATOR_PROMPT as _ORCHESTRATOR_PROMPT

# Process + state — both have direct importers and patch seams.
# `StateManager` is monkey-patched in tests/test_inbox_triage.py and resolved
# lazily in MR1.__init__ via `mr1.orchestrator.root._resolve_state_manager()`.
from mr1.orchestrator.process import MR1Process as MR1Process
from mr1.orchestrator.state import (
    _MAX_CONVERSATION as _MAX_CONVERSATION,
    _MAX_DECISIONS as _MAX_DECISIONS,
    _STATE_PATH as _STATE_PATH,
    _TERMINAL_TASK_STATUSES as _TERMINAL_TASK_STATUSES,
    StateManager as StateManager,
)

# MRn step/run classes — kept here so `@patch("mr1.mr1.MRnRunRunner.run")`
# and `@patch("mr1.mr1.MRnStepRunner.step")` resolve.
from mr1.mrn_loop import MRnStepRunner as MRnStepRunner
from mr1.mrn_run import MRnRunRunner as MRnRunRunner

# Class + constants + helpers from the root implementation.
from mr1.orchestrator.root import (
    MR1 as MR1,
    TestAgentRecord as TestAgentRecord,
    _AGENTS_DIR as _AGENTS_DIR,
    _CONTEXT_PATH as _CONTEXT_PATH,
    _MR1_CONFIG_PATH as _MR1_CONFIG_PATH,
    _MRN_CONFIG_PATH as _MRN_CONFIG_PATH,
    _KAZI_CONFIG_PATH as _KAZI_CONFIG_PATH,
    _OUTPUT_TRUNCATE_LINES as _OUTPUT_TRUNCATE_LINES,
    _MAX_DELEGATION_ROUNDS as _MAX_DELEGATION_ROUNDS,
    _MAX_OBSERVATION_ROUNDS as _MAX_OBSERVATION_ROUNDS,
    _TEST_AGENT_MAX_HEIGHT as _TEST_AGENT_MAX_HEIGHT,
    _TEST_AGENT_PREFIX as _TEST_AGENT_PREFIX,
    _GROUNDING_AGENT_LIMIT as _GROUNDING_AGENT_LIMIT,
    _GROUNDING_WORKFLOW_LIMIT as _GROUNDING_WORKFLOW_LIMIT,
    _GROUNDING_MESSAGE_LIMIT as _GROUNDING_MESSAGE_LIMIT,
    _GROUNDING_APPROVAL_LIMIT as _GROUNDING_APPROVAL_LIMIT,
    _GROUNDING_EVENT_LIMIT as _GROUNDING_EVENT_LIMIT,
    _GROUNDING_MESSAGE_BODY_LIMIT as _GROUNDING_MESSAGE_BODY_LIMIT,
    _GROUNDING_MESSAGE_TRUNCATION_SUFFIX as _GROUNDING_MESSAGE_TRUNCATION_SUFFIX,
    _AGENT_ID_PATTERN as _AGENT_ID_PATTERN,
    _WORKFLOW_ID_PATTERN as _WORKFLOW_ID_PATTERN,
    _TASK_ID_PATTERN as _TASK_ID_PATTERN,
    _MESSAGE_ID_PATTERN as _MESSAGE_ID_PATTERN,
    _APPROVAL_ID_PATTERN as _APPROVAL_ID_PATTERN,
    _REPLY_INTENT_PATTERN as _REPLY_INTENT_PATTERN,
    _WORKFLOW_INSPECTION_INTENT_PHRASES as _WORKFLOW_INSPECTION_INTENT_PHRASES,
    _WORKFLOW_FINDINGS_INTENT_PHRASES as _WORKFLOW_FINDINGS_INTENT_PHRASES,
    _WORKFLOW_AUTHORING_SUPPRESSION_PHRASES as _WORKFLOW_AUTHORING_SUPPRESSION_PHRASES,
    _WORKFLOW_REFERENCE_ALIASES as _WORKFLOW_REFERENCE_ALIASES,
    _AGENT_REFERENCE_ALIASES as _AGENT_REFERENCE_ALIASES,
    _AGENT_CREATED_REFERENCE_ALIASES as _AGENT_CREATED_REFERENCE_ALIASES,
    _AUTO_REPLY_SYNTHESIS_MARKERS as _AUTO_REPLY_SYNTHESIS_MARKERS,
    _DEFAULT_PERSISTENT_CHILD_TITLES as _DEFAULT_PERSISTENT_CHILD_TITLES,
    _BULK_AGENT_ACTIONS as _BULK_AGENT_ACTIONS,
    _FINDINGS_PREFERRED_LABEL_TOKENS as _FINDINGS_PREFERRED_LABEL_TOKENS,
    _has_workflow_pronoun_reference as _has_workflow_pronoun_reference,
    _DELEGATE_PATTERN as _DELEGATE_PATTERN,
    _PERSISTENT_DELEGATION_MARKERS as _PERSISTENT_DELEGATION_MARKERS,
    _PERSISTENT_CHILD_TITLE_PATTERNS as _PERSISTENT_CHILD_TITLE_PATTERNS,
    _META_EXPLANATION_PATTERNS as _META_EXPLANATION_PATTERNS,
    _PERSISTENT_DELEGATION_IMPERATIVE_PATTERNS as _PERSISTENT_DELEGATION_IMPERATIVE_PATTERNS,
    _DUMP_COMPLETE_SIGNAL as _DUMP_COMPLETE_SIGNAL,
    _normalize_routing_text as _normalize_routing_text,
    _truncate_grounding_message_body as _truncate_grounding_message_body,
)


def main():
    mr1 = MR1()
    mr1.run()


if __name__ == "__main__":
    main()
