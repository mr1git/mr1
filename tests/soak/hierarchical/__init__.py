"""
Hierarchical-autonomy conversation soak.

This package drives the *real* MR1 natural-language runtime path
(`MR1.step(...)` via the same session shape the runtime-QA runner uses)
across a long, stateful, Marwan-style conversation, and asserts that MR1
makes sensible orchestration decisions — respond, clarify, create a
workflow, create/reuse an orchestrator, delegate hierarchically,
message collaborators, recover, escalate — without runaway hierarchy or
degraded runtime health.

It is deliberately *not* a benchmark of low-level method calls. The harness
never calls `create_child_agent`, `submit_workflow`, or `send_message`
directly to "prove they work"; it speaks to MR1 in human language and
observes what MR1 chose to do.

Modules
-------
* ``fixture``    — a disposable synthetic repo for MR1 to inspect
* ``fakes``      — a scripted brain + compiler for quick fake-planner runs
* ``outcomes``   — per-turn orchestration-decision classification + assertions
* ``arcs``       — the stateful conversation corpus (10 project arcs)
* ``invariants`` — hierarchical + runtime-health invariants
* ``driver``     — the session wrapper, scheduler drain, sampling, checkpoint
* ``report``     — the human-readable conversation-quality review
* ``__main__``   — the CLI (fake / real / report modes)
"""

from __future__ import annotations

__all__ = [
    "fixture",
    "fakes",
    "outcomes",
    "arcs",
    "invariants",
    "driver",
    "report",
]
