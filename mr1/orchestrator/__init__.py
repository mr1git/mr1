"""Generalized MRn / MR1 orchestrator package.

MR1 is the root orchestrator (MRn at mr_level=1); deeper-level MRn
agents share the same loop, action handlers, and runtime grounding via
this package. Concrete root-specific behavior — the `MR1` class itself,
`TestAgentRecord`, the orchestrator-only constants, and small routing
helpers — lives in `root`. `mr1.mr1` is a facade that re-exports the
public surface for historical importers and monkeypatch seams.
"""
