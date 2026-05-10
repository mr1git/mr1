"""Generalized MRn / MR1 orchestrator package.

MR1 is the root orchestrator (MRn at tree_level=1); deeper-level MRn
agents share the same loop, action handlers, and runtime grounding via
this package. Concrete root-specific behavior lives in `root` (added in
later refactor stages).
"""
