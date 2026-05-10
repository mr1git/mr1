"""MRn step/run loop package.

A persistent scoped MRn agent advances by repeatedly executing one
deterministic step. Each step parses one action, executes it, and
persists the result. This package owns the action parser, executors,
and dispatch table; the runner class itself currently still lives in
`mr1.mrn_loop` and will move here in a later stage.
"""
