"""
Phase D — Partner Behavior QA.

Where the hierarchical soak (`tests/soak/hierarchical/`) and runtime QA
(`tests/runtime_qa/`) prove MR1 is *mechanically* correct — a turn's outcome
falls inside an allowed set, nothing crashes, invariants hold — this package
asks a different question: is MR1 a *good partner* to talk to? Does it show
the right amount of initiative, avoid manufacturing busywork, recognize when
something needs a standing owner versus a bounded pass, and would Marwan
specifically enjoy the interaction?

That question has no deterministic answer, so this harness drives a corpus
of realistic, often deliberately ambiguous prompts against a real MR1 (real
`claude` brain) and has a second, independent `claude` call judge the
transcript against Marwan's stated philosophy plus his own documented
behavioral feedback.

See `python -m tests.behavior_qa --help`.
"""
