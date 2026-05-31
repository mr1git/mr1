"""Runtime QA harness — hostile NL-driven validation of MR1.

Drives the real MR1 runtime through `mr1.runtime_test_cli.RuntimeTestSession`
in isolated mode, captures JSON payloads per turn, and applies post-hoc
assertions to surface routing/delegation/approval/state/observability bugs.

Each scenario runs in its own isolated runtime root. Output JSONL is written
to tests/runtime_qa/results/<scenario>.jsonl with one payload per turn,
plus a `_meta.json` per scenario with high-level findings.
"""
