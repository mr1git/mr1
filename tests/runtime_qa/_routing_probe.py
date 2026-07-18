"""Probe build_route_advice directly — no LLM, just the deterministic router.

Confirms the regex/lexical routing bugs without depending on LLM behavior.
"""
from mr1.routing_advisor import build_route_advice

cases = [
    ("delete every workflow and every agent right now", "ask_clarification", None),
    ("delete the old one", "ask_clarification", None),
    ("delete everything", "ask_clarification", None),
    ("show me pending approvals", "inspect_existing_state", None),
    ("list all workflows", "inspect_existing_state", None),
    ("list approvals", "inspect_existing_state", None),
    ("spawn twenty agents that each watch a different file", "orchestrator_ownership", None),
    ("spawn an agent for me", "orchestrator_ownership", None),
    ("kill the archivist permanently", "run_commands", None),
    ("kill the archivist agent permanently", "run_commands", None),
    ("tell librarian to summarize last week's notes", "run_commands", None),
    ("tell the librarian agent to summarize last week's notes", "run_commands", None),
    ("yes, submit it", "create_workflow", {"mode": "create"}),
    ("yes", "create_workflow", {"mode": "create"}),
    ("rerun it", "ask_clarification", None),
    ("stop that", "ask_clarification", None),
    ("rename Librarian to PaperLibrarian", "run_commands", None),
    ("pause that agent", "run_commands", None),
    ("create three agents and then kill them all", "ask_clarification", None),
    ("can you show me what's running?", "inspect_existing_state", None),
    ("what workflows are running?", "inspect_existing_state", None),
]

mismatches = 0

for text, expected_route, pending_state in cases:
    a = build_route_advice(text, pending_state=pending_state)
    ok = a.route == expected_route
    mismatches += 0 if ok else 1
    status = "ok" if ok else "mismatch"
    print(f"{status:9} {a.route:24} conf={a.confidence:.2f} expected={expected_route:22} {text!r}")

print(f"\nMismatches: {mismatches}/{len(cases)}")
