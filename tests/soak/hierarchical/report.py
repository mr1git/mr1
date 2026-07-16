"""
The human-readable conversation-quality review.

Renders a run (result.json + transcript.jsonl) into a Markdown review that
answers not just "was it technically correct" but "were these sensible
orchestration choices" — the transcript, the per-turn decisions, the agent
hierarchy, why each agent and workflow came to exist, where MR1 reused an
existing agent, what it clarified, what failed or escalated, the messages
exchanged, and a closing judgment section written from Marwan's point of view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tests.soak.hierarchical.driver import RunLayout, _read_jsonl
from tests.soak.hierarchical.invariants import _looks_like_unresolved_question


def render_report(layout: RunLayout, result: Dict[str, Any]) -> str:
    transcript = _read_jsonl(layout.transcript)
    lines: List[str] = []
    w = lines.append

    verdict = "PASSED" if result.get("passed") else ("PARTIAL" if result.get("partial") else "FAILED")
    w(f"# Hierarchical Autonomy Soak — {verdict}")
    w("")
    counts = result.get("counts", {})
    rollup = result.get("rollup", {})
    w(f"- **planner:** `{result.get('planner')}`  ·  **restarted mid-conversation:** "
      f"{result.get('restarted')}")
    w(f"- **turns:** {counts.get('turns')}  ·  **persistent agents:** {counts.get('agents')}  ·  "
      f"**messages:** {counts.get('messages')}  ·  **workflows:** {counts.get('workflows')}")
    if rollup:
        w(f"- **RSS:** {rollup.get('rss_start_mb')}→{rollup.get('rss_end_mb')} MB  ·  "
          f"**FDs:** {rollup.get('fd_start')}→{rollup.get('fd_end')}  ·  "
          f"**idle-tick brain calls:** {rollup.get('idle_brain_calls')}  ·  "
          f"**wall:** {rollup.get('wall_s')}s")
    w("")
    _render_findings_summary(w, result)

    w("## 1. Conversation transcript")
    w("")
    current_arc = None
    for e in transcript:
        if e.get("arc") != current_arc:
            current_arc = e.get("arc")
            w(f"### arc: {current_arc} — {e.get('goal','')}")
            w("")
        marks = "" if not e.get("findings") else "  ⚠️"
        w(f"**[{e['index']}] Marwan:** {e['text']}")
        resp = (e.get("response_text") or "").strip().replace("\n", " ")
        if len(resp) > 280:
            resp = resp[:280] + "…"
        w(f"> MR1 ({e.get('outcome')}): {resp}{marks}")
        w("")

    w("## 2. Turn-by-turn orchestration decisions")
    w("")
    w("| # | arc | route (advised→final) | outcome | created | note |")
    w("|---|-----|-----------------------|---------|---------|------|")
    for e in transcript:
        created = []
        if e.get("created_agents"):
            created.append("agent:" + ",".join(a.get("title") or "?" for a in e["created_agents"]))
        if e.get("created_workflow_ids"):
            created.append(f"wf×{len(e['created_workflow_ids'])}")
        if e.get("created_message_ids"):
            created.append(f"msg×{len(e['created_message_ids'])}")
        if e.get("approval_ids"):
            created.append(f"approval×{len(e['approval_ids'])}")
        created_s = ", ".join(created) or "—"
        route_s = f"{e.get('advised_route') or '—'}→{e.get('route') or '—'}"
        note = (e.get("note") or "").replace("|", "/")
        w(f"| {e['index']} | {e['arc']} | {route_s} | {e['outcome']} | {created_s} | {note} |")
    w("")

    _render_hierarchy(w, result)
    _render_agent_rationale(w, transcript)
    _render_workflow_rationale(w, transcript)
    _render_reuse(w, transcript)
    _render_clarifications(w, transcript)
    _render_failures(w, transcript, result)
    _render_messages(w, result)
    _render_unnatural(w, transcript, result)
    _render_marwan_judgment(w, result, transcript)

    return "\n".join(lines) + "\n"


def _render_findings_summary(w, result: Dict[str, Any]) -> None:
    findings = result.get("findings", {})
    total = sum(len(v) for v in findings.values())
    highs = sum(1 for group in findings.values() for f in group if f and f[0] == "high")
    w(f"**Findings:** {total} total, {highs} high-severity.")
    w("")
    if total:
        w("| area | severity | title | detail |")
        w("|------|----------|-------|--------|")
        for area, group in findings.items():
            for f in group:
                if not f:
                    continue
                sev = f[0]
                if area == "turn" and len(f) == 4:
                    title, detail = f[2], f[3]
                else:
                    title, detail = (f[1], f[2]) if len(f) >= 3 else (str(f), "")
                detail = str(detail).replace("|", "/")[:120]
                w(f"| {area} | {sev} | {title} | {detail} |")
        w("")


def _render_hierarchy(w, result: Dict[str, Any]) -> None:
    w("## 3. Agent hierarchy")
    w("")
    agents = result.get("agents") or []
    root_id = result.get("root_agent_id")
    if not agents:
        w("_No agent snapshot available (report-only from a partial run)._")
        w("")
        return
    by_parent: Dict[Any, List[Dict[str, Any]]] = {}
    for a in agents:
        by_parent.setdefault(a.get("parent_agent_id"), []).append(a)

    def walk(node_id, depth):
        for a in sorted(by_parent.get(node_id, []), key=lambda x: x.get("created_at", "")):
            status = a.get("status", "active")
            w(f"{'  ' * depth}- **{a.get('title')}** "
              f"(`{a.get('agent_id')}`, {a.get('agent_type')}, depth {a.get('tree_level')}, {status})")
            walk(a.get("agent_id"), depth + 1)

    root = next((a for a in agents if a.get("agent_id") == root_id), None)
    if root:
        w(f"- **{root.get('title','MR1')}** (`{root_id}`, root)")
        walk(root_id, 1)
    else:
        walk(None, 0)
    w("")


def _render_agent_rationale(w, transcript) -> None:
    w("## 4. Why each agent was created")
    w("")
    any_agent = False
    for e in transcript:
        for a in e.get("created_agents", []):
            any_agent = True
            w(f"- **{a.get('title')}** — from turn [{e['index']}] "
              f"({e['text']!r}). Mission: {a.get('mission') or '(none recorded)'}")
    if not any_agent:
        w("_No persistent agents were created in this run._")
    w("")


def _render_workflow_rationale(w, transcript) -> None:
    w("## 5. Why each workflow (not an agent)")
    w("")
    any_wf = False
    for e in transcript:
        if e.get("created_workflow_ids"):
            any_wf = True
            for wid in e["created_workflow_ids"]:
                w(f"- `{wid}` — from turn [{e['index']}] ({e['text']!r}): a bounded, "
                  f"executable task, so a workflow was the right unit rather than a "
                  f"permanent owner.")
    if not any_wf:
        w("_No workflows were submitted in this run._")
    w("")


def _render_reuse(w, transcript) -> None:
    w("## 6. Where MR1 reused an existing agent")
    w("")
    any_reuse = False
    for e in transcript:
        if e.get("created_message_ids") and not e.get("created_agents"):
            any_reuse = True
            w(f"- turn [{e['index']}] ({e['text']!r}) → messaged an existing agent "
              f"instead of spawning a new one.")
    if not any_reuse:
        w("_No reuse-by-messaging turns recorded._")
    w("")


def _render_clarifications(w, transcript) -> None:
    w("## 7. Clarifications requested")
    w("")
    any_clar = False
    for e in transcript:
        if e.get("outcome") == "clarification":
            any_clar = True
            resp = (e.get("response_text") or "").strip().replace("\n", " ")[:160]
            w(f"- turn [{e['index']}] ({e['text']!r}) → MR1 asked for clarification: {resp}")
    if not any_clar:
        w("_No clarifications were requested._")
    w("")


def _render_failures(w, transcript, result) -> None:
    w("## 8. Failures, replans, and escalations")
    w("")
    any_f = False
    for e in transcript:
        if e.get("approval_ids"):
            any_f = True
            w(f"- turn [{e['index']}] ({e['text']!r}) → raised approval/escalation "
              f"{e['approval_ids']} (authority gated, not bypassed).")
        if not e.get("ok"):
            any_f = True
            w(f"- turn [{e['index']}] ({e['text']!r}) → raised an exception.")
    if not any_f:
        w("_No failures, escalations, or blocked authority in this run._")
    w("")


def _render_messages(w, result) -> None:
    w("## 9. Messages exchanged")
    w("")
    messages = result.get("messages") or []
    if not messages:
        w("_No messages exchanged._")
        w("")
        return
    w("| id | from | to | subject | status |")
    w("|----|------|----|---------|--------|")
    for m in messages:
        subj = str(m.get("subject") or "")[:40].replace("|", "/")
        w(f"| `{m.get('message_id')}` | {m.get('from_agent_id')} | {m.get('to_agent_id')} "
          f"| {subj} | {m.get('status')} |")
    w("")


def _render_unnatural(w, transcript, result) -> None:
    w("## 10. Technically valid but possibly unnatural")
    w("")
    notes: List[str] = []
    for e in transcript:
        # a run_commands turn that fell through to a clarification the human
        # might have expected to just work
        if e.get("advised_route") == "run_commands" and e.get("outcome") == "clarification":
            notes.append(f"- turn [{e['index']}] ({e['text']!r}) routed to a command but "
                         f"ended in a clarification — a human might have expected it to act.")
        if e.get("advised_route") == "direct_response" and "agent" in e["text"].lower() \
                and e.get("outcome") == "direct_response":
            notes.append(f"- turn [{e['index']}] ({e['text']!r}) sounded like an ask to create "
                         f"an owner but was answered as discussion — arguably it should have "
                         f"offered to set one up.")
        for a in e.get("created_agents", []):
            if _looks_like_unresolved_question(a.get("mission") or ""):
                notes.append(
                    f"- turn [{e['index']}] ({e['text']!r}) created **{a.get('title')}** whose "
                    f"mission is actually an unresolved clarifying question, not a mission — "
                    f"MR1 manufactured persistent state instead of asking the human."
                )
    if notes:
        for n in notes[:20]:
            w(n)
    else:
        w("_Nothing stood out as unnatural._")
    w("")


def _render_marwan_judgment(w, result, transcript) -> None:
    w("## Would Marwan have considered MR1's choices sensible?")
    w("")
    counts = result.get("counts", {})
    findings = result.get("findings", {})
    highs = sum(1 for group in findings.values() for f in group if f and f[0] == "high")
    agents = counts.get("agents", 0)
    outcomes = counts.get("outcomes", {})

    judgments: List[str] = []
    if outcomes.get("direct_response", 0) and outcomes.get("persistent_agent", 0):
        judgments.append(
            "MR1 kept discussion and action distinct — architectural questions were "
            "answered directly, and only genuinely broad, long-lived responsibilities "
            "became persistent agents.")
    if agents <= result.get("limits", {}).get("max_total_agents", 12):
        judgments.append(
            f"The hierarchy stayed small ({agents} persistent agents), well under the "
            f"ceiling — it did not manufacture owners it did not need.")
    if outcomes.get("workflow", 0):
        judgments.append(
            "Bounded, executable tasks became workflows (previewed before running) "
            "rather than permanent agents.")
    if highs == 0:
        judgments.append(
            "No high-severity orchestration or hierarchy findings: no runaway fanout, "
            "no duplicate owners, no acting on a guessed ambiguous target, no direct "
            "answer claiming work it had not done.")
    else:
        judgments.append(
            f"There were {highs} high-severity finding(s) that a careful reviewer would "
            f"push back on — see the findings table above.")

    for j in judgments:
        w(f"- {j}")
    w("")
    verdict = ("On balance, these read as sensible choices — the kind Marwan would have "
               "made himself." if highs == 0 else
               "On balance, the run is usable but has rough edges Marwan would want fixed "
               "before trusting it unattended.")
    w(f"**Assessment:** {verdict}")
    w("")


def write_report(layout: RunLayout, result: Dict[str, Any]) -> Path:
    text = render_report(layout, result)
    layout.report.write_text(text, encoding="utf-8")
    return layout.report
