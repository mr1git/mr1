"""
Renders a Phase D run (the `result` dict from `runner.run_corpus`) into a
human-readable Markdown review — the deliverable this whole harness exists
to produce: not "did MR1 execute the correct command" but "would Marwan
enjoy interacting with MR1 as a partner."
"""

from __future__ import annotations

from typing import Any, Dict, List

from tests.behavior_qa.judge import DIMENSION_KEYS
from tests.soak.hierarchical.outcomes import (
    OBJECTIVE_CREATED,
    OBJECTIVE_UPDATED,
    ORCHESTRATOR_CREATED,
    ORCHESTRATOR_REUSED,
    WORKER_SPAWN,
    WORKFLOW,
)


def render_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    w = lines.append

    clusters = result.get("clusters") or []
    counts = result.get("counts") or {}
    rollup = (result.get("rollup") or {}).get("overall") or {}
    planner = result.get("planner")

    w("# Phase D — Partner Behavior QA")
    w("")
    if planner != "real":
        w("> **Dry run (`--planner fake`).** No real brain or judge calls were "
          "made — every verdict below is a canned `FakeJudge` placeholder. "
          "This proves the harness wiring, not MR1's behavior. Run with "
          "`--planner real` for an actual behavioral report.")
        w("")
    error_names = result.get("error_cluster_names") or []
    if error_names:
        infra_names = result.get("infra_error_cluster_names") or []
        other_names = [n for n in error_names if n not in infra_names]
        w(
            f"> **⚠ PARTIAL RUN — {len(error_names)} of {counts.get('clusters')} "
            "cluster(s) excluded from the scores below.** These did not produce a "
            "valid partner-behavior verdict and are infrastructure/judge errors, "
            "not behavior findings:"
        )
        if infra_names:
            w(f"> - infrastructure/rate-limit failures: {', '.join(infra_names)}")
        if other_names:
            w(f"> - other judge errors: {', '.join(other_names)}")
        w("> Retry with `--resume` (re-drive + re-judge) or `--rejudge` "
          "(judge-only, no re-driving) to complete this run.")
        w("")
    w(f"- **planner:** `{planner}`  ·  **judge model:** `{result.get('judge_model')}`")
    w(f"- **clusters:** {counts.get('clusters')}  ·  "
      f"**judged:** {counts.get('judged_clusters')}  ·  "
      f"**turns:** {counts.get('turns')}")
    if error_names:
        w(f"- **excluded from scores (infra/judge error):** {len(error_names)} — "
          f"{', '.join(error_names)}")
    if rollup:
        w(f"- **mean partner score:** {rollup.get('mean_partner_score')}/10  ·  "
          f"**would continue using:** {_pct(rollup.get('would_continue_using_pct'))}  ·  "
          f"**would have done it himself:** {_pct(rollup.get('would_have_done_himself_pct'))}  ·  "
          f"**initiative calibration:** {rollup.get('initiative_calibration')}")
    w("")

    _render_transcript(w, clusters)
    _render_actions(w, clusters)
    _render_worker_and_ownership_decisions(w, clusters, result)
    _render_judge_report(w, clusters, result)
    _render_initiative_by_action_class(w, result)
    _render_examples(w, clusters)
    _render_recommendations(w, result)
    _render_closing(w, result)

    return "\n".join(lines) + "\n"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _cluster_heading(c: Dict[str, Any]) -> str:
    tag = f"Category {c.get('category')}"
    if c.get("resumes_from"):
        tag += f", continues {c['resumes_from']!r}"
    return f"### {c.get('name')} ({tag}) — {c.get('topic')}"


# ---------------------------------------------------------------------
# 1. transcript
# ---------------------------------------------------------------------

def _render_transcript(w, clusters: List[Dict[str, Any]]) -> None:
    w("## 1. Transcript")
    w("")
    for c in clusters:
        w(_cluster_heading(c))
        w(f"*{c.get('goal')}*")
        w("")
        for t in c.get("turns") or []:
            w(f"**[{t['index']}] Marwan:** {t['text']}")
            for se in t.get("soft_expectations") or []:
                w(f"> *(soft expectation, not a rule: {se})*")
            resp = (t.get("response_text") or "").strip().replace("\n", " ")
            if len(resp) > 320:
                resp = resp[:320] + "…"
            marker = "" if not t.get("findings") else "  ⚠️"
            w(f"> MR1 ({t.get('outcome')}): {resp}{marker}")
            w("")
        w("")


# ---------------------------------------------------------------------
# 2. actions taken
# ---------------------------------------------------------------------

def _render_actions(w, clusters: List[Dict[str, Any]]) -> None:
    w("## 2. Actions taken")
    w("")
    w("_Workers spawned and orchestrators created/reused are reported in "
      "separate columns on purpose — a worker spawn is never the same "
      "behavioral class as creating (or reusing) a durable orchestrator._")
    w("")
    w("| cluster | workers spawned | orchestrators created | orchestrators "
      "reused | workflows created | messages sent | approvals raised |")
    w("|---|---|---|---|---|---|---|")
    any_action = False
    for c in clusters:
        turns = c.get("turns") or []
        workers = sum(len(t.get("worker_spawns") or []) for t in turns)
        orchestrators_created = [
            a.get("title") for t in turns for a in t.get("created_agents") or []
        ]
        orchestrators_reused = sum(1 for t in turns if t.get("outcome") == ORCHESTRATOR_REUSED)
        workflows = sum(len(t.get("created_workflows") or []) for t in turns)
        messages = sum(len(t.get("created_messages") or []) for t in turns)
        approvals = sum(len(t.get("approval_ids") or []) for t in turns)
        if workers or orchestrators_created or orchestrators_reused or workflows or messages or approvals:
            any_action = True
        w(f"| {c.get('name')} | {workers or '—'} | "
          f"{', '.join(orchestrators_created) or '—'} | "
          f"{orchestrators_reused or '—'} | {workflows or '—'} | "
          f"{messages or '—'} | {approvals or '—'} |")
    w("")
    if not any_action:
        w("_No workers, orchestrators, workflows, messages, or approvals "
          "were created across the corpus._")
        w("")


# ---------------------------------------------------------------------
# 2b. worker & ownership decisions
# ---------------------------------------------------------------------

def _render_worker_and_ownership_decisions(
    w, clusters: List[Dict[str, Any]], result: Dict[str, Any],
) -> None:
    w("## Worker & ownership decisions")
    w("")
    metrics = result.get("metrics") or {}

    wu = metrics.get("worker_utilization")
    w("**Worker utilization** — among bounded-investigation-shaped turns, "
      "how often MR1 used a worker or a proportionate workflow instead of "
      "an ungrounded direct answer, refusal, or excessive clarification:")
    if wu:
        w(f"- {wu['numerator']}/{wu['denominator']} = **{wu['score']}**")
    else:
        w("- _no bounded-investigation-shaped turns in this run_")
    w("")

    ci = metrics.get("context_isolation")
    w("**Context isolation** — among worker spawns with a follow-up turn, "
      "how often the follow-up read as grounded recall rather than MR1 "
      "redoing the investigation inline in its own context:")
    if ci:
        w(f"- {ci['numerator']}/{ci['denominator']} = **{ci['score']}**")
    else:
        w("- _no worker spawns with a following turn in this run_")
    w("")

    w("**Worker vs. workflow decisions** — clusters where a "
      "bounded-investigation-shaped turn resolved to a worker spawn or a "
      "workflow:")
    w("")
    w("| cluster | outcome |")
    w("|---|---|")
    any_wf_wk = False
    for c in clusters:
        for t in c.get("turns") or []:
            if t.get("outcome") in (WORKER_SPAWN, WORKFLOW):
                any_wf_wk = True
                w(f"| {c.get('name')} [{t.get('index')}] | {t.get('outcome')} |")
    if not any_wf_wk:
        w("| _none_ | — |")
    w("")

    w("**Orchestrator ownership decisions** — clusters where an "
      "orchestrator was created or reused:")
    w("")
    w("| cluster | outcome | title(s) |")
    w("|---|---|---|")
    any_owner = False
    for c in clusters:
        for t in c.get("turns") or []:
            if t.get("outcome") == ORCHESTRATOR_CREATED:
                any_owner = True
                titles = [a.get("title") for a in t.get("created_agents") or []]
                w(f"| {c.get('name')} [{t.get('index')}] | created | {', '.join(titles) or '—'} |")
            elif t.get("outcome") == ORCHESTRATOR_REUSED:
                any_owner = True
                w(f"| {c.get('name')} [{t.get('index')}] | reused | — |")
    if not any_owner:
        w("| _none_ | — | — |")
    w("")

    oc = metrics.get("orchestrator_creation")
    oj = metrics.get("ownership_judgment")
    if oc:
        w(f"- orchestrator creation score (ownership_judgment + "
          f"orchestrator_reuse, normalized, averaged): **{oc['score']}** "
          f"across {oc['clusters']} cluster(s)")
    if oj:
        detail_parts = ", ".join(
            f"{k}={v}" for k, v in oj.items() if k != "score"
        )
        w(f"- ownership judgment composite: **{oj['score']}** ({detail_parts})")
    w("")

    escalation = next(
        (c for c in clusters if c.get("name") == "worker_to_orchestrator_escalation"), None,
    )
    if escalation:
        w("**Worker-to-orchestrator escalation case** — bounded probe -> "
          "repeated responsibility -> durable owner:")
        w("")
        for t in escalation.get("turns") or []:
            w(f"- [{t.get('index')}] {t.get('text')!r} -> outcome=`{t.get('outcome')}`")
        w("")

    objective_turns = [
        (c.get("name"), t)
        for c in clusters
        for t in c.get("turns") or []
        if t.get("outcome") in (OBJECTIVE_CREATED, OBJECTIVE_UPDATED)
    ]
    if objective_turns:
        w("**Objective creation/update:**")
        for name, t in objective_turns:
            w(f"- {name} [{t.get('index')}] -> `{t.get('outcome')}`")
        w("")
    else:
        w("_No objective was created or updated this run — objectives "
          "are not currently reachable from the conversational MR1 "
          "path the harness drives (see deliverable notes)._")
        w("")


# ---------------------------------------------------------------------
# 3. judge report
# ---------------------------------------------------------------------

def _render_judge_report(w, clusters: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    w("## 3. Judge report")
    w("")
    preferences = result.get("preferences") or []
    if preferences:
        w("Behavioral preferences replayed into the judge for this run:")
        for p in preferences:
            first_line = p.strip().splitlines()[0] if p.strip() else ""
            w(f"- {first_line}")
        w("")
    else:
        w("_No behavioral preference memories were found/replayed for this run._")
        w("")

    header = (
        "| cluster | cat | behavior | approval | continue? | himself? | surprised? | score | "
        + " | ".join(DIMENSION_KEYS) + " |"
    )
    sep = "|" + "---|" * (8 + len(DIMENSION_KEYS))
    w(header)
    w(sep)
    for c in clusters:
        j = c.get("judge") or {}
        if j.get("judge_error"):
            label = (
                "**infra error**" if str(j["judge_error"]).startswith("infrastructure error")
                else "**judge error**"
            )
            w(f"| {c.get('name')} | {c.get('category')} | {label} | — | — | — | — | — | "
              + " | ".join("—" for _ in DIMENSION_KEYS) + " |")
            continue
        dims = j.get("dimensions") or {}
        w(
            f"| {c.get('name')} | {c.get('category')} | {j.get('behavior')} | "
            f"{j.get('marwan_approval')} | {j.get('would_continue_using')} | "
            f"{j.get('would_marwan_have_done_this_himself')} | "
            f"{'yes' if j.get('surprised_by_action') else 'no'} | {j.get('partner_score')} | "
            + " | ".join(str(dims.get(k, "—")) for k in DIMENSION_KEYS) + " |"
        )
    w("")

    w("### Comments")
    w("")
    for c in clusters:
        j = c.get("judge") or {}
        if j.get("judge_error"):
            w(f"- **{c.get('name')}** — judge call failed: {j['judge_error']}")
            continue
        comments = j.get("comments") or {}
        parts = []
        if comments.get("intelligent"):
            parts.append(f"intelligent: {comments['intelligent']}")
        if comments.get("annoying"):
            parts.append(f"annoying: {comments['annoying']}")
        if comments.get("unnatural"):
            parts.append(f"unnatural: {comments['unnatural']}")
        if parts:
            w(f"- **{c.get('name')}** — " + "; ".join(parts))
    w("")


# ---------------------------------------------------------------------
# 3b. initiative calibration by action class
# ---------------------------------------------------------------------

def _render_initiative_by_action_class(w, result: Dict[str, Any]) -> None:
    w("## Initiative calibration by action class")
    w("")
    w("_Reported separately per class so aggressive orchestrator creation "
      "can never hide behind conservative worker behavior in one overall "
      "number. A cluster's class is its dominant outcome across turns._")
    w("")
    by_class = (result.get("metrics") or {}).get(
        "initiative_calibration_by_action_class"
    ) or {}
    w("| class | clusters | calibration (0=balanced, +over-acting, -under-acting) |")
    w("|---|---|---|")
    for cls in ("worker", "workflow", "orchestrator", "objective"):
        entry = by_class.get(cls)
        if entry:
            w(f"| {cls} | {entry['clusters']} | {entry['calibration']:+.2f} |")
        else:
            w(f"| {cls} | 0 | n/a |")
    w("")


# ---------------------------------------------------------------------
# 4. examples
# ---------------------------------------------------------------------

def _render_examples(w, clusters: List[Dict[str, Any]]) -> None:
    w("## 4. Examples")
    w("")
    for label, behavior in (
        ("Too passive", "TOO_PASSIVE"),
        ("Too aggressive", "TOO_AGGRESSIVE"),
        ("Balanced initiative", "BALANCED"),
    ):
        w(f"### {label}")
        w("")
        matches = [c for c in clusters if (c.get("judge") or {}).get("behavior") == behavior]
        if not matches:
            w(f"_No clusters judged {behavior}._")
        for c in matches:
            j = c.get("judge") or {}
            w(f"- **{c.get('name')}** (category {c.get('category')}, score "
              f"{j.get('partner_score')}) — {c.get('goal')}")
        w("")


# ---------------------------------------------------------------------
# 5. recommendations
# ---------------------------------------------------------------------

def _render_recommendations(w, result: Dict[str, Any]) -> None:
    w("## 5. Recommendations")
    w("")
    rollup = (result.get("rollup") or {}).get("overall") or {}
    rollup_a = (result.get("rollup") or {}).get("category_a") or {}
    rollup_b = (result.get("rollup") or {}).get("category_b") or {}
    notes: List[str] = []

    calib = rollup.get("initiative_calibration")
    calib_b = rollup_b.get("initiative_calibration")
    if calib is not None:
        if calib > 0.15:
            notes.append(
                f"Overall initiative calibration is {calib:+.2f} — MR1 is trending "
                "toward over-acting. Look at the 'Too aggressive' examples above first."
            )
        elif calib < -0.15:
            notes.append(
                f"Overall initiative calibration is {calib:+.2f} — MR1 is trending "
                "toward under-acting. Look at the 'Too passive' examples above first."
            )
        else:
            notes.append(f"Overall initiative calibration is {calib:+.2f} — roughly balanced.")
    if calib_b is not None and abs(calib_b) > abs(calib or 0) + 0.15:
        notes.append(
            f"Category B (ambiguous prompts) calibration is {calib_b:+.2f}, notably "
            "different from the overall number — the ambiguity-only set is where "
            "over/under-acting actually matters most."
        )

    no_continue = rollup.get("would_continue_using_no") or []
    if no_continue:
        notes.append(
            "Clusters where the judge said Marwan would NOT keep using MR1 if it "
            f"behaved this way repeatedly: {', '.join(no_continue)}."
        )
    no_himself = rollup.get("would_have_done_himself_no") or []
    if no_himself:
        notes.append(
            "Clusters where the action MR1 took doesn't match what Marwan would "
            f"have decided himself: {', '.join(no_himself)}."
        )
    surprising_bad = rollup.get("surprising_and_not_approved")
    if surprising_bad:
        notes.append(
            f"{surprising_bad} cluster(s) were both surprising and not approved — "
            "the most concerning combination, worth reviewing first."
        )

    for label, r in (("Category A", rollup_a), ("Category B", rollup_b)):
        score = r.get("mean_partner_score")
        if score is not None and score < 6:
            notes.append(f"{label} mean partner score is {score}/10 — below the midpoint.")

    if not notes:
        notes.append("No systemic issues surfaced by the aggregate numbers; see the "
                      "per-cluster comments above for texture.")
    for n in notes:
        w(f"- {n}")
    w("")


# ---------------------------------------------------------------------
# 6. closing verdict
# ---------------------------------------------------------------------

def _render_closing(w, result: Dict[str, Any]) -> None:
    w("## Would Marwan enjoy this?")
    w("")
    rollup = (result.get("rollup") or {}).get("overall") or {}
    if not rollup or rollup.get("judged_clusters", 0) == 0:
        w("_No judged clusters — nothing to conclude (dry run or all judge calls failed)._")
        w("")
        return
    w(f"- **Mean partner score:** {rollup.get('mean_partner_score')}/10 across "
      f"{rollup.get('judged_clusters')} judged clusters")
    w(f"- **Would continue using MR1 this way:** {_pct(rollup.get('would_continue_using_pct'))}")
    w(f"- **Would have made the same call himself:** "
      f"{_pct(rollup.get('would_have_done_himself_pct'))}")
    w(f"- **Initiative calibration:** {rollup.get('initiative_calibration')} "
      "(0 = balanced, positive = over-acting, negative = under-acting)")
    w(f"- **Behavior distribution:** {rollup.get('behavior_distribution')}")
    twin = (result.get("metrics") or {}).get("digital_twin")
    if twin:
        w(f"- **Digital twin score:** {twin['score']} across {twin['clusters']} "
          f"judged clusters — formula: {twin['formula']}")
    w("")
