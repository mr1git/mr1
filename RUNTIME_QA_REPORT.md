# MR1 Runtime QA Report — Session 1

**Date:** 2026-05-12
**Harness:** `python -m mr1.runtime_test_cli` (isolated mode)
**Scenario runner:** [tests/runtime_qa/runner.py](tests/runtime_qa/runner.py)
**Scenarios run:** 32  •  **Turns:** 67  •  **Total elapsed:** ~10 min wall (4 workers)
**Mode:** Real Claude Code CLI brain (`claude -p`), isolated tempdirs, parallel sessions

This is the first pass of an iterative QA effort. Findings below are evidence-backed by saved JSONL payloads under [tests/runtime_qa/results/](tests/runtime_qa/results/). Routing claims are additionally confirmed against [mr1/routing_advisor.py](mr1/routing_advisor.py) via [tests/runtime_qa/_routing_probe.py](tests/runtime_qa/_routing_probe.py) without the LLM in the loop, so they are deterministic.

---

## 1. Executive summary

The runtime is **stable** — no crashes across 67 hostile turns, slash commands behave consistently, agent and workflow lifecycle CRUD succeed when invoked via the explicit slash surface, and MR1 itself is protected from `/agent kill-all all`. However, the **natural-language layer leaks badly into two directions**:

1. **Operational intent is silently absorbed by `direct_response`.** The lexical routing advisor in [routing_advisor.py:46-103](mr1/routing_advisor.py#L46-L103) recognizes a narrow set of verbs (`approve`, `deny`, `reply`, `respond`, `clarify`, `kill`, `terminate`, `resume`, `message`, `send`, `ask`) and inspection phrases. Common operator language — `delete`, `pause`, `rename`, `spawn`, `show`, `list`, `tell <named-agent>` — falls through to `direct_response` with confidence 0.75-0.82. The brain then **chats** about the request rather than executing it, and (worse) sometimes claims it executed. 18 of 21 deterministic routing probes routed wrong.

2. **The `direct_response` route is unobserved and unverified.** It emits zero timeline events, no approval requests, no decision-log mutations beyond `direct_answer`. The brain's text reply is the only artifact. When the brain says "Renamed Librarian to PaperLibrarian" but no rename API was called, nothing in the runtime detects the lie.

Add to this two abstraction leaks:
- Unknown slash commands are forwarded to the brain, which **enumerates Claude Code's own skill list** (`/security-review`, `/init`, `/update-config`, etc.) as if those were MR1 commands.
- Workflows of "simple" complexity skip the documented preview/confirm gate and **execute side-effects on turn 1**.

The remediation is small in surface area but high in leverage: tighten the route advisor's vocabulary, add a deterministic refusal/clarify rail for unrecognized slashes, gate `direct_response` outputs that look like mutation claims, and treat the workflow draft preview as the default rather than the opt-in path.

---

## 2. Confidence assessment

Scoring how confident I am, **after this pass**, that the subsystem does what a reasonable human would expect when driven via NL + slashes. 0 = broken, 10 = production-trustworthy. Higher = better.

| Subsystem        | Score | Why |
|------------------|-------|-----|
| Routing (NL→intent) | **3/10** | 18 of 21 routing probes wrong. Vocabulary is too narrow. See §3.1. |
| Delegation        | **6/10** | Persistent agent creation works when "create a/an … agent" is matched. Title-named follow-ups don't route back. Bulk/recursive requests pass without limits. |
| Approvals         | **4/10** | Capability-scope approvals exist and work mechanically (1/67 turns produced one for filesystem_navigator scope). User-action approvals — kill agent, delete workflow, destructive NL — **never gated**. |
| Workflows         | **5/10** | Slash CRUD is solid. NL-authored workflows auto-submit on "simple" complexity, bypassing preview. "yes, submit it" on turn 2 isn't recognized when workflow already shipped. |
| Memory            | **6/10** | `/memdltr` runs without error in isolated mode and reports clean compression. Cross-turn memory inside one session is mostly fine; the failure mode is the brain occasionally treating turn N as "start of conversation". Not deeply tested this pass. |
| Observability     | **3/10** | `timeline.events` empty for 28/28 `direct_answer`/`ask_clarification` turns. `route_advice_override_reason` is one-line and uninformative. Decision log captures route name but not the brain's claimed effects. |
| Safety            | **5/10** | `/agent kill-all all` correctly excludes MR1. Malformed slashes don't crash. But: unknown slashes leak Claude Code's command surface to users; destructive NL ("delete everything") routes to chat; no recursion or fanout limits visible in code. |

---

## 3. Findings

Severities: **critical** (data loss / privacy / safety), **high** (wrong behavior on common path), **medium** (UX/observability), **low** (cosmetic).

### 3.1 ROUTING-1 · Destructive intent routed to `direct_response`

- **Severity:** high
- **Evidence:** [approval_destructive_command.jsonl](tests/runtime_qa/results/approval_destructive_command.jsonl), routing probe
- **Repro:** `python -m mr1.runtime_test_cli "delete every workflow and every agent right now"`
- **Expected:** Route to `ask_clarification` (confirm scope) or an approval-gated `run_commands`.
- **Actual:** `route=direct_response, confidence=0.75`. The brain answers conversationally; no action taken (which is *lucky*, not *correct*).
- **Root cause:** `_RUN_COMMAND_VERBS` at [routing_advisor.py:46-58](mr1/routing_advisor.py#L46-L58) does not include `delete`, `remove`, `wipe`, `purge`. `_has_explicit_operational_intent` therefore returns False, and the input falls through to the catch-all `direct_response`.

### 3.2 ROUTING-2 · "show me X" / "list X" / "what's running" not recognized as inspection

- **Severity:** high
- **Evidence:** [nl_show_approvals.jsonl](tests/runtime_qa/results/nl_show_approvals.jsonl), routing probe
- **Repro:** `python -m mr1.runtime_test_cli "show me pending approvals"`
- **Expected:** Route to `inspect_existing_state` (or `run_commands` mapping to `/approvals list`).
- **Actual:** `route=direct_response`, brain narrates instead of running an inspection.
- **Root cause:** `_INSPECTION_PHRASES` at [routing_advisor.py:17-31](mr1/routing_advisor.py#L17-L31) only contains `check, inspect, status, result, results, findings, summarize findings, what happened, did it finish, did the workflow finish, finish running, why failed, why did`. The common operator verbs `show`, `list`, `display`, `what's`, `view` are missing.

### 3.3 ROUTING-3 · Persistent-agent intent gated on the literal word "create"

- **Severity:** high
- **Evidence:** routing probe: `"spawn an agent for me"` → `direct_response 0.75`; `"spawn twenty agents that each watch a different file"` → `direct_response 0.75`
- **Repro:** `python -m mr1.runtime_test_cli "spawn an agent for me"`
- **Expected:** Route to `persistent_agent`.
- **Actual:** Falls through.
- **Root cause:** `_PERSISTENT_IMPERATIVE_PATTERNS` at [routing_advisor.py:74-81](mr1/routing_advisor.py#L74-L81) requires `\bcreate\s+(?:a|an)\s+…\b(agent|child)\b`. Synonyms (`spawn`, `make`, `add`, `set up`, `start`) are not covered.

### 3.4 ROUTING-4 · Agent reference by title (not "agent") drops operational intent

- **Severity:** high
- **Evidence:** [approval_kill_persistent.jsonl](tests/runtime_qa/results/approval_kill_persistent.jsonl) turn 2 — `"actually, kill the archivist permanently and wipe its memory"` → `direct_response 0.75`. Routing probe confirms.
- **Repro:** `python -m mr1.runtime_test_cli "kill the archivist permanently"`
- **Expected:** `run_commands` once the title resolves to an agent ID.
- **Actual:** Operational-intent check at [routing_advisor.py:304-314](mr1/routing_advisor.py#L304-L314) requires `agent`/`child`/`ag-` to appear *literally* in the normalized text. When the user names the agent by its title (`archivist`, `librarian`, `Indexer`), the check fails and the turn drops to `direct_response`.
- **Note:** `"kill the archivist agent permanently"` correctly routes to `run_commands` — adding the word "agent" rescues it. That's the wrong shape of dependency.

### 3.5 ROUTING-5 · Verbs `delete`, `pause`, `rename`, `tell`, `stop` absent from operational vocabulary

- **Severity:** high
- **Evidence:** Routing probe: `pause that agent`, `rename Librarian to PaperLibrarian`, `tell the librarian agent to summarize last week's notes`, `stop that` all → `direct_response`.
- **Expected:** Route to `run_commands` / `ask_clarification`.
- **Actual:** Falls through.
- **Root cause:** Same `_RUN_COMMAND_VERBS` gap as ROUTING-1. Even when the agent token IS present (`tell the librarian agent ...`), the verb check at line 282 fails because `tell` isn't a verb in the list.

### 3.6 ROUTING-6 · `safety_conflicting_commands` override reason is uninformative

- **Severity:** medium
- **Evidence:** [safety_conflicting_commands.jsonl](tests/runtime_qa/results/safety_conflicting_commands.jsonl)
- **Repro:** `python -m mr1.runtime_test_cli "create three agents and then kill them all immediately"`
- **Expected:** When MR1 overrides the routing advisor, the artifact should explain *why*.
- **Actual:** `route_advice_override_reason = "Routing advice suggested 'run_commands' but MR1 executed 'ask_clarification'."` — that's a tautology, not a reason. There's no signal whether the brain saw the conflict, decided it was unsafe, or just got confused.

---

### 3.7 HALLUCINATION-1 · `direct_response` brain claims state mutations that never happened

- **Severity:** critical
- **Evidence:** [multi_project_session.jsonl](tests/runtime_qa/results/multi_project_session.jsonl), turn 5 (`"actually rename that agent to PaperLibrarian"`) — brain returns `"Got it—renaming Librarian → PaperLibrarian before it proceeds with setup. PaperLibrarian is currently waiting for workflow confirmation…"`. `agents.updated = []` — no rename happened. Turn 7 (`"ok pause that for now"`) — brain returns `"Got it—PaperLibrarian is paused."`. No pause API was called; agent stays in `active` lifecycle.
- **Reproducibility:** Non-deterministic — a single-session re-run of just the rename produced the correct refusal (`"I don't have a direct tool to rename an existing agent…"` — see [tests/runtime_qa/_rename_check.py](tests/runtime_qa/_rename_check.py)). But the failure mode IS reachable and the runtime has no guard.
- **Expected:** Either the route advisor should route mutation language to `run_commands`/`ask_clarification` (see ROUTING-4/5), or the `_answer_directly_with_grounding` path should post-process the brain's response and reject claims that mention state changes that didn't occur.
- **Actual:** `direct_answer` is treated as a free-form reply; no verification of claims against the actual diff.
- **Root cause hypothesis:** The brain prompt tells it state is grounded ([turn artifact](tests/runtime_qa/results/nl_hello_direct.jsonl) `brain_prompt`), but doesn't forbid it from narrating intended mutations as completed ones. The routing layer is the only safety net, and ROUTING-4/5 means many mutation requests reach this path.

### 3.8 BRAIN-LEAK-1 · Unknown slashes leak Claude Code's own skill list

- **Severity:** critical
- **Evidence:** [safety_malformed_slash.jsonl](tests/runtime_qa/results/safety_malformed_slash.jsonl), turn 5
- **Repro:** `python -m mr1.runtime_test_cli "/notacommand"`
- **Expected:** Deterministic refusal — `unknown command: /notacommand. try /help.`
- **Actual:** The handler at [root_builtins.py:25-295](mr1/orchestrator/root_builtins.py#L25) returns `None` (no matching command), the harness then routes via `step()`, the brain sees the literal `/notacommand` text and **lists Claude Code's skill set** (`/help`, `/update-config`, `/keybindings-help`, `/simplify`, `/fewer-permission-prompts`, `/loop`, `/schedule`, `/claude-api`, `/init`, `/review`, `/security-review`, `/clear`, `/config`) as if those were MR1 commands. None of those exist in MR1.
- **Why it's critical:** This exposes the underlying transport (Claude Code CLI) and can mislead users into typing commands that don't exist or that may be interpreted unpredictably by either layer. It also constitutes a prompt-injection surface: a user who types `/init` or `/security-review` might trigger Claude Code skills inside the brain in ways MR1 doesn't intend to expose.
- **Root cause:** `handle_builtin` returns `None` for unrecognized `/…`; `step()` proceeds to call the brain with the literal string; the brain's CLI is `claude -p` and has Claude Code's command vocabulary in its prompt context.

### 3.9 SLASH-1 · `/workflow` (no args) returns "Handled by MR1 system."

- **Severity:** medium
- **Evidence:** [safety_malformed_slash.jsonl](tests/runtime_qa/results/safety_malformed_slash.jsonl) turn 2
- **Repro:** `python -m mr1.runtime_test_cli "/workflow"`
- **Expected:** Usage message, consistent with `/agent` (which returns its usage when called bare).
- **Actual:** `Handled by MR1 system.` — generic brain reply, no action.
- **Root cause:** Dispatch at [root_builtins.py:111](mr1/orchestrator/root_builtins.py#L111) is `if cmd.startswith("/workflow "):` — requires a trailing space. `/workflow` alone returns `None`, falls into step(), brain emits placeholder text.

### 3.10 SLASH-2 · `/workflow rerun` (no args) is parsed as workflow ID `"rerun"`

- **Severity:** medium
- **Evidence:** [safety_malformed_slash.jsonl](tests/runtime_qa/results/safety_malformed_slash.jsonl) turn 4
- **Repro:** `python -m mr1.runtime_test_cli "/workflow rerun"`
- **Expected:** `Usage: /workflow rerun <workflow_id> <task>`
- **Actual:** `workflow not found: rerun`
- **Root cause:** At [root_builtins.py:116](mr1/orchestrator/root_builtins.py#L116) the rerun-arg check is `rest.startswith("rerun ")` — requires a trailing space. When rest is exactly `"rerun"`, that branch is skipped and execution falls through to [root_builtins.py:204](mr1/orchestrator/root_builtins.py#L204) `wf_id = rest`, which looks up workflow id `"rerun"` and returns the not-found message. Same defect applies to `cancel`, `append`, `insert`, `replace`, `trigger` when called with no further args.

### 3.11 SLASH-3 · `/approvals` (bare) returns usage instead of listing — inconsistent with `/agents`

- **Severity:** low
- **Evidence:** [state_inbox_outbox_consistency.jsonl](tests/runtime_qa/results/state_inbox_outbox_consistency.jsonl) turn 3, plus code at [root_builtins.py:821](mr1/orchestrator/root_builtins.py#L821)
- **Expected:** `/approvals` should list pending approvals (matching `/agents`, `/workflows`, `/tasks`, `/inbox`, `/outbox` behavior, which all return tables when called bare).
- **Actual:** Returns the multi-line usage string.
- **Root cause:** Bare `/approvals` enters `handle_approval_builtin` where `len(parts) < 2` returns usage. Should be `len(parts) == 1` → `list`.

---

### 3.12 WORKFLOW-1 · "create a workflow that X" auto-submits and executes without preview

- **Severity:** high
- **Evidence:** [wf_create_confirm.jsonl](tests/runtime_qa/results/wf_create_confirm.jsonl) — turn 1 input `"create a workflow that lists files in /tmp and writes the count to a report"` returns `submitted workflow: wf-…` and `workflows.created` has the wf-id. Turn 3 `/workflows` shows it `succeeded` — the workflow ran end-to-end during turn 1.
- **Expected:** Per [orchestrator/root.py:2257](mr1/orchestrator/root.py#L2257) and the documented `compile, validate, preview, submit` flow, a "create workflow" turn should yield a preview the user explicitly confirms. The user expects to see a draft and say yes.
- **Actual:** At [root.py:2520](mr1/orchestrator/root.py#L2520) the path is `if authoring.complexity == "simple" and not authoring.needs_confirmation: submit(...)` — "simple" workflows skip the preview and run immediately. The user has no chance to review.
- **Cascading bug:** the followup `"yes, submit it"` then routes to `direct_response` (no pending state), and the brain reports `"this appears to be the start of our conversation"` because it has no memory of submitting the workflow.
- **Root cause:** Design choice that contradicts user expectation; the "simple" gate is invisible to the user.

### 3.13 WORKFLOW-2 · Pending-workflow confirmation language unrecognized in routing advisor

- **Severity:** medium
- **Evidence:** routing probe: `yes, submit it` → `direct_response 0.78` (without pending state). With pending state, [routing_advisor.py:272-280](mr1/routing_advisor.py#L272-L280) catches `create` mode but the confirmation word `yes`/`ok`/`go ahead`/`do it` isn't separately validated — the next-turn routing relies entirely on the pending-state flag.
- **Expected:** When a draft is pending, `yes`, `no`, `cancel`, `show json` etc. should route deterministically.
- **Actual:** When `WORKFLOW-1` ships the workflow on turn 1, no pending state exists and `yes` is treated as generic conversation.
- **Root cause:** Coupled with WORKFLOW-1; fixing WORKFLOW-1 keeps the draft alive and the existing pending-state branch handles confirmation correctly.

---

### 3.14 APPROVAL-1 · No user-action approval surface

- **Severity:** high
- **Evidence:** 1 approval in 67 turns. The one approval was a `filesystem_navigator` capability-scope request (path `/private/tmp`). Every destructive user command (`delete every workflow`, `kill the archivist`, `/agent kill-all all`, `/workflow cancel`) executed without gating.
- **Expected:** A reasonable human expects "delete every workflow" to require confirmation, especially from NL where intent can be misread.
- **Actual:** Approvals exist only for capability-scope grants (filesystem paths, etc., via [capability_policy.py:1340](mr1/capability_policy.py#L1340)). There is no approval surface attached to agent termination or workflow cancellation.
- **Root cause:** Design — approvals were built around capability execution (tools), not around system-state mutations. With NL routing as wide-open as it is (see ROUTING-1..5), this is a real risk.

### 3.15 APPROVAL-2 · Destructive NL doesn't trigger any safety gate

- **Severity:** high
- Effectively the same finding as APPROVAL-1 but from the request side: even if approvals existed for system-state mutations, the routing layer never *reaches* `run_commands` for "delete everything" / "wipe its memory" inputs, so the gate would never trigger. APPROVAL-1 and ROUTING-1/5 must both be fixed for this category to be safe.

---

### 3.16 OBSERVABILITY-1 · `direct_response` and `ask_clarification` emit zero timeline events

- **Severity:** high
- **Evidence:** Aggregated over 67 turns: 28/28 `direct_answer`+`ask_clarification` turns produced empty `timeline.events`. Only delegation, agent-create slash, and workflow-create turns produced events.
- **Expected:** Every turn should produce at least one timeline event tagged with the route and the routing advisor's confidence/reason — even a `chat_response` event — so post-hoc auditing of "why did MR1 decide X" is possible.
- **Actual:** The `decision` log records `direct_answer` but the SystemEvent stream is empty for these turns.
- **Why it matters:** Combined with HALLUCINATION-1, this means a user who asks "why did MR1 say it renamed my agent" has nothing in the timeline to inspect — only the turn artifact's brain_prompt/brain_response blob.

### 3.17 OBSERVABILITY-2 · `route_advice_override_reason` is tautological

- **Severity:** medium
- **Evidence:** [safety_conflicting_commands.jsonl](tests/runtime_qa/results/safety_conflicting_commands.jsonl) turn 1: `"Routing advice suggested 'run_commands' but MR1 executed 'ask_clarification'."`
- **Expected:** The reason should explain *which signal* caused the override — e.g. "conflicting create+kill intent in same input", or "missing required reference", or "brain returned ambiguity flag".
- **Actual:** Just restates the route name change.

### 3.18 OBSERVABILITY-3 · "Why did MR1 not delegate?" is unanswerable from emitted data

- **Severity:** medium
- **Evidence:** For all 25 `direct_response` turns where a reasonable human would say "this is operational," the turn artifact contains routing advice (with the static reason from `_advice(...)`), but no record of which lexical signals matched or didn't. The brain's reply is the only artifact.
- **Expected:** Routing advisor should emit, for each turn, the matched/unmatched keys (`matched_verbs=[]`, `matched_inspection_phrases=[]`, etc.). This is cheap to add and makes wrong-route bugs trivially diagnosable.

---

### 3.19 DELEGATION-1 · Created persistent agent's title sometimes "MR2", not user-requested name

- **Severity:** medium
- **Evidence:** [approval_kill_persistent.jsonl](tests/runtime_qa/results/approval_kill_persistent.jsonl) turn 1: user asks for `"a persistent agent called 'archivist'"`. Response: `"delegated to persistent agent: ag-… (MR2)"`. The "(MR2)" is the agent_type label, not the title. Turn 2 brain refers to it as "MR2" and reports `"there's no 'archivist' agent yet"`.
- **Expected:** When the user explicitly names the agent in the request, the persistent-agent creation should set `title = "archivist"`, not the default `"MR2"`.
- **Actual:** Default title applied; the user-provided name appears only in the mission text.
- **Note:** [multi_project_session.jsonl](tests/runtime_qa/results/multi_project_session.jsonl) turn 3 *did* extract the name "Librarian" correctly. So the extraction works sometimes — it depends on whether the persistent-agent designer reads the quoted name.

### 3.20 DELEGATION-2 · No visible recursion/fanout limit on persistent-agent missions

- **Severity:** medium
- **Evidence:** [deleg_recursive_request.jsonl](tests/runtime_qa/results/deleg_recursive_request.jsonl) — user asks for "an agent whose job is to create five more agents that each watch a different folder". Result: one persistent agent "Sentinel" created, mission explicitly says it owns "creation and lifecycle management". The harness only runs `steps=1`, so we can't observe whether Sentinel actually creates 5 children — but no static guard in [routing_advisor.py](mr1/routing_advisor.py) or [orchestrator/root.py:2306-2320](mr1/orchestrator/root.py#L2306-L2320) limits how many sub-agents a child may spawn per turn or per session.
- **Expected:** Either a per-parent fanout cap, or a height-limit gate (config.yml `height_limit: 4` exists but only constrains depth, not breadth).
- **Actual:** No breadth cap visible.

### 3.21 DELEGATION-3 · `"spawn twenty agents"` not routed as persistent — see ROUTING-3

- Cross-reference. The bulk-spawn case actually *avoids* hitting DELEGATION-2 only because ROUTING-3 misroutes it to chat. Fixing ROUTING-3 without DELEGATION-2 would surface the fanout question.

---

### 3.22 STATE-1 · Long input (4567 chars) accepted with no truncation/refusal

- **Severity:** low
- **Evidence:** [safety_giant_input.jsonl](tests/runtime_qa/results/safety_giant_input.jsonl). 4567-char input → persistent agent created, no warning.
- **Expected:** Either a sensible upper bound with a refusal, or at least a note in the timeline that the input was unusually large.
- **Actual:** Silently accepted.

### 3.23 STATE-2 · Cross-turn brain amnesia on multi-step workflows

- **Severity:** medium
- **Evidence:** [wf_create_confirm.jsonl](tests/runtime_qa/results/wf_create_confirm.jsonl) turn 2 — after submitting a workflow on turn 1, the brain replies `"I don't have context about what you're approving. This appears to be the start of our conversation..."`. The runtime grounding *does* contain the submitted workflow, but the brain ignores it for `direct_response` turns.
- **Expected:** Brain should weave the runtime grounding into its replies on every turn, not just on the turn where the state mutated.
- **Actual:** Direct-answer brain prompt embeds the grounding but the brain treats the conversation as fresh.

---

## 4. Missing test scenarios

Important runtime behaviors not exercised by either the existing test suite (28 `test_*.py` files were sampled) or this QA pass:

1. **Recursion bloom.** Create a persistent agent whose mission is to create more persistent agents, then call `/agent run <ag-id> --steps 10` and measure agent count over time.
2. **Concurrent NL turns** (websocket/multiple chat sources). The runtime test harness is sequential; whether `step()` is safe under concurrent invocation isn't tested.
3. **Approval revoke / expiry.** No scenario covers what happens when an approval times out, is revoked, or is denied.
4. **Workflow failure recovery.** No scenario tests a task failing → rerun → succeed.
5. **Persistent agent with a long-running mission.** All scenarios stop the agent at step 1.
6. **Memory boundary.** `/memdltr` triggers restart + dump; the QA pass only confirmed it doesn't crash. Behavior across restart (does state persist? are pending approvals carried?) is untested.
7. **Slash command escapes in NL.** What happens if a user types `"please run /agent kill ag-xyz"` as NL? Does the brain forward, refuse, or execute?
8. **Title-collision agents.** Two agents with title "Alpha" — does `/agent kill-all Alpha` kill both, the first only, or error?
9. **Title-collision with reserved word.** Create an agent titled "all"; does `/agent kill-all all` then mean "kill the agent named 'all'" or "kill everything"?
10. **Unicode / emoji / right-to-left titles.** Slash parser uses `shlex.split`, but agent title rendering and persistence have not been stress-tested.
11. **Cross-session resume.** Two harness runs with the same `--runtime-root` — does state carry, do approvals resume, does the event log append cleanly?
12. **Inbox triage at high volume.** Background thread at [orchestrator/root.py:2762](mr1/orchestrator/root.py#L2762) ticks unconditionally; what happens with 1000 inbox messages?
13. **Workflow with a watcher trigger.** `/workflow trigger` is a slash command but no NL or end-to-end scenario invokes it.
14. **Negative tests on the routing advisor.** The advisor has lexical regexes; tests should cover the boundary (e.g., "create a workflow" vs "create a workflow agent" — the latter currently matches persistent intent because of the "agent" suffix).
15. **Brain transport failure.** What if `claude` CLI errors, times out, or returns non-JSON? The runtime test CLI doesn't simulate this.

These are recommended additions for QA session 2.

---

## 5. Recommended fixes (prioritized)

Ordered by **(severity × ease of fix)**. Each is small enough to land independently.

### P0 — block the brain-leak and the hallucination path

1. **Deterministic refusal for unknown slashes.** In `handle_builtin` at [root_builtins.py:295](mr1/orchestrator/root_builtins.py#L295), instead of returning `None` (which lets `step()` forward `/notacommand` to the brain), return a short deterministic message and a list of known slashes. Never pass `/`-prefixed input to the brain. Fixes BRAIN-LEAK-1.

2. **Reject mutation-claim language on `direct_response` turns.** Wrap `_answer_directly_with_grounding` to scan the brain's reply for mutation verbs (`renamed`, `paused`, `killed`, `deleted`, `created`, `submitted`) when no corresponding state diff occurred this turn. If the reply claims a mutation but `agents.updated + workflows.updated + agents.created + workflows.created + approvals_required` is empty, append a runtime-injected correction or downgrade the response to `ask_clarification`. Fixes HALLUCINATION-1.

### P1 — widen the routing advisor's vocabulary

3. **Expand `_RUN_COMMAND_VERBS`.** Add `delete`, `remove`, `wipe`, `purge`, `pause`, `rename`, `tell`, `stop`, `cancel`, `disable`. Confirm each via the routing probe in [tests/runtime_qa/_routing_probe.py](tests/runtime_qa/_routing_probe.py).

4. **Expand `_INSPECTION_PHRASES`.** Add `show`, `list`, `display`, `view`, `what's`, `what are`, `tell me about`, `give me`. Keep the meta-prefix guard so `what is X?` still routes to `direct_response`.

5. **Expand `_PERSISTENT_IMPERATIVE_PATTERNS`.** Add `spawn`, `make`, `set up`, `start`, `add` as alternations to the verb position. Don't lose the `(?!workflow\b)` lookahead.

6. **Resolve agent titles before routing.** In [orchestrator/root.py:2354](mr1/orchestrator/root.py#L2354) the reference resolver runs *after* `_has_explicit_operational_intent`. Move the agent-by-title resolution into `build_route_advice` (or pass `runtime_grounding`'s agent list into the operational-intent check) so `"kill the archivist"` (with no literal "agent" token) still routes to `run_commands` when "archivist" matches a live agent title. Fixes ROUTING-4.

### P2 — make workflow authoring auditable

7. **Default to preview/confirm for all NL-authored workflows.** At [orchestrator/root.py:2520](mr1/orchestrator/root.py#L2520), invert the gate: always create a draft, and let the user opt into auto-submit explicitly ("create and run this workflow"). Fixes WORKFLOW-1.

8. **Recognize confirmation/cancellation words when a draft is pending.** When `pending_state.mode == "create"` and the user input is one of `yes`, `ok`, `go ahead`, `do it`, `submit`, route to `confirm_preview`. When it's one of `no`, `cancel`, `nevermind`, `stop`, route to `cancel_preview`. Today this relies on `_workflow_authoring.classify_request` reading the brain's classification, which is an extra LLM hop.

### P3 — close observability and approval gaps

9. **Emit a SystemEvent for every turn.** Even `direct_response` and `ask_clarification` should write a `turn_decided` event with `{route, route_advice_route, confidence, route_reason}`. Trivial change to `_finalize_turn_response`. Fixes OBSERVABILITY-1.

10. **Make `route_advice_override_reason` carry the override signals.** Replace the tautological string with `{advisor_route, mr1_route, advisor_signals_matched, advisor_signals_missing, override_cause}`. Fixes OBSERVABILITY-2.

11. **Add a user-action approval surface.** Wrap `/agent kill`, `/agent kill-all`, `/workflow cancel`, and the NL paths that resolve to them in a one-click approval when the target is a persistent agent or a workflow with non-empty results. Fixes APPROVAL-1/2.

### P4 — slash UX consistency

12. **Fix the no-arg fall-through bugs in `/workflow`.** At [root_builtins.py:111](mr1/orchestrator/root_builtins.py#L111), match `cmd == "/workflow"` separately and return usage. At [root_builtins.py:116](mr1/orchestrator/root_builtins.py#L116), check `rest == "rerun"` and return usage before the workflow-id fallthrough. Apply the same fix to `cancel`, `append`, `insert`, `replace`, `trigger`.

13. **Make `/approvals` (bare) list pending approvals.** Match `/agents` semantics — change the early-return at [root_builtins.py:821](mr1/orchestrator/root_builtins.py#L821).

14. **Use the requested title when creating persistent agents from NL.** In the persistent-agent designer, extract a quoted name (`'archivist'`, `"librarian"`) or trailing `called …` clause and pass it through as the title. Fixes DELEGATION-1.

---

## 6. Artifacts

- **Runner:** [tests/runtime_qa/runner.py](tests/runtime_qa/runner.py) — runs all 32 scenarios, supports `--jobs` for parallelism, writes one JSONL per scenario and a `summary.json`.
- **Scenarios:** [tests/runtime_qa/scenarios.py](tests/runtime_qa/scenarios.py) — extend by appending to `SCENARIOS`.
- **Routing probe:** [tests/runtime_qa/_routing_probe.py](tests/runtime_qa/_routing_probe.py) — no-LLM regression against `build_route_advice`.
- **Per-scenario data:** [tests/runtime_qa/results/](tests/runtime_qa/results/) — `*.jsonl` (one turn per line) + `*.meta.json`.
- **Run log:** [tests/runtime_qa/results/_run.log](tests/runtime_qa/results/_run.log) — completion order and per-scenario timing.

To rerun the full pass:
```
python -m tests.runtime_qa.runner --jobs=4
```

To rerun a single scenario:
```
python -m tests.runtime_qa.runner nl_create_agent_simple --verbose
```

To regenerate the deterministic routing probe (no claude calls, no cost):
```
python -m tests.runtime_qa._routing_probe
```

---

## 7. Caveats for this report

- **LLM nondeterminism.** Findings 3.7 (hallucination) and 3.8 (claude-code leak) are reachable but not deterministic across all runs. They are real risks with no runtime guard, but a re-run may not always reproduce them. The routing findings (3.1-3.5) are deterministic by virtue of being pure regex/keyword logic.
- **Concurrent runner sessions race on module-level patches.** `_patched_runtime_paths` in [runtime_test_cli.py:111](mr1/runtime_test_cli.py#L111) mutates module globals; running 4 parallel threads can momentarily cross-contaminate the `_CONTEXT_PATH` / `_RAG_DIR` constants. None of the findings in this report depend on cross-session state, but a sequential rerun (`--jobs=1`) is the cleaner default for tight QA. A process-based runner would eliminate this risk; recommended for QA session 2.
- **Scope.** This pass did not exercise: cross-session resume, the inbox triage background loop under load, workflow watchers/triggers, capability runners beyond the one approval that fired, the Kazi runner, dataflow/scheduler corner cases, or `claude` CLI error paths. Those are tagged in §4.

— end of session 1.
