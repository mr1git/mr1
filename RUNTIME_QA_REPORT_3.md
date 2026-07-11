# MR1 Runtime QA Report — Session 3 (Pre-MCP Production Readiness Audit)

**Date:** 2026-06-29
**Mode:** Code-reading audit + targeted static investigation. No new Claude CLI scenario runs.
**Baseline assumed correct:** Session 1 (32→35 scenarios) and Session 2 (50 scenarios, 0 findings, 0 crashes), routing probe 0/21 mismatches, StateManager `RLock` added, WORKFLOW-3/IDENTITY fixes assumed landed.
**Mission:** *Invalidate confidence.* Find what deterministic scenario testing structurally cannot surface, with a specific eye on whether MR1 is ready to integrate external capabilities (MCP).

> The previous two sessions tested *behavior of the runtime as driven through NL + slashes*. This session audits *the runtime's structural assumptions*: what happens at scale, under failure, over long uptime, and — most importantly — what governance actually covers once tools can write files, run shells, and call APIs.

---

## 0. Headline

**MR1's governance story is real but partial, and the part it does not cover is exactly the part MCP will land on.**

There are **two capability planes** in this system:

1. **The governed plane** — workflow tool/watcher tasks and direct `CapabilityRunner` calls. This plane is genuinely well-built: a deterministic `PolicyEngine`, scope checks, an approval store, per-attempt audit records, and a causally-linked event timeline. The QA praise is earned *here*.

2. **The brain plane** — MR1's own reasoning loop *is* a `claude` subprocess launched with `--allowedTools Read,Write,Edit,Bash,Glob,Grep,Agent` ([mr1/agents/mr1.yml](mr1/agents/mr1.yml)). Every MRn child is the same ([mr1/agents/mrn.yml](mr1/agents/mrn.yml)). These tools execute **inside** the subprocess and **never touch** the PolicyEngine, the approval store, the scope checker, or the audit log. MR1 reads back only the final `result` text and discards every intermediate `tool_use` event ([orchestrator/process.py:116-144](mr1/orchestrator/process.py#L116-L144)). **MR1 has no record of what its own brain did.**

Today this is masked because the brain *usually narrates instead of acting* (Session 1's HALLUCINATION-1 is the benign symptom of a system whose brain has live Write/Bash and chooses to talk about it). The moment MCP tools are attached to the brain's CLI — the natural integration point — they inherit plane 2: **ungoverned and unobserved.**

So the answer to the brief's question — *"is MR1 ready for its next phase: integration with external capabilities?"* — is:

- **Not as-is, if MCP tools attach to the brain.** The entire approval/scope/audit apparatus is bypassed.
- **Conditionally yes, if MCP tools are onboarded as policy-metadata'd capabilities on the governed plane** — but only after the event-log and scheduler scaling defects below are fixed, because those fail first regardless of MCP.

The rest of this report is the evidence and the surrounding findings.

---

## 1. Production Readiness Report (findings by severity)

### CRITICAL

#### C-1 · The brain plane bypasses all capability governance and is unobservable
- **Area:** capability execution, observability, safety
- **Evidence:** [mr1/agents/mr1.yml](mr1/agents/mr1.yml) (`allowed_tools: Read, Write, Edit, Bash, Glob, Grep, Agent`); [orchestrator/root.py:472-475](mr1/orchestrator/root.py#L472-L475) passes these straight into `MR1Process`; [orchestrator/process.py:78-95](mr1/orchestrator/process.py#L78-L95) forwards them as `--allowedTools`; [orchestrator/process.py:116-144](mr1/orchestrator/process.py#L116-L144) parses only `type == "result"` and `session_id`, ignoring every `tool_use`/`tool_result` stream event.
- **Why it matters:** The `PolicyEngine`, `CapabilityApprovalStore`, `ScopeContext`, and `CapabilityAuditWriter` — the machinery Sessions 1–2 graded "approvals/observability hardened" — govern **only** workflow tasks and `CapabilityRunner`. They do not govern the brain. The brain can already write/edit files and run arbitrary Bash in MR1's cwd with zero MR1-side audit, approval, scope enforcement, or timeline event. "Direct-response is safe in practice" (Session 1/2) is luck, not a guarantee: the capability is present; only the brain's disposition restrains it.
- **MCP implication:** Wiring an MCP server into the brain's CLI (via `--mcp-config`/`.mcp.json`) places those tools on this same ungoverned plane. The audit log will show nothing; approvals will never fire; scope will never be checked.
- **Minimum bar before MCP:** decide explicitly which plane MCP lives on, and if any MCP/brain tool can mutate state, capture the subprocess's `tool_use` stream into the event log so the brain plane becomes observable.

### HIGH

#### H-1 · Brain-turn retry re-executes side effects (non-idempotent retry)
- **Area:** failure recovery, capability execution
- **Evidence:** [orchestrator/process.py:59-68](mr1/orchestrator/process.py#L59-L68). On *any* error after a turn that had a session id, `send()` clears the session and re-invokes the **same message from scratch**. Timeout (1800s) and "malformed stream-json" both reach this path.
- **Why it matters:** If the first invocation performed side effects (wrote a file, ran a Bash command, spawned an agent, and — with MCP — mutated a repo or called a non-idempotent API) and *then* the output stream failed to parse or the turn timed out, the retry repeats those effects. Reachable **today** because the brain has Write/Edit/Bash; strictly worse with MCP. There is no idempotency key, no "did the first attempt commit?" check.

#### H-2 · Global event log is O(n) per append → O(n²) per session, and never bounded
- **Area:** scalability, observability, operational readiness
- **Evidence:** [event_log.py:290-317](mr1/event_log.py#L290-L317). A single `append_event` calls `_load_events_locked()` (full file read + per-line JSON parse) for the dedup check, again via `_next_event_index_locked()`, and again inside `_resolve_correlation_id_locked` / `_resolve_parent_event_id_locked` (`_find_latest_locked` reverses a full load). One `events.jsonl` is shared by the scheduler, root, `CapabilityRunner`, and approval store ([scheduler.py:858](mr1/scheduler.py#L858), [root.py:421](mr1/orchestrator/root.py#L421), [capability_runner.py:98](mr1/capability_runner.py#L98)). It is never rotated or compacted.
- **Why it matters:** At the brief's "millions of timeline events," every new event re-parses millions of lines several times. Emit latency grows linearly with history; total cost is quadratic. This is the single component most likely to fall over first under sustained daily use, independent of MCP.

#### H-3 · Scheduler reloads and re-parses every workflow on disk every second, forever
- **Area:** scalability, scheduler degradation
- **Evidence:** [scheduler.py:919-925](mr1/scheduler.py#L919-L925) (auto-started 1.0s loop) → [scheduler.py:1069-1084](mr1/scheduler.py#L1069-L1084) `tick()` → [discovery.py:51-57](mr1/scheduler_core/discovery.py#L51-L57) `active_workflows()` → [workflow_store.py:163-188](mr1/workflow_store.py#L163-L188) `list_workflows()` opens and `json.load`s **every** `<wf>/workflow.json`. Terminal workflows are filtered out *after* being fully loaded, and nothing ever archives or deletes them (no `rmtree`/prune path exists outside `memory_reset`).
- **Why it matters:** Per-tick cost scales with **cumulative historical** workflow count, not active count. At 10,000 lifetime workflows the scheduler parses ~10,000 JSON files per second forever. The system gets monotonically slower the longer it is used. `WorkflowQueryService.get_task()` ([discovery.py:36-40](mr1/scheduler_core/discovery.py#L36-L40)) is similarly an O(workflows) full scan.

#### H-4 · State file write is not crash-safe; corruption silently discards all orchestrator state
- **Area:** failure recovery, state-machine correctness
- **Evidence:** [state.py:91-96](mr1/orchestrator/state.py#L91-L96) writes a `.tmp` and `rename`s but **never fsyncs** before the rename (contrast [event_log.py:313-315](mr1/event_log.py#L313-L315) which does). [state.py:50-69](mr1/orchestrator/state.py#L50-L69) `_load_or_init` catches `JSONDecodeError`/`KeyError` and **silently reinitializes a brand-new session**.
- **Why it matters:** Power loss / kill during save, or a torn write on a non-ordered filesystem, can leave a zero-length or partial `mr1_state.json`. On next start MR1 silently throws away tasks, decision history, reference aliases, the pending workflow draft, and the resumable `claude_session_id`, and presents as a fresh session. The "corrupted → reinitialise" comment frames data loss as recovery.

### MEDIUM

#### M-1 · Workflow-mode risk threshold is 1.00 — risk never triggers approval in workflows
- **Area:** capability execution, runtime invariants
- **Evidence:** [capability_policy.py:31-35](mr1/capability_policy.py#L31-L35) sets `mr1`/`mrn` workflow threshold to `1.00`; [capability_policy.py:1130-1138](mr1/capability_policy.py#L1130-L1138) gates only when `risk_score > max_risk`. `shell_command` (risk 1.0) and `write_file` (0.65) therefore pass the risk gate in workflow mode unconditionally; only the scope check remains. For root-owned workflows, scope includes the **entire workspace_root** ([capability_policy.py:840-841](mr1/capability_policy.py#L840-L841)).
- **Why it matters:** Once write/shell capabilities are wired into the workflow runner, an MR1-authored workflow can write or execute anywhere under the workspace with **no approval** — risk is fully trusted in workflow mode. Combined with Session 1 WORKFLOW-1 ("simple" NL workflows auto-submit), a single NL turn can reach unattended in-scope side effects. The approval gate, in practice, protects only (a) direct-mode high-risk calls and (b) out-of-scope paths.

#### M-2 · Event IDs dedupe on a millisecond bucket → silent drop of genuine duplicates
- **Area:** observability, state-machine correctness
- **Evidence:** [event_log.py:111-150](mr1/event_log.py#L111-L150) builds the id from content + a millisecond-bucketed timestamp; [event_log.py:291-294](mr1/event_log.py#L291-L294) returns the existing event when the id already exists.
- **Why it matters:** Two legitimately distinct events with identical fields within the same millisecond (e.g. two identical capability requests, rapid repeated lifecycle actions, batch fan-out) collapse into one. The audit/causal trail silently loses events with no error. High-frequency capability execution (MCP) makes same-ms collisions more likely.

#### M-3 · Event append aborts the caller on backward wall-clock movement
- **Area:** reliability, long-running behavior
- **Evidence:** [event_log.py:304-306](mr1/event_log.py#L304-L306) raises `RuntimeError` when `time.time_ns() < _LAST_APPEND_NS`. `time.time_ns()` is wall-clock, not monotonic.
- **Why it matters:** Over weeks of uptime, an NTP step-back (routine) makes `emit` raise. Whatever turn, tick, or capability execution triggered that emit can abort. A monotonic source (`time.monotonic_ns`) for ordering would remove the failure mode.

#### M-4 · Shared event log has no cross-process lock
- **Area:** concurrency, recoverability
- **Evidence:** [event_log.py:99-100](mr1/event_log.py#L99-L100), [event_log.py:281-316](mr1/event_log.py#L281-L316). `_LOCKS`/`_LAST_APPEND_NS` are per-process module globals. Session 2 confirmed cross-session resume on a shared runtime root is a supported scenario.
- **Why it matters:** Two MR1 processes appending to the same `events.jsonl` can each compute the same `next_event_index` and both write it (duplicate indices, broken ordering invariant); lines larger than `PIPE_BUF` can interleave/tear and are then silently skipped by `_load_events_locked`. The single-writer assumption is implicit and unenforced.

#### M-5 · Approval lifecycle is incomplete: no expiry, and grants don't resume work
- **Area:** state-machine correctness, operational readiness
- **Evidence:** `expired` is a declared status ([capability_policy.py:28](mr1/capability_policy.py#L28)) but **nothing ever transitions an approval to it** (grep across `mr1/` shows only the allow-set, a re-open branch, and a doctor counter). [cli/capabilities.py:199-204](mr1/cli/capabilities.py#L199-L204) explicitly tells the user an approval grant "does not automatically rerun the failed task."
- **Why it matters:** A capability that hits the approval gate fails the task permanently and parks a `pending` approval that never expires. To complete, a human must both approve **and** manually `/workflow rerun`. Unattended workflows cannot get past a gated capability. For a system meant to run gated workflows, every approval is a hard stop with manual fan-in.

#### M-6 · Inbox-triage background loop swallows all exceptions
- **Area:** operational readiness, observability
- **Evidence:** [orchestrator/root.py:3000-3025](mr1/orchestrator/root.py#L3000-L3025) — `except Exception: pass`, no log, no event.
- **Why it matters:** A persistent triage failure or a single poisoned inbox message silently disables the background actor while the system looks healthy. There is no counter, no event, and no surfaced error — the failure is invisible to an operator.

### LOW / data-quality

#### L-1 · Capability onboarding is double-bookkeeping and won't scale to many tools
- **Evidence:** every capability needs a hand-written entry in `_CAPABILITY_METADATA_RAW` ([capability_policy.py:424-803](mr1/capability_policy.py#L424-L803)) **and** a branch in `CapabilityRunner._dispatch`'s if-chain ([capability_runner.py:378-591](mr1/capability_runner.py#L378-L591)). Registration fails closed without metadata ([capabilities.py:63](mr1/capabilities.py#L63), [capability_policy.py:806-809](mr1/capability_policy.py#L806-L809)) — good for safety, but it means hundreds of MCP tools require hundreds of synchronized edits in two files.

#### L-2 · `load_workflow` / `_read_json_file` raise on a torn single file
- **Evidence:** [workflow_store.py:154-161](mr1/workflow_store.py#L154-L161) and [workflow_store.py:347-352](mr1/workflow_store.py#L347-L352) have no `try/except` (unlike `list_workflows`, which skips bad files). A partially-written `workflow.json`/`result.json` crashes the direct loader rather than degrading gracefully.

#### L-3 · ID space is 24 random bits with no store-level uniqueness check
- **Evidence:** [scoped_agents.py:48](mr1/scoped_agents.py#L48), [workflow_models.py:96-100](mr1/workflow_models.py#L96-L100) use `ts + uuid4().hex[:6]`. Fine today; at high create rates within a single timestamp tick, birthday collisions become non-negligible, and there is no uniqueness enforcement at creation (compounds Session 2 IDENTITY-2).

---

## 2. Architectural risks (not necessarily bugs)

1. **The two-plane split is implicit and undocumented.** Nothing in the code names "governed vs. brain" planes; an MCP integrator will reasonably attach tools to the brain and silently defeat the whole policy system. This is the highest-leverage architectural debt.
2. **Trust model collapses risk into scope in workflow mode.** The policy engine encodes nuanced per-capability `risk_score`s, but the workflow threshold of 1.00 makes them inert there. The risk taxonomy is decorative for the most dangerous mode.
3. **Append-only, full-scan persistence everywhere.** Event log, per-workflow event logs, and the workflow directory are all unbounded and read-in-full. The design docstrings ("Phase 1 counts are small," "switch to a summary cache later") acknowledge this is deferred work — but it is deferred *past* the scale the brief asks about.
4. **No archival / retention / rotation anywhere.** Weeks of uptime means linear disk growth (events, workflows, audit records, attempt dirs, messages) with no GC. Disk free space is an unmonitored cliff.
5. **Manual capability registry won't track an external tool ecosystem.** MCP servers advertise tools dynamically; the hardcoded metadata dict + dispatch if-chain is a static, two-place registry. A bridge (derive metadata from MCP tool annotations, dispatch generically) is needed, or onboarding becomes the bottleneck.
6. **Approvals require human fan-in to complete.** No expiry, no auto-resume — fine for an attended tool, structurally limiting for "daily autonomous use."
7. **Brain-session resumption hides prompt drift.** `--append-system-prompt` is only sent on a fresh session; resumed turns rely on Claude's session memory for the safety/identity prompt ([process.py:92-95](mr1/orchestrator/process.py#L92-L95)). Safety instructions are effectively set-once-per-session.

---

## 3. Missing runtime invariants (should be explicitly enforced)

| # | Invariant | Currently | Where it breaks |
|---|---|---|---|
| I-1 | Every side effect MR1 performs is observable in the timeline | **Unenforced** | brain-plane tool calls invisible (C-1) |
| I-2 | Every executed capability passed the policy gate | **Unenforced** | brain plane bypasses PolicyEngine (C-1) |
| I-3 | A turn's side effects execute at most once | **Unenforced** | retry re-execution (H-1) |
| I-4 | `event_index` is globally unique and contiguous | **Unenforced** cross-process | no cross-process lock (M-4) |
| I-5 | Every emitted event is persisted exactly once | **Violated** | ms-bucket dedup drop (M-2) |
| I-6 | State is always parseable or recoverable without data loss | **Violated** | silent reinit on corruption (H-4) |
| I-7 | `risk_score > threshold` ⇒ approval, in every mode | **Violated** in workflow mode | threshold = 1.00 (M-1) |
| I-8 | Every pending approval eventually reaches a terminal status | **Unenforced** | no expiry (M-5) |
| I-9 | Scheduler work per tick is bounded by *active* workflow count | **Violated** | full historical rescan (H-3) |
| I-10 | Event ordering uses a monotonic clock | **Violated** | wall-clock + abort-on-skew (M-3) |
| I-11 | Agent and workflow identities are unique | **Unenforced** at store | (L-3, Session 2 IDENTITY-2) |
| I-12 | Cancelled/terminal workflows never re-execute | Assumed fixed (Session 2 WORKFLOW-3) — keep as an explicit guard, not a per-call check |

---

## 4. Suggested monitoring (metrics, counters, health checks, alerts)

**Event system**
- Gauge: `events.jsonl` size (bytes) and line count. Alert above a threshold → trigger rotation/compaction.
- Histogram: `event_append_latency_ms` (p50/p99). Alert on upward drift → O(n²) biting.
- Counter: `event_append_rejected_monotonic` (M-3), `event_dedup_dropped` (M-2). Both should be ~0; non-zero = data loss.

**Scheduler**
- Histogram: `tick_duration_ms`; counter `workflows_loaded_per_tick`. Alert when `tick_duration_ms > tick_interval_s` (falling behind).
- Gauge: terminal vs. non-terminal workflow count on disk. Alert on unbounded terminal growth → archival overdue.

**Brain plane (highest priority pre-MCP)**
- Counter: `brain_tool_invocations` by tool name — **currently zero visibility**; requires capturing `tool_use` from the stream. This is the #1 observability gap to close before MCP.
- Counters: `brain_turn_timeouts`, `brain_turn_retries` (each retry = possible double-execution, H-1), `brain_parse_errors`.
- Histogram: `brain_turn_duration_ms`.

**Persistence / recovery**
- Counter: `state_reinit_events` (H-4) and `state_save_failures`. Any reinit is a data-loss alert.
- Gauge: disk free on the runtime root. Hard alert near full.

**Approvals**
- Gauge: oldest `pending` approval age (M-5). Alert past N minutes (no expiry today).
- Counter: approvals granted vs. tasks subsequently rerun (detect stuck gated tasks).

**Background actors**
- Counter: `inbox_triage_errors` (M-6, currently swallowed). Surface it.
- Heartbeat: last successful inbox-triage tick timestamp; alert on staleness.

**Identity**
- Gauge: agent count; counter: duplicate-title collisions at creation (L-3).

---

## 5. Final production readiness score

Scored 0–10 (10 = production-trustworthy at the brief's target scale and for MCP integration). Each score explains itself.

| Dimension | Score | Rationale |
|---|---|---|
| **Runtime correctness** | **7/10** | The deterministic core — scheduler state machine, `PolicyEngine`, workflow store atomicity, event causal graph — is well-structured, and Sessions 1–2 are real. Held back by unenforced invariants (I-1…I-5) that deterministic scenario tests can't surface: at-most-once execution, event uniqueness, and the dedup drop. |
| **Reliability** | **5/10** | Single-process happy path is solid. But the failure-injection surface the brief names is largely unhandled: backward-clock `RuntimeError` (M-3), non-idempotent brain retry (H-1), silent state reset on corruption (H-4), and silently-swallowed inbox failures (M-6). Weeks-of-uptime hazards are unaddressed. |
| **Observability** | **5/10** | Excellent on the governed plane — correlation IDs, parent-linked events, per-attempt audit records, `runtime_turn_decided` on every turn (Session 2). Blind on the brain plane: MR1 cannot see what its own brain did (C-1). Since the brain plane is where MCP risk concentrates, the net is middling. |
| **Recoverability** | **4/10** | `tmp`+`rename` gives atomicity but no `fsync` (H-4); corruption silently resets state; partial state write loses everything resumable; CLI failure mid-turn can double-execute (H-1); torn single files crash the direct loaders (L-2). No replay/repair tooling. Workflow `list` is partially resilient; little else is. |
| **Scalability** | **3/10** | Two unbounded, super-linear hot paths — event log O(n²) per session (H-2) and scheduler O(total-workflows)/second (H-3) — plus zero archival/rotation. Perfectly fine at QA scale (50 scenarios); structurally unfit for 10k workflows / millions of events as posed. |
| **Maintainability** | **6/10** | Clean module boundaries, strong docstrings, deterministic policy, fail-closed registration. Dragged down by capability double-bookkeeping (L-1), intricate event parent/correlation resolution, and the **undocumented two-plane split** that will mislead MCP integrators. |

**Overall verdict:** MR1 is a well-engineered orchestration core with a genuine, auditable governance plane — but it is **not yet ready to safely gain large numbers of external tools**, for two independent reasons:

1. **Governance gap:** the brain plane (the obvious MCP attachment point) is ungoverned and unobserved (C-1). MCP must be onboarded as policy-metadata'd capabilities on the governed plane, and the brain's tool stream must be captured, before external tools are safe.
2. **Scaling gap:** the event log (H-2) and scheduler (H-3) degrade super-linearly and without bound — these fail first under daily use, *independent of MCP*, and should be fixed before any capacity increase.

The recommended sequencing before MCP: **(a)** make the brain plane observable + decide its tool policy (C-1), **(b)** fix event-log and scheduler scaling + add archival/rotation (H-2/H-3), **(c)** close the crash-safety and retry-idempotency gaps (H-4/H-1), **(d)** restore the workflow-mode risk gate and approval lifecycle (M-1/M-5). None of these were reachable by the deterministic scenario suite, which is exactly why the prior sessions reported 0 findings.

---

## 6. What this session did *not* test (honest scope)

- No new live Claude CLI runs; H-1/C-1 are demonstrated from code paths, not reproduced end-to-end. They are reachable; a live repro would strengthen them.
- Scaling claims (H-2/H-3) are derived from algorithmic reading, not load-tested. A 10k-workflow / 1M-event microbenchmark would quantify the curve and is the recommended next artifact.
- The brain's *actual* propensity to call Write/Bash on a `direct_response` turn was not measured; C-1 is about the **absence of a guardrail**, not a measured exploit rate.
- Cross-process event-log corruption (M-4) is argued from the locking model, not reproduced with two concurrent processes.

— end of session 3.
