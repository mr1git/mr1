# MR1 Runtime QA Report — Session 5 (Final Independent Runtime-Hardening Audit)

**Date:** 2026-07-10
**Mode:** Independent meta-audit. Read Sessions 1–4, the runtime_qa suite, and traced the current committed tree (`7ad4a3a Runtime Hardening basically complete`). Verified — not assumed — the Session-4 P0/P1 open items against live code. No new Claude CLI scenarios, no load tests.
**Baseline claimed:** 1133 deterministic tests passing, 50 QA scenarios / 0 findings / 0 crashes, routing probe 0/21.
**Mandate:** *Not another bug hunt.* Answer whether runtime hardening has reached diminishing returns and whether engineering effort should shift elsewhere.

---

## 0. Headline

**Runtime hardening has reached diminishing returns for correctness, and the orchestration runtime is no longer the primary engineering bottleneck.**

The four-session arc converged. Session 1 closed the NL/routing leaks, Session 2 closed lifecycle/identity, Session 3 exposed the structural governance and scaling gaps, and Session 4 reviewed the round-2 hardening and left three concrete prerequisites open (shell/clearance contradiction + red suite, WorkflowStore resume race, event-log memory + state repair). **I verified all three are now closed in the committed tree:**

- **Suite is green** — `1133 tests collected`, the 7 red `shell_command` tests reconciled. ([tests/](tests/))
- **WorkflowStore is single-writer-safe** — a real cross-process `fcntl.flock` mutation lock now wraps every write ([workflow_store.py:165](mr1/workflow_store.py#L165), `self.locked()`), closing the N-3 lost-update race.
- **Event cache is bounded** — `_MAX_CACHE_EVENTS = 50_000` sliding window with `popleft` eviction ([event_log.py:109](mr1/event_log.py#L109), [event_log.py:693](mr1/event_log.py#L693)), closing the N-5 OOM cliff.
- **State corruption has an operator escape hatch** — `repair-state` CLI + `doctor.repair_state_file()` ([doctor.py:1620](mr1/doctor.py#L1620), [cli/main.py:201](mr1/cli/main.py#L201)) that quarantines rather than silently reinitializes, closing N-7.
- **Agent identity is O(1)** — `.title_index.json` replaces the N-4 full scan ([scoped_agents.py:582](mr1/scoped_agents.py#L582)).
- **Governance is coherent** — brain disarmed to `{Read,Glob,Grep}`, clearance-bounded risk, `grant_scope`/`approved_override` consented-autonomy machinery in place ([capability_policy.py:1074](mr1/capability_policy.py#L1074), [capability_policy.py:1259](mr1/capability_policy.py#L1259)).

What remains is **not** correctness debt. It is a small, well-understood set of *operational-longevity* items (retention/GC, wall-clock approval expiry) and the *capability-ecosystem* work MCP will need. Those are different disciplines. The runtime core is done enough to build on.

---

## A. Are there still major runtime correctness risks?

**No — none that materially threaten long-lived operation, governance, safety, or recoverability that are demonstrable today.** Every Critical/High from Sessions 1–4 is closed. The residuals I can *demonstrate* from code are Medium-or-below and are longevity/ergonomics, not correctness:

- **Wall-clock approval expiry still absent** *(demonstrated).* `expire_requests_for_workflow` is called only from the scheduler's cancel/terminal path ([scheduler.py:2501](mr1/scheduler.py#L2501)); there is no `expires_at`/TTL anywhere in [capability_policy.py](mr1/capability_policy.py). An unattended workflow that blocks on an approval parks a `pending` approval **forever** until a human acts or cancels. This is a governance-completeness gap, not a safety hole — the default fails *closed* (task stays BLOCKED). It caps *unattended* autonomy, which is why it belongs on the roadmap, not in the critical column.
- **No disk retention/rotation/archival anywhere** *(demonstrated).* Grep confirms no rotate/compact/archive/prune path in the event log, workflow store, or scheduler outside `memory_reset`. Memory is now bounded (50k); disk is not. `events.jsonl`, per-workflow dirs, audit dirs, and the `tasks` state dict grow linearly forever. This is the single item most likely to bite "after months of uptime," and it is a *slow* cliff (disk-full), not a correctness bug.

Neither threatens correctness. Both threaten *uptime past a horizon nobody has reached yet.*

## B. Are there architectural contradictions remaining?

**The one real contradiction Session 4 named — "autonomous milestone vs. shell always human-gated" — has been resolved into a deliberate design, not left as an accident.** The `0.99` ceiling stays as the safe default; the consented-autonomy tier (`grant_scope` standing grant → `approved_override`) is the sanctioned path for trusted workflows to run risk-1.0 capabilities unattended. That converts a contradiction into a policy with an owner. Verified present in code ([capability_policy.py:1074-1090](mr1/capability_policy.py#L1074), [capability_policy.py:1259](mr1/capability_policy.py#L1259)).

The residual *tension* (not contradiction): the consented-autonomy grant has **no TTL**, so "unattended autonomy" currently means "a standing grant that never expires." That is safe (revocable, scoped, audited) but not yet *time-bounded*. Pairing the grant with the missing wall-clock expiry (A) closes both with one mechanism — which is exactly the leverage Session 4 predicted.

No lifecycle contradictions remain: terminal workflows are guarded (`_ensure_workflow_mutable`), the BLOCKED→resume lifecycle is real, cancel expires approvals.

## C. What assumptions break first under scale / uptime?

In order of when they bite:

1. **Disk, under months of uptime** *(future-scalability, demonstrated absence of GC).* No archival anywhere. First to fail on a genuinely long-lived daemon. Slow, monitorable, non-corrupting.
2. **Title-space exhaustion, under thousands of agents over time** *(future-scalability, demonstrated).* Terminated agents' titles are **permanently reserved** by design ([scoped_agents.py:33](mr1/scoped_agents.py#L33)). Over months of create/kill churn, the usable title namespace monotonically shrinks and never recovers. Data-quality, not correctness.
3. **Capability onboarding, under hundreds of capabilities** *(architectural, L-1 still open).* The two-place registry (`_CAPABILITY_METADATA_RAW` + `_dispatch` if-chain) is a synchronized-double-edit per tool. This is the ceiling MCP hits first — it's an *ecosystem* bottleneck, not a runtime one.
4. **Cross-process torn-line reader** *(architectural, N-6 residual).* `_refresh_cache_locked` still takes no flock and commits the offset past an incomplete final line ([event_log.py:667-691](mr1/event_log.py#L667)). Only bites the multi-process shared-root case; writers are now flock-safe so the window is narrow. Low.
5. **At-most-once execution on the governed plane** *(architectural, future).* Enforced today by *removing* brain side-effects, not by a persisted idempotency key. Fine until MR1 drives irreversible/physical actions — then a commit token is required. Not needed for the current or MCP-software horizon.

None of these is "the runtime is wrong." They are "the runtime was built for the scale it has, and the next order of magnitude needs GC and a registry bridge."

## D. Remaining work, triaged

**1. Critical:** *None.* Stated explicitly.

**2. High leverage (do before/with MCP):**
- **Capability metadata bridge** (kills L-1 double-bookkeeping; derive `CapabilityMetadata` from MCP annotations, dispatch generically). This is the true gate on capability scale.
- **Wall-clock TTL on approvals + consented-autonomy grants** (closes the M-5 residual *and* time-bounds the autonomy tier — one mechanism, two wins).
- **Retention/rotation/archival** for events, workflow dirs, tasks (the disk cliff; also keeps MCP's higher event volume from accelerating it).

**3. Diminishing returns (real, but low marginal safety per hour):**
- Title-tombstone reclamation (N-4 residual).
- Cross-process reader flock (N-6).
- Direct-loader graceful-degrade on torn `workflow.json` (N-9, `list_*` already degrades).
- Building the Session-3 monitoring counters (valuable for *ops*, but they observe a system that already behaves; they don't change correctness).
- Any further routing/lifecycle/identity hardening — the deterministic surface is saturated; 0/21 and 0 findings across 50 scenarios is the signal that this vein is mined out.

## E. Has complexity's center of gravity moved?

**Yes, decisively.** Future complexity will originate from the **capability ecosystem, agent behavior, and product design** — not orchestration correctness. The evidence:

- The governed plane is deterministic and audited; the dangerous plane was *removed* (brain read-only). There is no longer a large surface of "does the orchestrator do the right thing" — there is a small surface of "does this new capability's metadata/risk/scope get onboarded correctly."
- Every remaining runtime item is either (a) operational hygiene (GC, TTL, monitoring) that is bounded and well-understood, or (b) a *bridge* into the ecosystem (metadata bridge, autonomy tier) — i.e., already ecosystem work wearing a runtime hat.
- The open-ended, hard-to-bound problems from here — *which* MCP tools to trust, how agents behave over long missions, how the product surfaces autonomy to the user — are behavior/policy/UX problems the runtime can only *enforce*, not *solve*.

## F. Would I personally keep investing heavily in runtime hardening now?

**No.** I would stop heavy runtime investment and shift to the capability/MCP plane, carrying exactly three runtime items with me because MCP will stand on them: the **metadata bridge**, **approval/grant TTL**, and **retention/GC**. Everything else is maintenance.

The reason is diminishing marginal safety. Sessions 1→2 bought large safety per unit effort (routing, lifecycle). Session 3→4 bought structural safety (governance plane, scaling cliffs). This session found *zero* new correctness defects and only longevity residuals — the curve has flattened. Continuing to hunt the runtime is polishing a foundation while the house isn't built. The highest-leverage next dollar is on the capability ecosystem the governance plane was *built to host*.

---

## Deliverable 1 — Production Readiness Report

**CRITICAL: none.** *(Stated explicitly per mandate.)* No demonstrable data-loss, safety, or governance-bypass defect remains in the committed tree.

**HIGH: none demonstrable.** The Session-4 HIGHs (red suite N-1, autonomy contradiction N-2) are resolved (green suite, consented-autonomy tier). What's left below is Medium.

**MEDIUM**
- **No wall-clock approval/grant expiry** — unattended blocked work parks a pending approval indefinitely; the consented-autonomy grant never expires. *Demonstrated absence.* (Governance completeness / unattended-autonomy limit.)
- **No disk retention/rotation/archival** — linear unbounded growth of events, workflow dirs, tasks. *Demonstrated absence.* (Months-of-uptime cliff.)

**LOW**
- Terminated-agent title tombstones permanently reserve namespace (N-4 residual, *demonstrated*).
- Cross-process incremental reader can skip a torn final line (N-6, *demonstrated*, narrow window).
- Direct workflow loaders still raise on a torn single file (N-9, *demonstrated*; `list_*` degrades).
- Session-3 monitoring counters still unbuilt (*architectural/ops*, not a defect).
- L-1 capability double-registry (*architectural*; becomes High **only** once MCP onboarding starts).

---

## Deliverable 2 — Remaining Runtime Roadmap

**Must do** *(and these are mostly MCP-enablement, not runtime-correctness)*
1. Approval + consented-autonomy-grant **TTL** (closes M-5 residual; time-bounds autonomy).
2. **Retention/rotation/archival** for events, workflow dirs, tasks state.
3. **Capability metadata bridge** before any large tool onboarding (kills L-1).

**Nice to do**
- Session-3 monitoring counters (event size, tick duration, oldest pending approval, disk free, brain retries).
- Cross-process reader flock (N-6); direct-loader graceful degrade (N-9).
- Persisted idempotency key on the governed plane (only before irreversible/physical actions).

**Diminishing returns**
- Title-tombstone reclamation.
- Any further routing / lifecycle / identity hardening — deterministic surface is saturated.
- Additional QA scenario breadth on already-green paths.

---

## Deliverable 3 — Bottleneck Assessment

The next major source of complexity is, in order:

1. **Capability ecosystem** — onboarding, metadata/risk mapping, generic dispatch, the trusted tier. This is where MCP lands and where the two-place registry breaks. **Primary.**
2. **Agent behavior** — long-mission behavior, fan-out/recursion in practice, autonomy-tier usage patterns. Enforced by the runtime, *solved* by behavior/policy design.
3. **Product / UX** — how autonomy, approvals, and standing grants are surfaced to Marwan. A human-consent model needs a human interface.
4. **Runtime** — now *fourth*, and only the three "Must do" items above; the rest is maintenance.

The bottleneck has moved off the runtime.

---

## Deliverable 4 — Final Recommendation

**B — Pause heavy runtime work and shift focus** *(to the capability/MCP plane)*, carrying the three "Must do" runtime items as prerequisites of that shift rather than as continued hardening.

Not C ("complete except maintenance") because two Medium longevity items (TTL, retention) are real and the metadata bridge is genuinely required before MCP — that's more than maintenance. Not A ("continue hardening") because this session found zero new correctness defects and the marginal safety of further runtime hunting is near zero. The honest position is between C and A, and the action it implies is B: the runtime earned the right to stop being the focus; the next effort belongs on the ecosystem it was built to govern.

---

## Deliverable 5 — Production Scores

Scored 0–10 for the brief's target (long-lived, autonomous, MCP-bound operation). Deltas vs. Session 4.

| Dimension | Score | Δ | Rationale |
|---|---|---|---|
| **Correctness** | **8/10** | +1 | Suite green (1133), N-1/N-3 closed. Core state machine, clearance policy, resume lifecycle sound. Held from 9 only by unenforced at-most-once on the governed plane (future concern). |
| **Reliability** | **7/10** | +2 | WorkflowStore now flock-serialized (N-3 closed), state repair path exists (N-7 closed), retry-off + crash-safe state hold. Docked for no approval TTL (unattended stall) and no idempotency token. |
| **Observability** | **6/10** | 0 | Governed-plane observability is strong; `approval_expired`/`runtime_turn_decided`/`runtime_errors` present. Prescribed ops counters still unbuilt; brain reads unaudited (confidentiality). |
| **Recoverability** | **7/10** | +2 | `repair-state` + `repair_state_file` quarantine (N-7 closed) is the big lift. fsync+atomic-replace correct. Docked for direct-loader torn-file crash (N-9). |
| **Scalability** | **6/10** | +2 | Event memory bounded (N-5 closed), agent identity O(1) (N-4 closed), CPU cliffs already fixed. Docked hard for **no disk retention anywhere** and title-tombstone growth. |
| **Governance** | **8/10** | +1 | Brain disarmed, clearance-bounded risk load-bearing, consented-autonomy tier real, N-2 contradiction resolved by design. Docked for missing grant/approval TTL. |
| **Maintainability** | **7/10** | +1 | Green suite restores the signal Session 4 flagged. Read-only brain policy is a clarity win for integrators. Docked for L-1 double-registry and multi-instance store pattern residue. |

Overall: a well-engineered, auditable orchestration core that is **done enough to build the capability ecosystem on top of.**

---

## Deliverable 6 — Confidence Score

**Confidence that "the orchestration runtime is no longer the primary engineering bottleneck": 8.5 / 10 (High).**

Why high:
- Every Critical/High across four sessions is closed, and I *verified* the three Session-4 prerequisites in code rather than trusting the changelog.
- This session's independent pass found **zero new correctness defects** — only longevity residuals and one architectural ceiling (L-1) that is itself *ecosystem* work. A flat defect curve after four hardening passes is the strongest available signal of diminishing returns.
- The remaining runtime items are bounded and well-understood (GC, TTL, a bridge), not open-ended.

Why not higher:
- Scaling claims (disk cliff, title exhaustion) are algorithmic, not load-tested — consistent with how Sessions 3/4 argued and then fixed H-2/H-3, but unproven at 10k-workflow / months-of-uptime scale. A single load/soak test would move this to 9.5.
- No new *live* Claude CLI runs this session; the behavioral surface is inherited from prior sessions' 0-findings, not re-measured. If agent behavior under long autonomous missions surprises us, some of that complexity could route back to the runtime — but it would be *policy* complexity the runtime enforces, not a runtime correctness regression.

**Bottom line:** MR1 should stop spending major effort on runtime hardening. The runtime is a sound, governed, recoverable foundation. Future engineering effort should move to the capability ecosystem, agent behavior, and product design — carrying only the TTL, retention, and metadata-bridge items forward, because those are the runtime's contribution to the *next* phase rather than a continuation of this one.

— end of session 5.
