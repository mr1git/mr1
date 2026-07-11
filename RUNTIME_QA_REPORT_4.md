# MR1 Runtime QA Report — Session 4 (Independent Pre‑MCP Production Readiness Audit)

**Date:** 2026-06-30
**Mode:** Independent audit of the *round‑2 hardening pass* (the uncommitted working tree) against Session 3's findings. Diff reading + wiring trace + **actually running the deterministic test suite**. No new live Claude CLI scenarios; no load tests.
**Baseline:** Sessions 1–3, routing probe 0/21, 50 QA scenarios / 0 findings. Session 3 (2026-06-29) raised C‑1, H‑1..4, M‑1..6, L‑1..3. The current working tree is the fix for those.

> Scope note. My job was **not** to re‑run prior QA or rediscover fixed issues. It was to check whether the fixes are internally consistent, whether they introduced new contradictions or missing invariants, and whether MR1 is now ready to shift effort from runtime hardening to MCP. Where I make a claim of fact, I ran it or traced it; where I reason about scale/concurrency, I say so.

---

## 0. Headline

**The round‑2 hardening is substantial and mostly correct — it closes the majority of Session 3's findings with real engineering, not cosmetics.** The single most important architectural decision is right: instead of trying to *observe* the ungoverned brain plane (C‑1), the pass **disarmed** it — the brain and every MRn are now hard‑restricted to `{Read, Glob, Grep}` via `governed_brain_tools()` at all three entry points. All state mutation now flows through deterministic orchestrator code or the governed capability plane. That is the correct precondition for MCP.

**But the pass is not landable as‑is, for three concrete reasons I demonstrated or traced:**

1. **The deterministic test suite is currently red.** `7 failed, 1103 passed`. All seven failures are `shell_command`‑in‑workflow tests, broken by the M‑1 clearance change — and `test_tools.py` was **not** updated to match. The pass shipped a semantic change without reconciling its own tests.
2. **M‑1 silently makes autonomous shell impossible, which contradicts the milestone.** With `MAX_AUTONOMOUS_CLEARANCE = 0.99` and `shell_command` risk `1.00`, *every* shell command in *every* mode now requires synchronous human approval. The stated next milestone is "autonomous long‑running operation." These are in direct tension and the decision is encoded in one constant, not owned anywhere.
3. **The M‑5 approval‑resume fix has a lost‑update race with the scheduler.** `_resume_blocked_workflow_task` builds a *fresh* `WorkflowStore` with its own lock and no file lock, while the scheduler mutates the same `workflow.json` on its 1 s background thread. An approval granted at the wrong moment can be clobbered — leaving the task `BLOCKED` forever, i.e. reintroducing the exact M‑5 symptom the fix was meant to remove.

Net: **GO with prerequisites.** The hardening earned the right to *plan* MCP, but three items below must land first, because two of them (shell policy, resume race) are precisely what MCP's gated‑tool flow will lean on.

---

## 1. What the round‑2 pass actually fixed (verified)

| Session 3 | Fix in working tree | Verdict |
|---|---|---|
| **C‑1** brain plane ungoverned | Brain + MRn stripped to `{Read,Glob,Grep}`; `mr1/brain_tools.py::governed_brain_tools()` enforces it in `root`, `mrn_loop`, `inbox_triage` even if the yml is re‑widened | **Closed** by removal. Brain reads remain unaudited (confidentiality, not integrity) — acceptable. |
| **H‑1** non‑idempotent retry | `send(retriable=False)` default; both call sites omit it → auto re‑invoke path is dead | **Closed.** With a read‑only brain, re‑execution is moot anyway. |
| **H‑2** O(n²) event append | Incremental `_EventCache` (offset tail‑read, `event_by_id` O(1) dedup) | **CPU closed**; see N‑5 for the new cost. |
| **H‑3** scheduler rescans all workflows/sec | `.active_workflows.json` index; `list_active_workflows()` loads only live ones | **Closed** for workflows; the *pattern* re‑appears for agents (N‑4). |
| **H‑4** state not crash‑safe / silent reinit | `flush`+`fsync(file)`+`replace`+`fsync(dir)`; corruption now **raises** `StateCorruptionError` | **Closed**, but see N‑7 (fail‑to‑boot). |
| CONCURRENCY‑1 | `StateManager` RLock + `deepcopy` on every accessor | **Closed.** |
| **M‑1** workflow risk threshold 1.00 | `MAX_AUTONOMOUS_CLEARANCE=0.99`; `max_risk = min(mode_threshold, actor_clearance)` | **Closed** — but overshoots (N‑1/N‑2). |
| **M‑2** ms‑bucket dedup drop | microsecond‑precision normalized timestamp | **Closed** (µs collision only). |
| **M‑3** backward‑clock `RuntimeError` | monotonic abort removed; ordering via `event_index` | **Closed.** |
| **M‑4** no cross‑process event lock | `fcntl.flock` around append | **Closed for writers**; readers still unlocked (N‑6). |
| **M‑5** no expiry / grants don't resume | `_resume_blocked_workflow_task` auto‑reopens; tasks go `BLOCKED` not `FAILED`; workflow stays `RUNNING` while pending; cancel expires approvals; `approval_expired` events; denied approvals sticky (`approval_previously_denied`) | **Mostly closed** — no *time‑based* expiry; resume has a race (N‑3). |
| **M‑6** inbox loop swallows errors | `record_runtime_error()` + `inbox_triage_failed` event + `runtime_errors` ring in state | **Closed.** |
| WORKFLOW‑3/4, IDENTITY‑1/2/3 | `_ensure_workflow_mutable` (CANCELLED); cancel distinguishes states; reserved `all`; title uniqueness; control‑char reject | **Closed** — uniqueness has a scaling cost (N‑4). |

This is a genuinely strong pass. Credit where due: the clearance change even had correct ripple‑patches (root bypasses the `< 1.0` scope‑grant and approver‑clearance checks, since root itself is now `0.99`). The internal‑consistency work was careful.

---

## 2. Production Readiness Report — NEW findings

Severity reflects impact on MR1's *evolution toward autonomous, MCP‑enabled operation*, per the brief.

### CRITICAL
**None.** No new data‑loss, safety, or governance‑bypass defect. The governance story is now coherent: the dangerous plane is disarmed and the governed plane enforces clearance‑bounded risk. I state this explicitly as the brief requests.

### HIGH

#### N‑1 · The hardening pass ships a red deterministic test suite
- **Demonstrated.** `python -m pytest tests/ --ignore=tests/runtime_qa` → **`7 failed, 1103 passed`**. All seven are `tests/test_tools.py::TestShellCommandTool*` / `TestToolDataflowIntegration*`. Example: `assert consume.status is TaskStatus.FAILED` now yields `TaskStatus.BLOCKED`.
- **Cause:** `shell_command` risk `1.00` > `MAX_AUTONOMOUS_CLEARANCE 0.99` ⇒ every workflow shell task now blocks on approval instead of executing. `test_tools.py` was **not** touched in this pass (empty diff).
- **Why it's High, not cosmetic:** "0 findings / tests green" is no longer true for the working tree. A change that alters a core capability's runtime semantics must move its tests in the same commit; leaving them red means the contradiction was never reconciled — nobody decided whether the new behavior is *intended* or an *overshoot*. Merging in this state normalizes a red baseline, which erodes the very signal the three QA sessions were built on.

#### N‑2 · M‑1 makes autonomous shell impossible — contradicting the stated milestone
- **Traced.** `max_risk = min(thresholds[mode], actor_clearance)`; both terms ≤ `0.99`; `shell_command = 1.00`. Therefore shell is gated in **every** mode for **every** actor, forever. The Kazi runner and existing workflows shell out (`claude -p` via `shell_command`); the on‑disk task logs confirm shell is the workhorse.
- **The contradiction:** the next milestone is "autonomous long‑running operation," yet the most‑used autonomous capability now hard‑stops on a human on *every* invocation. Combined with M‑5 still lacking **time‑based** expiry, an unattended shell workflow parks a `pending` approval indefinitely. Session 3 flagged human‑fan‑in as "structurally limiting for daily autonomous use"; the M‑1 fix has now made it **mandatory** for the shell path.
- This is an *architectural decision* worth an explicit owner, not a silent constant. Options: keep 0.99 and accept "shell always needs a human" (then MR1 is a supervised, not autonomous, shell operator); or introduce a scoped/consented autonomy tier (e.g. a clearance‑1.0 agent or a per‑workflow "operator‑preapproved" grant) so specific trusted workflows can run shell unattended while the default stays gated.
- **Decision (2026-06-30, owner: Marwan): consented‑autonomy tier.** The `0.99` ceiling stays as the safe default (shell requires approval); a per‑workflow / trusted‑agent standing grant lets specific workflows run shell unattended. See §6a for how this reuses the existing `grant_scope` approval + `approved_override` machinery rather than a redesign.

### MEDIUM

#### N‑3 · Lost‑update race between approval‑grant resume and the scheduler
- **Traced.** `CapabilityApprovalStore._resume_blocked_workflow_task` (`capability_policy.py:1131`) does `store = WorkflowStore(self._root.parent / "workflows")` — a **new instance** with its own `threading.RLock`. `capability_runner.py:582` does the same. The scheduler mutates the *same* `workflow.json` on its auto‑tick background thread through the **shared** instance injected at `root.py:436`. `WorkflowStore` has **no file lock** (unlike `EventLog`, which just got `fcntl.flock`). Per‑instance `RLock`s do not mutually exclude.
- **Effect:** atomic `tmp.replace()` prevents *corruption*, not *lost updates*. Grant an approval (main thread) while the scheduler is processing that workflow (background thread) and one write clobbers the other. If the scheduler's stale write wins, the task stays `BLOCKED` and the granted approval never resumes it — the M‑5 bug, back as a race.
- **Why it matters for MCP:** gated MCP tools will use exactly this approve→resume path. It must be single‑writer‑safe before the ecosystem grows.
- **Fix shape (no redesign):** inject the shared `WorkflowStore` into the resume path, **or** give `WorkflowStore` the same `fcntl.flock` treatment `EventLog` received. The asymmetry (event log hardened, workflow store not) is the tell.

#### N‑4 · IDENTITY‑2 uniqueness re‑introduces the H‑3 full‑scan anti‑pattern for agents
- **Traced.** `_find_title_conflict_locked` globs and `json.load`s **every** `ag-*.json` on every `save_agent` where the title changed — which includes **every new‑agent creation**. At the brief's "thousands of agents," creation is O(n) disk reads. This is structurally the H‑3 pattern the pass just deleted for workflows; there is no active‑agent index. Secondary: the scan includes terminated/dead agents, so a tombstone permanently blocks title reuse.

#### N‑5 · Event log is now fully resident in memory, unbounded
- **Traced.** The H‑2 fix keeps `_EventCache.events` (list) + `event_by_id` (dict) for the whole log, with **no eviction** and still **no rotation/compaction**. The fix traded O(n²) CPU for O(n) permanent RAM. For "months of uptime / millions of events," memory grows linearly without bound — arguably a worse failure mode for a long‑lived daemon than the CPU curve it replaced, because it ends in an OOM cliff rather than gradual slowdown. H‑2's "never bounded" sub‑point remains open.

#### N‑6 · Incremental event reader can permanently skip a torn cross‑process line
- **Traced.** `_refresh_cache_locked` takes **no** flock (only `append` does) and sets `file_offset = handle.tell()` after `readline()`, even when the final line lacks a trailing `\n`. If another process is mid‑append at read time, the partial line fails JSON parse, is skipped, and the offset is committed *past* it — that event is never re‑read by this reader. Single‑process safe; only bites the multi‑process shared‑runtime‑root case Session 2 called "supported." Fix: advance the offset only to the last complete newline, or take the append flock on reads too.

#### N‑7 · State corruption is now fail‑to‑boot with no repair path
- **Traced.** H‑4 correctly stopped silent data loss, but `StateManager.__init__` now **raises** `StateCorruptionError` on any parse/type failure. A single corrupt byte in `mr1_state.json` means MR1 will not start until a human manually intervenes — and there is no quarantine/backup/`--repair` tooling. For an autonomous, long‑lived system, fail‑loud is the right instinct but must be paired with an operator escape hatch (rename‑corrupt‑to‑`.bad`, reinit, and emit a loud recovery event). As written, availability now hinges on never corrupting the one file that is rewritten and double‑fsynced on *every* mutation.

### LOW

- **N‑8 · Unbounded `tasks` dict, full‑state double‑fsync per mutation.** `decisions/conversation/runtime_errors` are windowed; `tasks` is never pruned. Every mutation rewrites the whole state file and now does two fsyncs. Scale‑linear write cost + linear disk growth; not a correctness bug.
- **N‑9 · L‑2 only half‑fixed.** `load_workflow` / `_load_workflow_locked` still raise on a torn `workflow.json`; only the `list_*` paths degrade gracefully. Direct loaders (mutations, resume) still crash on a partial file.
- **N‑10 · Two `CapabilityApprovalStore` instances (root + scheduler) over one dir.** Lower risk than N‑3 (approvals are per‑file by ID), but it's the same "multiple instances, per‑instance locks" shape and reinforces N‑3's root cause.
- **L‑1 (Session 3) still open.** Capability onboarding is still double‑bookkeeping (`_CAPABILITY_METADATA_RAW` **and** the `_dispatch` if‑chain). Not new, but it is the item that most directly caps MCP scale (see §6).

---

## 2b. New / still‑missing invariants (not repeating Session 3's I‑1…I‑12)

| # | Invariant | Status | Where it breaks |
|---|---|---|---|
| J‑1 | `workflow.json` mutation is serialized across **all** writers, not just within one store instance | **Violated** | N‑3 (resume path + capability runner build their own instances; no file lock) |
| J‑2 | Agent‑create cost is bounded by **active** agent count | **Violated** | N‑4 (full scan of all `ag-*.json`) |
| J‑3 | Event‑log memory footprint is bounded (eviction or rotation) | **Violated** | N‑5 (whole log resident, no eviction) |
| J‑4 | An incremental reader never advances past an incomplete record | **Violated** | N‑6 |
| J‑5 | A corrupt state file yields a *running* system without manual intervention | **Violated** | N‑7 (fail‑to‑boot) |
| J‑6 | Every capability runnable in a workflow has a reachable **non‑interactive** completion path, or is explicitly labelled human‑gated | **Unenforced / implicit** | N‑2 (shell is silently human‑gated by a constant) |
| J‑7 | A granted approval deterministically resumes its task exactly once | **Violated under concurrency** | N‑3 |

---

## 3. Architecture Review

- **Governance — now the strong point (was the weak point).** Disarming the brain plane collapses Session 3's "two‑plane" ambiguity: there is now one governed plane, and the brain is a read‑only reasoner. Clearance‑bounded risk (`min(threshold, clearance)`), sticky denials, and a real `BLOCKED→resume` lifecycle are exactly the primitives MCP needs. The residual governance gaps are N‑2 (autonomy contradiction) and N‑3 (resume race), not a missing mechanism.
- **Capability execution.** Correct and auditable on the governed plane. The two‑place registry (L‑1) is the scaling ceiling. Risk scoring is now *load‑bearing* again (M‑1) — which is good, but it means a single risk number per tool decides autonomy, and the 0.99 ceiling is a blunt instrument (everything at 1.0 is equally, permanently gated).
- **Persistence.** State: crash‑safe and concurrency‑safe now, at the cost of fail‑to‑boot (N‑7) and per‑mutation double‑fsync (N‑8). Event log: CPU‑bounded but memory‑unbounded (N‑5) and reader‑racy cross‑process (N‑6). Workflow store: atomic per file but **not** serialized across instances (N‑3). The recurring theme: **`threading.RLock` is the wrong primitive for a design that constructs multiple store instances per process** — event log fixed this with `flock`; the others did not.
- **Scheduler.** The active‑index (H‑3) is the highest‑value scaling fix in the pass. The new `BLOCKED`/pending‑approval status handling is careful (workflow stays `RUNNING`, blocked tasks skipped, cancel expires approvals). Watch for stale terminal entries left in the active index after a crash between "write terminal json" and "remove from index" — a self‑limiting perf leak, not a correctness bug.
- **Runtime lifecycle.** Brain retry is now safe‑by‑omission. One subtlety: on any brain error the session id is dropped, so the next turn starts cold — this trades the H‑1 double‑execute risk for occasional context loss (the "brain amnesia" Session 1 saw). Acceptable, worth a metric.
- **Observability.** Up from Session 3: `inbox_triage_failed`, `approval_expired`, `runtime_turn_decided`, and a bounded `runtime_errors` ring. Brain *reads* remain unaudited — fine for integrity, a confidentiality gap if the brain can read secrets.
- **Maintainability.** The explicit read‑only brain policy is a big clarity win for future MCP integrators. Dragging against it: the multi‑instance store pattern (N‑3/N‑10), the min‑clearance subtlety, the approval‑resume path reaching across module boundaries via a hardcoded `../workflows` path, and L‑1.

---

## 4. Future Risks (MCP, capability growth, scaling, ops)

- **MCP attachment point.** The C‑1 fix makes the *correct* answer enforceable: MCP tools must **not** attach to the brain CLI (now read‑only) and must onboard as governed capabilities. Good. But: (a) any MCP tool with risk > 0.99 inherits shell's fate — permanently gated (N‑2); (b) gated MCP tools depend on the racy resume path (N‑3); (c) hundreds of MCP tools × two‑place registry (L‑1) = onboarding bottleneck.
- **Capability growth.** L‑1 and N‑4 both say the same thing: MR1's registries and identity checks assume small N and full scans. An external tool ecosystem and a large agent fleet break both.
- **Scaling.** The CPU cliffs are fixed; the **memory** cliff (N‑5) and **disk** cliff (no archival/retention anywhere — events, workflows dir, audit dirs, tasks) are the next to bite. Nothing GCs.
- **Operational maintenance.** N‑7 (fail‑to‑boot) + no repair tooling + no rotation means the failure modes that appear "after months of uptime" are unattended‑hostile. Session 3's monitoring list (event size, tick duration, oldest pending approval, state‑reinit, disk free, brain retries) is still the right instrumentation and is still unbuilt.

---

## 5. Would I trust this runtime to control…?

- **Repositories / production files — conditionally yes.** The brain can no longer write; governed `write_file` (0.65) still runs unattended within `workspace_root` scope for a 0.99 actor — acceptable *if* scope discipline holds and **N‑3 is fixed** (so gated edits resume deterministically). shell is gated, which is conservative and fine here.
- **Autonomous workflows — not yet.** N‑2 (shell always human‑gated) + no time‑based approval expiry + N‑3 (resume race) means unattended shell workflows stall or strand. Autonomy is currently *supervised*, not autonomous, for anything that shells out.
- **Physical devices — no.** The lost‑update race (N‑3), fail‑to‑boot recovery (N‑7), and absence of an idempotency key on the governed plane (Session 3 H‑1's spirit: "a turn's side effects execute at most once" is enforced by *removing* brain effects, not by a commit token on the capability plane) are disqualifying for irreversible physical actions. Get an at‑most‑once execution guarantee with a persisted idempotency key before anything with real‑world side effects.

---

## 6. MCP Readiness — should MCP tools plug into the current capability system?

**Directionally yes — the governed plane is the right home, and the brain disarm makes that enforceable — but not by dropping MCP tools into it unchanged.** Minimal remaining architectural work (no redesign):

1. **A metadata bridge.** Derive `CapabilityMetadata` (risk, scope requirements, config schema) from MCP tool annotations, and dispatch MCP tools **generically** instead of adding a `_dispatch` branch per tool (kills L‑1). Without this, "hundreds of tools = hundreds of synchronized two‑file edits."
2. **A risk→autonomy policy for external tools.** Decide how MCP tool risk maps onto the `0.99` ceiling and per‑agent clearance. Today a single risk number = permanent gate at 1.0. MCP needs at least a "trusted, pre‑consented" tier so useful tools can run without a human on every call (this is the same decision N‑2 forces for shell).
3. **Fix N‑3 first.** MCP's whole value proposition on the governed plane is "block → approve → resume." That path is currently racy.

Everything else Session 3 named (capture the brain tool stream) is **obsolete** — the brain no longer has tools worth capturing. That is the right call.

---

## 6a. Consented‑autonomy tier — design sketch (reuses existing machinery, no redesign)

Decision recorded in §2/N‑2: keep the `0.99` default gate; add an explicit standing grant for trusted workflows/agents. The pieces already exist:

- `_ALLOWED_APPROVAL_SCOPES` already includes **`grant_scope`** (a persistent grant), alongside `single_use`. A standing pre‑consent is a `grant_scope` approval that is not consumed on use.
- `PolicyEngine` already honors an **`approved_override`** that bypasses the risk gate when a matching approval exists. Today it's produced by a human approving a *blocked* task; the tier just lets that same override be created *ahead of time*.
- Agents already carry `scope_grants`; the same shape can bound a capability pre‑consent to a path/scope.

So the tier is: **a revocable, scoped, standing `grant_scope` authorization keyed to `(actor_or_workflow, capability_name, scope)` that the policy engine treats as an `approved_override` even for risk `1.0`.** Concretely, the minimal work:

1. **Create path** — an operator action (slash command / CLI) that mints a standing `grant_scope` approval for a specific workflow or trusted agent + capability, without a prior block. This is the "consent."
2. **Override match** — confirm `_approved_override_matches` honors a standing (non‑single‑use, non‑consumed) grant against the ceiling, and that it does **not** silently satisfy risk‑1.0 for actors that lack the grant.
3. **Audit every unattended use** against the granting approval id (the audit writer already exists; just thread the id).
4. **Revocation + scope/expiry** — make grants revocable and ideally bounded (path scope, and a TTL — which also gives you the time‑based approval expiry M‑5 still lacks).

This keeps the safe default intact, makes autonomy an explicit, auditable, revocable opt‑in, and — importantly — is the **same** mechanism MCP tools will use for their "trusted, pre‑consented" tier (§6, item 2). Build it once, here, and MCP inherits it.

---

## 7. Go / No‑Go

**GO with prerequisites.**

The round‑2 hardening did the hard, correct thing on governance and closed most of Session 3. Effort *should* be shifting toward MCP — but three items must land first, because two of them are exactly what MCP will stand on:

- **P0 — reconcile the shell/clearance contradiction (N‑1 + N‑2).** Direction chosen: **consented‑autonomy tier** (§6a) — `0.99` stays the default gate, trusted workflows get a standing revocable grant. Ship it, update `test_tools.py` (either assert `BLOCKED` for the ungranted default case or grant consent in the fixture), and confirm Kazi and existing autonomous workflows function under a grant. Turn the suite green.
- **P0 — make `WorkflowStore` writes single‑writer‑safe (N‑3).** Inject the shared store into the resume/capability‑runner paths, or add `fcntl.flock` as was done for `EventLog`. Same primitive, same fix.
- **P1 — bound event‑log memory + add rotation/retention (N‑5), and add a state‑repair path (N‑7).** These are the "months of uptime" killers; MCP raises event/audit volume, so they get worse, not better.

Then onboard MCP as governed capabilities behind the metadata bridge (§6). Do **not** begin MCP wiring while the suite is red and the resume path is racy — you'd be building the ecosystem on top of an unreconciled semantic change and a known concurrency bug.

---

## 8. Production Score (independent; differs from Session 3 where the pass moved the needle)

Scored 0–10 for the brief's target: long‑lived, autonomous, MCP‑enabled operation.

| Dimension | Score | Rationale |
|---|---|---|
| **Correctness** | **7/10** | Core state machine, clearance‑bounded policy, resume lifecycle, and event causal graph are sound and mostly well‑tested. Docked by N‑1 (7 red tests = an unreconciled semantic change) and the N‑3 resume race. |
| **Reliability** | **5/10** | Retry‑off + crash‑safe state + concurrency‑safe `StateManager` are real gains. Held down by N‑3 (lost update), N‑7 (fail‑to‑boot, no repair), and the absence of an at‑most‑once guarantee on the governed plane. |
| **Observability** | **6/10** | Up from Session 3's 5: inbox failures surfaced, `approval_expired`/`runtime_turn_decided` events, `runtime_errors` ring. The prescribed metrics remain unbuilt; brain reads unaudited (confidentiality). |
| **Recoverability** | **5/10** | fsync + atomic replace + dir‑fsync is correct. But corruption is now fail‑to‑boot with no quarantine/replay tooling (N‑7); direct loaders still crash on torn files (N‑9). |
| **Scalability** | **4/10** | The two CPU cliffs (event O(n²), scheduler O(total wf)) are genuinely fixed — the highest‑value work in the pass. But memory is now unbounded (N‑5), agent‑create is O(n) (N‑4), `tasks` is unbounded (N‑8), and nothing archives. Fixed the fast cliffs, opened a slower one. |
| **Governance** | **7/10** | The standout improvement: brain disarmed, clearance‑bounded risk load‑bearing again, sticky denials, real resume lifecycle. Docked by the N‑2 autonomy contradiction (governance now *blocks* the milestone for shell) and the N‑3 race stranding granted approvals. |
| **Maintainability** | **6/10** | The explicit read‑only brain policy is a real clarity win for MCP integrators. Dragged by the multi‑instance store/lock pattern, the cross‑module `../workflows` reach in the resume path, L‑1 double‑registry, and — most tellingly — a red suite that signals the pass wasn't fully reconciled. |

**Verdict:** MR1 has, in substance, reached the point where runtime‑hardening effort should give way to capability expansion — the governance foundation for MCP is now real and enforceable. It has **not** quite reached a clean handoff: it ships red tests, a governance/autonomy contradiction encoded in one constant, and a concurrency bug in the approval path MCP will depend on. Close P0/P1 above and the shift to MCP is justified.

---

## 9. Honest scope of this session

- I **ran** the deterministic suite (`7 failed, 1103 passed`) — N‑1 is demonstrated, not argued. `test_tools.py` being unmodified is from the diff.
- N‑2, N‑3, N‑4, N‑5, N‑6, N‑7 are traced from the current code paths (diffs + wiring), not reproduced end‑to‑end. N‑3 in particular would be worth a two‑thread stress repro (grant an approval while the scheduler ticks the same workflow) to quantify the collision window.
- No new live Claude CLI scenarios and no load tests were run; the scaling claims (N‑4/N‑5) are algorithmic, consistent with how Session 3's H‑2/H‑3 were argued and then fixed.
- I did not re‑audit anything Session 3 marked closed unless the round‑2 fix changed it.

— end of session 4.
