# Phase A — Autonomous Operation: Implementation Report

**Date:** 2026-07-12
**Baseline:** 1133 deterministic tests, 50 QA scenarios / 0 findings (Session 5)
**Now:** **1373 deterministic tests**, 50 QA scenarios / 0 findings (jobs=1 and jobs=4), 10 000-tick soak green

MR1 is no longer an interactive orchestration runtime. It starts headlessly, holds
persisted objectives, plans only when something changed, executes risk-1.0
capabilities unattended under objective-scoped consent, recovers within bounded
budgets, escalates to a human when it cannot continue safely, and pauses, stops,
or halts predictably — surviving restart with its authority and lifecycle intact.

---

## 1. The finding that shaped the work

The Opus plan's central claim was verified against the committed tree and is
**correct**: the "consented-autonomy tier" Sessions 4 and 5 reported as built did
not exist.

- `_approved_override_matches` returned true only for `approval_scope == "single_use"`.
- An approval's ID is a SHA-256 over `(actor, capability, mode, args, scope, workflow_id, task_id)` — bound to one invocation in one workflow.
- `apply_decision` raised `ValueError` on `grant_scope` without a `requested_scope_path`, which a *risk* block never populates. So `approvals approve <id> --grant-scope` on a risk-blocked shell task errored out.

Consequence, as predicted: a recurring objective that shells out minted a new
`workflow_id` per run, hashed to a new approval, and asked a human forever.
There was no standing consent to time-bound — the mechanism had to be built.

---

## 2. Checkpoint-by-checkpoint

Each checkpoint landed independently, with focused tests plus a full-suite
regression gate before the next began.

### A0 — Clock seam  (1133 → 1149)
**New:** `mr1/clock.py` — `Clock` protocol, `SystemClock`, `VirtualClock` (`advance`, `sleep`, `wait`), `parse_iso`.
**Changed:** `mr1/scheduler.py`, `mr1/scheduler_core/watcher_runtime.py`, `mr1/capability_policy.py`.
**Tests:** `tests/test_clock.py` (14), `tests/test_scheduler_clock.py` (3 — including one pinning that default behaviour is still real time).
**Invariant:** every timestamp the autonomy layer reasons about comes from an injectable clock. Two real seam gaps surfaced later and were fixed: `CapabilityApprovalRequest.created_at` and `Workflow.created_at` were still real-time, which silently broke TTL comparison and made the stuck-workflow sweep unfireable under virtual time.

### A1 — Fail-loud loops  (1149 → 1156)
**Changed:** `mr1/scheduler.py` (`_run_loop` no longer `except Exception: pass`), `mr1/event_log.py` (new event types), `mr1/orchestrator/root.py` (wires `state.record_runtime_error` as the sink).
**Tests:** `tests/test_scheduler_fail_loud.py` (7).
**Invariant:** a scheduler tick failure increments a counter, persists a runtime error, emits `scheduler_tick_failed`, and the loop survives. A wedged scheduler and an idle one are now distinguishable from outside the process. `Scheduler.metrics()` exposes tick count, error count, consecutive errors, last duration, last error.

### A2 — Service + control plane  (1156 → 1180)
**New:** `mr1/autonomy/control.py` (`ControlPlane`, `ControlState`, `ServiceLock`), `mr1/autonomy/health.py` (`HealthReporter`, heartbeat, `health.json`), `mr1/autonomy/service.py` (`Supervisor`), `mr1/cli/service.py`.
**Tests:** `tests/test_control_plane.py` (10), `tests/test_supervisor_service.py` (14).
**Invariants:** `control.json` is fsync + atomic-replace; a *missing* file means `running`, an *unreadable* one means `paused` (fail closed). Singleton via `fcntl.flock` on a pidfile — the lock dies with the process, so a crash leaves no stale lock. SIGINT/SIGTERM → graceful stop. Heartbeat + gauges rewritten to `health.json` every tick.
**Verified live:** a CLI-submitted workflow reaches `succeeded` with no REPL open, and `mr1 stop` drains and exits cleanly. This also fixes the pre-existing bug that CLI-submitted workflows never advanced without an interactive MR1.

### A3 — Approval wall-clock TTL  (1180 → 1191)
**Changed:** `mr1/capability_policy.py` (`expires_at`, `is_expired`, `expire_stale_requests`, `DEFAULT_APPROVAL_TTL_S = 24h`), `mr1/autonomy/service.py` (SWEEP).
**Tests:** `tests/test_approval_ttl.py` (11).
**Invariants:** every routed approval carries a deadline; expiry emits `approval_expired`, and **fail-closed** — the task stays BLOCKED, `apply_decision` refuses a non-pending request, and an expired approval authorizes nothing. Workflow cancellation still expires approvals immediately (same code path). Legacy approvals without `expires_at` load and never self-expire.

### A4 — Objective-scoped consent grants  (1191 → 1238)  ← the keystone
**New:** `mr1/autonomy/consent.py` (`ConsentGrant`, `ConsentGrantStore`), `mr1/cli/consent.py`.
**Changed:** `mr1/capability_policy.py` (`PolicyEngine.evaluate` second override path; `CapabilityRequest.objective_id`), `mr1/scheduler_core/capability_gate.py`, `mr1/scheduler.py`.
**Tests:** `tests/test_consent_grants.py` (37), `tests/test_consent_acceptance.py` (10 — the acceptance gate).
**Invariants:** objective-scoped only; TTL **required**; revocable; risk-, scope-, and predicate-matched; a grant cannot authorize another objective, cannot widen its own scope, and cannot exceed its own `max_risk`; only root may grant above the 0.99 autonomous ceiling; standing consent never authorizes a *direct* invocation; every consent-authorized execution carries `consent_grant_id` in the audit record and the timeline, and increments `use_count` under a cross-process lock.
**One-off approval matching is untouched** — `_approved_override_matches` keeps its single-use semantics exactly. Consent is a *second, parallel* path, not an overload of the first.

### A6 — Failure recovery ladder  (1238 → 1270)
**New:** `mr1/autonomy/recovery.py` — pure `classify()` and `decide()`.
**Tests:** `tests/test_recovery.py` (32).
**Invariants:** `transient → backoff+retry`, `planning → replan`, `blocked → escalate (never self-authorize)`, `fatal/exhausted → quarantine+escalate`. Blocked wins over every other class. Backoff 30s → 2m → 8m, capped, optionally jittered (deterministic without an RNG). Global stops — consecutive failures, elapsed runtime, and *the same failure repeated* — are checked before any per-class budget, so no class can outrun them. `infrastructure_failure` is transient, so a restart does not convert in-flight work into permanent failure. A property test drives every class to exhaustion and asserts every path terminates.

### A5 + A7 — Objectives, supervisor loop, escalation  (1270 → 1311)
**New:** `mr1/autonomy/objectives.py`, `mr1/autonomy/budget.py`, `mr1/autonomy/escalation.py`, `mr1/autonomy/planner.py`, `mr1/cli/objectives.py`.
**Changed:** `mr1/autonomy/service.py` (the full ordered tick).
**Tests:** `tests/test_objectives.py` (18), `tests/test_supervisor_autonomy.py` (23), `tests/test_budget.py` (10, landed with A9).
**Invariants:**
- Tick order is `observe → gate → sweep → reconcile → plan (→ recover → escalate)`; every gate runs before any work is created.
- **A steady-state tick makes zero brain calls** — pinned by three separate tests. A *retry* replays a spec MR1 already holds and costs zero tokens; only a new plan, replan, or fallback reaches the planner.
- Every workflow an objective creates carries `objective_id` metadata — that stamp is what makes side effects attributable and what the capability gate reads back to decide which grants apply.
- **Authority preflight:** after planning and *before* submission, every capability in the spec is evaluated against the same PolicyEngine, scope rules, and grants the scheduler will use. Missing authority → escalate and ask. MR1 does not submit work it already knows it cannot finish, and never grants itself the consent it is missing.
- Escalation always does four things together: park the objective, message the inbox, emit `escalation_raised`, and say what happened / what it tried / what authority it needs / what to do next. Re-escalating the same condition is a no-op — a 60s tick does not send 60 identical messages an hour.
- Backpressure: max concurrent workflows, plans/hour, workflows/objective/day, and degraded health all stop planning without stopping draining.

### A9 — Governed inbox triage  (1311 → 1339)
**New:** `mr1/autonomy/triage.py` (`GovernedTriage`).
**Changed:** `mr1/orchestrator/root.py` (the existing loop now runs behind the governor — triage logic is not duplicated).
**Tests:** `tests/test_governed_triage.py` (18).
**Invariants:** pause / stopping / halted stop new triage actions; a corrupt control file stops them too; triage spends from the *same* budget ledger as the supervisor's planner; failures are recorded, emitted, and escalated after repeats; with no actionable unread mail there is no LLM call. A skipped pass is emitted once per reason, not once per tick.

### A8 — Soak and fault injection  (1339 → 1373)
**New:** `tests/soak/harness.py` (`VirtualClock` + `FakeBrain` + `FaultInjector` + restartable `SoakRuntime`), `tests/soak/test_soak_10k.py` (11), `tests/soak/test_fault_injection.py` (10), `tests/soak/test_crash_restart.py` (6), `tests/soak/test_control_plane_soak.py` (7).
No real LLM, no real subprocess, no sleeping.

---

## 3. Architecture actually implemented

```
mr1 serve  (headless, singleton via flock on supervisor.pid)
│
├── Supervisor.tick()            [slow: 60s default]        NEW
│     observe → gate → sweep → reconcile → plan → recover → escalate
│       • decides WHAT work exists; executes nothing
│       • calls the brain ONLY from PLAN, only on a state change
│
├── Scheduler._run_loop()        [fast: 1s]        EXISTS — unchanged design,
│       • executes everything                       now fail-loud + clock-injected
│
└── GovernedTriage               [REPL loop, now gated + budgeted]

runtime_root/
  control.json     health.json     autonomy_budget.json     supervisor.pid
  objectives/      consent_grants/                          ← new
  workflows/  agents/  events/  messages/  capability_approvals/   ← existing
```

The supervisor creates and classifies; the scheduler executes. That split is what
made this an additive layer rather than a rewrite. The scheduler's reconciliation
loop was not redesigned.

---

## 4. Differences from the Opus plan, and why

| # | Plan said | Built | Why |
|---|---|---|---|
| 1 | "emit a `supervisor_tick` event every tick" | Emit on activity, or every 60th tick as a liveness marker. The per-tick pulse lives in `health.json` (rewritten, not appended). | Appending 10 000 rows of "nothing happened" grows `events.jsonl` without bound and makes each tick cost more than the last — the exact creep the soak asserts against. Retention is Phase B; the loop must not depend on it. |
| 2 | Supervisor runs inbox triage | The REPL's loop is governed (that is A9); supervisor triage is **opt-in** (`mr1 serve --triage`), and `GovernedTriage` refuses to act on MR1's own escalations either way. | Found while building: escalations are addressed to root and land in root's own inbox, so triage saw them as unread mail — and triage may `archive`/`mark_read`. MR1 could have quietly disposed of the message asking Marwan for the consent it was blocked on. The human's inbox is not MR1's to tidy. |
| 3 | Recovery ladder as `retry → replan → fallback` | Same, plus three *global* stops checked first: consecutive failures, elapsed runtime, and identical-failure repeat. | "Repeated identical failures must eventually terminate" needs to short-circuit the per-class budgets, or a deterministic bug burns the whole replan budget re-deriving the same failure. |
| 4 | Trigger is "a WatcherRegistry spec — reuse, don't invent" | Native `immediate` / `interval` / `manual`, **plus** `{"type": "watcher", ...}` which synthesizes a probe Task and calls `WatcherRegistry.evaluate`. | Genuine reuse where it fits, but the registry has no interval watcher, and interval is what Genesis actually needs. |
| 5 | Consent grant in `PolicyEngine` | Grants own their own matching (`grant.matches(...)`); the engine duck-types them. | Keeps `capability_policy` free of an import on `mr1.autonomy` — no cycle, and the grant's rules live with the grant. |
| 6 | `halt` acts inside the supervisor | Also acts in the CLI (`halt_runtime`), idempotently. | A halt that only takes effect if a supervisor happens to be alive to read it is not a halt. Authority is revoked wherever the halt is requested. |
| 7 | Objective preflight described as "check capabilities against grants" | Preflight runs the **real** `PolicyEngine` with the real scope context and the real grants. | A hand-rolled check that drifts from the gate is worse than none: it would either block work the scheduler would allow, or wave through work that stalls on a human halfway in. |

**Not built (correctly out of scope):** retention/GC, recurring-watcher registration into `WatcherRegistry`, notification transport, the `_resume_blocked_workflow_task` store race, cross-process tick safety. These are Phase B in the plan and remain so.

---

## 5. Two real bugs found and fixed while building

1. **Concurrency cap was per-tick, not per-submission.** `max_concurrent_workflows` was evaluated once in GATE, so a single tick could submit unlimited workflows past the cap. Now the cap binds inside the plan loop, before each submission. (Caught by `test_the_concurrency_cap_stops_new_plans_but_not_draining`.)
2. **Two timestamps escaped the clock seam.** `CapabilityApprovalRequest.created_at` and `Workflow.created_at` still came from real time. The first made TTL comparison meaningless under an injected clock; the second made the stuck-workflow sweep unfireable under virtual time (a workflow "created" in the future is never old). Both now read the injected clock.

---

## 6. Results

### Deterministic suite
```
python -m pytest tests/ --ignore=tests/runtime_qa
1373 passed in ~33s        (baseline 1133 → +240 tests, 0 regressions)
```

### Runtime QA
```
python -m tests.runtime_qa.runner --jobs=1   → 50 scenarios, 0 findings, 0 crashed  (569s)
python -m tests.runtime_qa.runner --jobs=4   → 50 scenarios, 0 findings, 0 crashed  (189s)
```

### Soak (10 000 supervisor ticks ≈ 6.9 simulated days, recurring objective due every 30 min)
```
ticks                 10 000          tick errors            0
objective runs        323 ok / 0 fail stuck objectives       0
brain calls           323             (== runs; zero idle-tick calls)
tick median, first500 0.637 ms        objective history      25 entries (bounded)
tick median, last 500 0.734 ms        supervisor_tick events 484 (not 10 000)
tick p95,     last 500 1.463 ms       health.json            constant size
events.jsonl          2.2 MB          (7.3 KB per completed run — grows with work, not ticks)
```
Tick cost grew 15% across 10 000 ticks — noise, not O(n) creep.

Fault injection (10 / 10), crash-restart (6 / 6, incl. 50 mid-flight crashes), control plane (7 / 7) all green.

---

## 7. Definition of done — verified

Proven live with a real supervisor process (real scheduler, policy engine, consent
store; only the brain stubbed, since a real one spawns `claude`):

```
objective status: active  successes=1  brain_calls=1
grant use_count:  1
workflow:         succeeded   task: succeeded
stdout:           command exited 0: echo genesis ran unattended

timeline:
  objective_planned      planned: Genesis                    obj-20260712T005308-094666
  capability_allowed     capability allowed: shell_command   grant-20260712T005308-74f4ae
  consent_grant_used     consent grant used: shell_command
  capability_executed    capability executed: shell_command
```

| # | Requirement | Status |
|---|---|---|
| 1 | Start headlessly | ✅ `mr1 serve`, singleton, SIGINT/SIGTERM, heartbeat |
| 2 | Hold a persisted objective | ✅ `ObjectiveStore`, flock + atomic write |
| 3 | Plan work only when needed | ✅ zero brain calls in steady state (3 tests + soak: 323 calls / 323 runs) |
| 4 | Submit governed workflows | ✅ every one carries `objective_id` |
| 5 | Execute risk-1.0 under matching consent | ✅ acceptance gate + live run above |
| 6 | Recover from bounded failures | ✅ ladder + budgets; every path terminates |
| 7 | Escalate when it cannot continue safely | ✅ inbox + timeline + explicit park; never self-authorizes |
| 8 | Pause/resume/stop/halt predictably | ✅ incl. halt revoking all standing authority |
| 9 | Survive restart | ✅ 50 mid-flight crashes; grants/approvals keep correct status |
| 10 | Pass deterministic + QA + soak | ✅ 1373 / 50-0-0 / 10k-tick |

---

## 8. Known limitations

1. **Retention/GC is still absent (Phase B).** `events.jsonl` grows ~7 KB per completed run. The autonomy loop itself is O(1) per tick, so this is a slow disk cliff, not a correctness bug — but a year of Genesis at weekly cadence is fine, and a daily objective is not. This is the top Phase-B item.
2. **`EventLog` 50k-event cache truncation** (Session 5's "N-5 closed") is still a silent read-truncation for history queries past 50 000 events. Nothing in the autonomy layer reads full history — recovery reads the objective's own bounded history — but it should be fixed with retention.
3. **Two supervisors on one runtime root are prevented; a supervisor + an open REPL are not.** Both would drive `Scheduler._launch_ready()` for the same task (`_tick_lock` is in-process only). This is the plan's B8, unchanged. The budget ledger *is* cross-process safe, so the token/action ceiling holds regardless.
4. **At-most-once is enforced by classification, not by a commit token.** `infrastructure_failure` is retried, which is safe only for idempotent-ish missions. `Objective.idempotent` records the operator's claim, and `mr1 objective create` warns when it is not set — but nothing enforces it. Phase C's persisted idempotency token is the real fix.
5. **A hung workflow holds its concurrency slot** until the stuck sweep escalates (24h default) and a human cancels it. The supervisor deliberately does not auto-cancel — cancellation is destructive and the human is told instead.
6. **`CompilerPlanner` inherits the compiler agent's behaviour.** If it returns `needs_confirmation`, the objective escalates as ambiguous rather than guessing. Whether the compiler produces good specs for long-horizon missions is a behaviour question this phase enforces but does not answer.

---

## 9. Exact commands

**Start the service**
```bash
python -m mr1.workflow_cli serve                       # default: 60s tick, real planner
python -m mr1.workflow_cli serve --tick-interval-s 30 --workspace-root ~/Projects/mr1
python -m mr1.workflow_cli serve --no-planner          # drain workflows only; zero brain calls
python -m mr1.workflow_cli serve --triage              # also run governed inbox triage
```

**Create an objective**
```bash
python -m mr1.workflow_cli objective create "run the weekly genesis cycle" \
    --title Genesis --kind recurring --every 7d --idempotent
python -m mr1.workflow_cli objective list
python -m mr1.workflow_cli objective show <obj-id>
python -m mr1.workflow_cli objective run <obj-id>        # queue for the next tick
python -m mr1.workflow_cli objective pause|resume|abandon <obj-id>
```

**Create / revoke consent**
```bash
python -m mr1.workflow_cli grant create \
    --objective <obj-id> --capability shell_command \
    --scope ~/Projects/mr1 --allow '^(pytest|ruff|git status)' --ttl 7d \
    --reason "genesis weekly cycle"

python -m mr1.workflow_cli grant list [--active] [--objective <obj-id>]
python -m mr1.workflow_cli grant show <grant-id>
python -m mr1.workflow_cli grant revoke <grant-id>
python -m mr1.workflow_cli grant revoke --all
```

**Pause / resume / stop / halt**
```bash
python -m mr1.workflow_cli pause     # stop creating work; keep draining
python -m mr1.workflow_cli resume    # allow planning again
python -m mr1.workflow_cli stop      # drain in-flight, then exit
python -m mr1.workflow_cli halt      # cancel running work, REVOKE EVERY GRANT,
                                     # pause every objective, exit
```

**Inspect status**
```bash
python -m mr1.workflow_cli status            # mode, heartbeat age, health, objectives,
                                             # grants, budgets, oldest pending approval
python -m mr1.workflow_cli status --json
python -m mr1.workflow_cli doctor
python -m mr1.workflow_cli timeline recent
python -m mr1.workflow_cli inbox             # where escalations land
```

**Run the suites**
```bash
python -m pytest tests/ --ignore=tests/runtime_qa    # 1373
python -m tests.runtime_qa.runner --jobs=1           # 50 / 0 findings
python -m tests.runtime_qa.runner --jobs=4           # 50 / 0 findings
python -m pytest tests/soak/                         # 34, incl. the 10 000-tick soak
python -m pytest tests/soak/test_soak_10k.py -q      # the gate that must pass before autonomy ships
```
