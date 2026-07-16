# Phase B — Comfortable Daily Operation: Implementation Report

**Date:** 2026-07-12 (24h soak completed 2026-07-13)
**Baseline (Phase A):** 1373 deterministic tests · 50 QA scenarios / 0 findings · 10 000-tick virtual soak
**Now:** **1492 deterministic tests** · 50 QA scenarios / 0 findings (jobs=1 and jobs=4) · 49 soak tests · **a real 24-hour wall-clock soak, passed**

Phase A made MR1 *capable* of running unattended. Phase B makes it *safe to leave
alone*: one process owns execution, history never lies, disk has a bottom, a
recurring objective survives a week of downtime without firing a week of work,
an operator can see what is stuck without reading files, pressure stops MR1
creating work rather than corrupting it, escalations have a transport, and
there is honest tooling to prove an 8h or 24h run actually happened.

---

## 1. Checkpoint by checkpoint

Each landed independently, with focused tests plus a full-suite regression gate
before the next began.

### B8 — Cross-process execution ownership  (1373 → 1385)

**New:** `mr1/autonomy/ownership.py` — `ExecutionOwnership`, `ExecutionOwner`.
**Changed:** `mr1/scheduler.py` (`execution_ownership` param, `has_execution_authority()`, delegated-tick counter, release on shutdown), `mr1/autonomy/service.py`, `mr1/orchestrator/root.py` (`enforce_execution_ownership`, startup banner), `mr1/mr1.py`, `mr1/cli/service.py`, `mr1/cli/workflows.py`.
**Tests:** `tests/test_execution_ownership.py` (12), including **two real interpreters** driving one runtime root.

The lock is `fcntl.flock` on `<runtime_root>/execution.lock`, chosen for two
properties a PID file cannot offer. It is held by the *open file description*,
so a crashed owner releases it the moment the kernel closes its files — ownership
never goes stale. And it conflicts across separate `open()` calls even within one
process, so two `Scheduler`s in one interpreter exclude each other exactly as two
processes do.

`execution_owner.json` beside it is advisory only: the lock is the authority, the
JSON is the label. A leftover label after `kill -9` reads as "not owned", which
is the failure mode B8 exists to avoid.

**Invariants:** exactly one process launches, polls, or finalizes tasks per
runtime root. A non-owner may still read every store, submit specs, and issue
control commands — it is *execution* that is exclusive, not access. A follower
promotes itself the moment the owner exits, so authority is never orphaned.
`mr1 serve` singleton behaviour (`ServiceLock`) is unchanged and independent.

Enforcement is opt-in **at process entry points** (`mr1 serve`, the REPL, the
one-shot CLI tick), not inside `Scheduler` by default. A library caller that
builds a `Scheduler` directly is still the unconditional executor — which is what
keeps every embedded use and all 1373 existing tests behaving exactly as before.

### B7 — Approval resume / store safety  (1385 → 1388)

**Audited, not assumed.** The Opus plan listed `_resume_blocked_workflow_task` as
an open race. It is not. Every `save_workflow` call site in the tree sits inside
`store.locked()` **and reloads the workflow after taking the lock** — the resume
included (`capability_policy.py:1206-1219`). `WorkflowStore.locked()` is a real
`fcntl.flock`. There was no stale load → modify → save anywhere.

**Changed (smallest localized correction):** `CapabilityApprovalStore` now accepts
a `workflow_store`, and `Scheduler` passes its own. The fresh instance the resume
used to build was cross-process *correct* but forfeited reentrancy: a future
caller deciding an approval from inside a store lock would have deadlocked
against its own flock on a second fd. Sharing the instance removes that class.

**Tests:** `tests/test_approval_resume_safety.py` (3). The load-bearing one asserts
*call ordering* — lock, then load, then save, then unlock — because an
outcome-only test passes on an idle system and only fails under a concurrent
tick. Mutation-tested: moving the load outside the lock makes it fail
deterministically. A second test proves the window excludes a real second
process; a third races 20 rounds of scheduler ticks against approval decisions.

### B1 — Retention, archival, full-history correctness  (1388 → 1411)

**New:** `mr1/event_archive.py` (`EventArchive`, `ArchiveSegment`), `mr1/autonomy/retention.py` (`RetentionManager`, `RetentionPolicy`, `RetentionReport`), `mr1/cli/maintenance.py`.
**Changed:** `mr1/event_log.py` (rotation, complete-history reads, truncation tracking, archive-aware lookup), `mr1/autonomy/service.py` (periodic retention phase), `mr1/cli/main.py`.
**Tests:** `tests/test_retention.py` (24).

**The truncation fix.** `list_events()` read a 50 000-event deque and returned it
as though it were history. Past the limit, every history query silently answered
with the tail — and nothing raised. It now returns **complete history**: served
from the cache when the cache provably holds all of it (the normal case, at the
cost it always had), and from disk (archive segments + live file) when it does
not. `recent_events()` is the explicit, by-name opt-in to the cheap window. The
memory bound is untouched; it simply stopped being mistaken for a history bound.

One existing test asserted the defect (`test_rebuild_applies_bound` expected
`list_events()` to return 8 of 12 events). It now asserts both properties
separately: the cache is still bounded, and history is still complete.

**Rotation.** `events.jsonl` seals into numbered gzip segments under
`events/archive/`, indexed by `events/segments.json`. The manifest is the
authority, not the directory listing. Segment written and fsynced → manifest
updated → live file rewritten, so a crash anywhere in that sequence either
orphans a segment (ignored, overwritten next time) or duplicates events in both
places (deduped by `event_id` on read). Nothing leaves the live file until it is
durably archived.

Rotation keeps a **tail** of recent events behind (`events_keep_recent`, default
1000). That is not a convenience: `emit()` resolves `parent_event_id` and
`correlation_id` by searching backwards through the cache, so rotating an
in-flight causal chain out from under itself would orphan the next event in it.
Event indices continue across rotation from the manifest — an index that restarts
at 1 forks the log.

**Archive, never delete.** Terminal workflows, capability audits, and snapshots
move into `<runtime_root>/archive/`, where they stay queryable and restorable.
A workflow is archivable only when it is terminal, old enough, holds no BLOCKED
task, and is referenced by no live objective and no pending approval — any doubt
keeps it. A `workflow_keep_recent` floor beats the age rule, so a quiet month
never empties the history an operator reads. **Deletion happens only when an
operator sets `purge_archives_after_days`**, and even then only inside the
archive, never against live state.

The archive writes an `.archive_index.json` ledger recording *when* it took
custody of each item. `shutil.move` preserves the original mtime, so an
mtime-based purge would delete a workflow the instant it was archived — and would
be unauditable besides.

**Operational controls:** `mr1 maintenance run [--dry-run]` and `mr1 maintenance
status`, every threshold overridable per-invocation, a persisted report per run
(`retention/reports/`), and a `retention_run` timeline event. The dry run computes
exactly the decisions the real run makes and changes nothing. Retention also runs
inside the supervisor on a slow cadence (6h default), deliberately *outside* the
planning gate — a paused supervisor should still reclaim disk — and it never calls
the brain.

### B2 — Recurring triggers  (1411 → 1431)

**New:** `mr1/autonomy/triggers.py` — `CronSpec`, `TriggerDecision`, `evaluate_recurrence`, `occurrences_due`, `validate_trigger`.
**Changed:** `mr1/autonomy/objectives.py` (`next_due_at`, `catch_up_remaining`, `last_fired_at`; `evaluate_trigger`; validation at creation), `mr1/autonomy/service.py` (persist trigger state at the moment of firing), `mr1/cli/objectives.py`, `mr1/cli/main.py`.
**Tests:** `tests/test_triggers.py` (18).

The case that matters is not "does an interval fire" — Phase A did that. It is
**time MR1 did not observe**. A daily objective and a week of downtime is seven
pending runs; a scheduler with no opinion fires seven workflows on return.

Semantics, stated precisely: `due` is the number of scheduled occurrences elapsed
since the trigger last fired. `due == 1` is normal operation and fires, with no
policy involved. `due > 1` means MR1 fell behind, and `missed = due - 1`
occurrences were lost:

| policy | make-up runs after any outage |
|---|---|
| `skip` | 0 — realign to the next boundary, run nothing |
| `catch_up_once` *(default)* | 1 — the backlog coalesces into a single run |
| `bounded` | up to `max_catch_up_runs`, worked off one per tick |

**The number of workflows an outage can produce is bounded by configuration,
never by the length of the outage.** A 168-hour outage of an hourly objective
yields at most `max_catch_up_runs` extra workflows.

**Restart safety** comes from `last_fired_at`, written *when the supervisor
decides to fire*, not when the work finishes. Phase A anchored on the last
completion, so a supervisor that fired at 09:00 and was killed at 09:01 had
recorded nothing and fired the same occurrence again on restart. Recording the
fire makes the recurrence at-most-once — which matters most for exactly the
objectives that hold standing consent.

**Cron** is a hand-rolled 5-field parser (`*`, `a`, `a,b`, `a-b`, `*/n`; Sunday is
0 or 7; restricted DOM and DOW match on *either*, as Vixie cron does). No
dependency, because a loop that must not acquire new failure modes should not
acquire new imports. **Timezone semantics are explicit:** `interval` is elapsed
wall-clock time and has no timezone; `cron` is a calendar and must have one
(IANA name, default UTC), so `0 9 * * 1` keeps meaning 09:00 local across a DST
shift instead of silently sliding an hour.

A bad cron is now rejected when a human types it, not on an unattended tick at
3am — and the fail-closed evaluation is *kept* for objective files edited by hand.

### B3 — Operator status and health  (1431 → 1446)

**New:** `mr1/autonomy/status.py` — `collect_status`, `RuntimeStatus`, `Finding`, `StatusThresholds`.
**Changed:** `mr1/cli/service.py` (`mr1 status` rebuilt on it; the ad-hoc `status_payload` deleted rather than left as cruft).
**Tests:** `tests/test_status.py` (15).

Every gauge already existed. What did not exist was an answer to the only
question an operator asks, which is never "what is
`oldest_pending_approval_age_s`" but **"is anything stuck, and do I have to do
something?"**

`mr1 status` now assembles one stable machine-readable payload (`schema_version`,
14 fixed sections) from every store, and rolls it up to **ok / warning / error**
with plain-language findings that each name a problem *and the command that fixes
it*. Exit codes are `0 / 1 / 2`, so cron and shell `&&` can branch without
parsing anything — verified live across paused (1), halted (2), and healthy (0).

Surfaced: service mode and ownership, heartbeat age, scheduler health and errors,
objectives by status with next-due times, blocked tasks, oldest pending approval
and its expiry, active and expiring grants, plans/actions against budgets, recent
autonomous failures and escalations, event and archive sizes, disk free, retention
status, triage status.

The load-bearing check is **stale heartbeat**: a supervisor holding its lock but
not ticking looks identical to a healthy one from outside. Nothing else in the
runtime can tell them apart. (A *stopped* MR1 is deliberately not an error — only
a running one that has gone quiet is wedged. Shouting about a clean shutdown is
how alerts get ignored.)

`collect_status` never raises: every section degrades to a recorded error, because
a status command that dies on a corrupt runtime fails exactly when it is needed.

### B4 — Backpressure and adaptive degradation  (1446 → 1462)

**New:** `mr1/autonomy/backpressure.py` — `evaluate_backpressure` (pure), `BackpressureLimits`, `BackpressureReporter`.
**Changed:** `mr1/autonomy/service.py` (GATE rebuilt on it, config, gauges), `mr1/cli/main.py`, `mr1/cli/service.py`.
**Tests:** `tests/test_backpressure.py` (16).

Phase A capped what MR1 could *spend*. B4 covers what MR1 does when the machine
underneath it is in trouble. One rule governs all of it:

> **Backpressure stops MR1 creating work, and never stops it finishing work.**

A runtime that sheds in-flight tasks when the disk gets tight has converted a
resource problem into a correctness problem. Under every signal, planning stops
and the scheduler keeps draining, retention keeps reclaiming, and results keep
being written.

Signals (all deterministic, all configurable, no adaptive rate control):
`health_degraded`, `disk_pressure`, `supervisor_degraded`, `scheduler_degraded`,
`concurrency_cap`. All are evaluated together and *all* are reported — an operator
with two problems should be told about two problems.

**Disk pressure refuses before the wall, not at it.** The threshold is not where
MR1 dies; it is where MR1 stops digging, with room left to drain, persist, and
archive. An unreadable volume (`-1`) is explicitly *not* treated as a full one:
reading "unknown" as "no space" would turn any transient `statvfs` error into an
outage.

Observability is the subtle part. A 60-second tick under six hours of pressure
would emit 360 identical events — indistinguishable from no signal at all. The
reporter emits on **edges**: once when a signal starts applying, once when it
lifts. Pinned by a test: 21 ticks under pressure produce exactly 2 events.

Budgets already survived restart (file-backed, flock'd, clock-injected); B4 adds
the test that proves it, because a crash loop that hands back a fresh hour of
planning budget every time is a crash loop that pays for itself.

### B6 — Notification interface  (1462 → 1477)

**New:** `mr1/autonomy/notify.py` — `Notification`, `NotificationSink`, `LocalOnlySink`, `FileSink`, `StdoutSink`, `CallableSink`, `Notifier`, `build_sinks`.
**Changed:** `mr1/autonomy/escalation.py` (optional notifier), `mr1/autonomy/service.py`, `mr1/cli/service.py`, `mr1/cli/main.py` (`--notify`).
**Tests:** `tests/test_notify.py` (15).

This is the **seam**, not the transport. Two sinks that depend on nothing: a
local no-op, and an append-only JSONL feed any external adapter can tail. No
Gmail, no Slack, no MCP — the point of a seam is that the runtime does not know
what is on the other side of it.

The invariant everything else serves: **a notification is never the delivery of
record.** The inbox message, the timeline event, and the objective's parked state
are written first and are not conditional on any sink. Pinned by a test in which
*every* sink fails and the escalation is still in the inbox, still on the
timeline, still parked — and the delivery failure is itself recorded. A
notification layer that can swallow an escalation is not a feature; it is a way
to be lied to.

Retries are bounded (`max_attempts`, default 3 — no growing queue, no ladder).
Delivery is idempotent per (escalation, sink) via a persisted ledger, so a
restart mid-escalation or a re-raised condition does not alert twice. One broken
sink does not stop the others. Failures emit `notification_failed` by default —
observability is not contingent on a caller having remembered to pass a log.

### B5 — Real-time soak readiness  (1477 → 1491)

**New:** `tests/soak/realtime.py` (the harness + analyser + CLI), `tests/soak/test_realtime_harness.py` (14), `pytest.ini`.

The 10 000-tick virtual soak proves the *logic* terminates. It cannot prove MR1
survives eight hours of wall-clock operation, because it never spends any: no
real subprocesses, no real file descriptors, no real memory pressure, no real
disk. Those are what kill long-running processes, and none of them can be
simulated away.

The harness runs a **real** supervisor — real `SystemClock`, real scheduler, real
`KaziAsyncRunner` spawning real subprocesses, real `ServiceLock` and
`ExecutionOwnership`, real consent grants, real capability audit, real disk — for
however long you ask, sampling throughout. The only thing it will stub is the
brain, and only if told to (`--planner fake`). **Nothing fakes the passage of
time.**

The test objective is Genesis-shaped with bounded scope: `git status --short`
against the repo, as a risk-1.0 `shell_command` that must pass a real consent
grant whose `arg_predicate` authorizes `^git status` in one directory and nothing
else. Read-only and idempotent, so a crashed-and-retried soak cannot damage
anything.

Captured per sample: RSS, CPU, threads, open FDs, runtime bytes, live/archive
event bytes, archive segments, total events, workflow counts, objective status and
runs, brain calls, grant use count, supervisor/scheduler tick counts and errors,
tick latency, heartbeat age, health, active backpressure. Samples are appended to
`samples.jsonl` and fsynced, so an interrupted soak is still analysable and
`--resume` picks it back up. SIGINT is a clean stop.

Nine invariants are checked, and **every one is proven capable of failing** —
`test_realtime_harness.py` feeds the analyser samples describing broken runtimes
(leaking memory, leaking FDs, wedged heartbeat, planning on idle ticks, runaway
replanning, duplicate launches, tick errors, creeping latency, capability executed
outside consent) and asserts it says so. An empty soak reports **failure**, never
success. A soak tool that cannot fail manufactures confidence.

---

## 2. Architecture actually implemented

```
mr1 serve  (headless, singleton via ServiceLock; execution authority via flock)
│
├── Supervisor.tick()            [slow: 60s default]
│     observe → gate → sweep → RETENTION → reconcile → plan → recover → escalate
│       • GATE now evaluates backpressure: health, disk, degraded loops, concurrency
│       • RETENTION (B1) sits outside the planning gate — a paused MR1 still reclaims disk
│       • recurrence (B2) is persisted at the moment of firing, not on completion
│       • escalation (B6) fans out to sinks *after* the inbox and timeline are durable
│
├── Scheduler._run_loop()        [fast: 1s]   — executes everything
│       • tick() is a no-op without execution authority (B8); counts a delegated tick
│
└── GovernedTriage               [gated + budgeted, unchanged]

runtime_root/
  control.json   health.json   autonomy_budget.json   supervisor.pid
  execution.lock  execution_owner.json                          ← B8
  objectives/    consent_grants/
  events/  events.jsonl  segments.json  archive/events-NNNNNN.jsonl.gz   ← B1
  archive/  workflows/  capability_audits/  snapshots/  .archive_index.json  ← B1
  retention/reports/                                            ← B1
  notifications/  delivered.json  notifications.jsonl           ← B6
  workflows/  agents/  messages/  capability_approvals/
```

---

## 3. Differences from the Opus plan, and why

| # | Plan said | Built | Why |
|---|---|---|---|
| 1 | B1: "prune terminal workflow dirs, prune tasks state, prune snapshots + audit dirs" | **Archive**, never prune. Deletion is opt-in (`--purge-archives-after-days`) and confined to the archive. | The stated constraint: *never silently delete data merely because it is old*. Archival is recoverable; pruning is not. |
| 2 | B1: "fix the 50k silent read-truncation" | `list_events()` became a *complete-history* query (cache when provably complete, disk otherwise); `recent_events()` is the explicit window. | A flag would have left the dangerous default in place. The correct default is the one that cannot silently lie. |
| 3 | B1 (unspecified) | The archive records `archived_at` in a ledger rather than trusting file mtime. | `shutil.move` preserves the original mtime, so an mtime-based purge would delete a workflow the instant it was archived. Found by a test. |
| 4 | B2: "register an interval/cron watcher into `WatcherRegistry`" | Recurrence is a pure module (`mr1/autonomy/triggers.py`) evaluated by the supervisor. Watcher-backed triggers still work, unchanged. | `WatcherRegistry` watchers are polled *as tasks inside a workflow* — they need a Task and a workflow to be evaluated. That is the wrong shape for "does this objective have work yet". Phase A had already established the objective-trigger seam; B2 extends it rather than bending the registry. |
| 5 | B2 (unspecified) | Persisted `next_due_at` / `last_fired_at` / `catch_up_remaining`, and a missed-run policy. | The requirement — "an objective must never produce unlimited catch-up workflows after downtime" — cannot be met by deriving due-times from the last completion, which cannot distinguish "never ran" from "ran, then crashed". |
| 6 | B4: "adaptive slowdown on degraded health" | Deterministic stop-planning signals. No rate adaptation. | The stated constraint: *do not invent dynamic AI-based rate control; keep policy deterministic and configurable*. "Why did it not run" must never be a research question. |
| 7 | B8: "`_tick_lock` is in-process only" — implies a global fix | Ownership is enforced at **process entry points** (`serve`, REPL, one-shot CLI tick), not by default inside `Scheduler`. | Making it default-on would change the behaviour of every embedded `Scheduler` and every existing test. Ownership is a property of *processes* sharing a root, which is exactly where it is now enforced — and the multi-process test proves it with two real interpreters. |
| 8 | B7: "just use `WorkflowStore.locked()`" | It already did. Verified, pinned with an ordering test, and a latent self-deadlock removed by sharing the store instance. | The plan's premise was already fixed in Phase A. Rewriting a correct path to match a stale diagnosis would have been the actual regression. |
| 9 | B5 (unspecified) | The soak harness runs the supervisor **in-process** rather than spawning `mr1 serve`. | The RSS and FD numbers then measure the thing under test, and the planner can be swapped without a CLI hook. It still takes the real `ServiceLock` and `ExecutionOwnership`, so it is the same deployment shape. |

---

## 4. Results

### Deterministic suite
```
python -m pytest tests/ --ignore=tests/runtime_qa
1492 passed in 47.7s          (Phase A baseline 1373 → +119, 0 regressions)
```

New tests by checkpoint: B8 12 · B7 3 · B1 24 · B2 18 · B3 15 · B4 16 · B6 15 · B5 14 (+1 from splitting an objectives test).

### Runtime QA
```
python -m tests.runtime_qa.runner --jobs=1  → 50 scenarios, 0 findings, 0 crashed  (597.4s)
python -m tests.runtime_qa.runner --jobs=4  → 50 scenarios, 0 findings, 0 crashed  (197.6s)
```

### Soak
```
python -m pytest tests/soak/   → 49 passed in 29.7s
```
Includes the Phase-A 10 000-tick virtual soak, fault injection, 50 mid-flight
crash/restarts, the control-plane soak, and the 14 new real-time-harness tests
(one of which is a genuine 8-second real soak).

### Real-time smoke soak — **actually run**
```
python -m tests.soak.realtime --duration 60s --planner fake \
    --tick-interval 3s --objective-interval 10s --sample-interval 5s

REAL-TIME SOAK — PASSED          60s of 60s requested, 13 samples
  objective runs   5 (active)     brain calls      5
  workflows        5              grant uses       5
  capability execs 5              supervisor ticks 20
  RSS              35.3 → 35.9 MiB  (1.02x)
  file descriptors 35 → 35
  tick latency     p50 2.2ms  p95 13.1ms   early median 4.4 → late median 1.7
  all 9 invariants held
```

A second real run with a forced rotation threshold proved retention fires under
live load: **4 archive segments written mid-soak**, live log shrank to 35 KiB,
and all 122 events still reachable — history stayed complete across rotation
while the runtime was running.

### 24-hour real-time soak — **actually run** (2026-07-12 → 07-13)

```
python -m tests.soak.realtime --duration 24h --planner fake \
    --objective-interval 30m --tick-interval 10s --sample-interval 60s

REAL-TIME SOAK — PASSED       86 406s of 86 400s requested, 1440 samples
  objective runs   48 (active)   brain calls      48
  workflows        48            grant uses       48
  capability execs 48            task starts      48
  supervisor ticks 8 632         scheduler ticks  85 879
  RSS              43.3 → 43.4 MiB  (1.00x)      file descriptors 39 → 39
  threads          2             cpu              114.9s of 86 406s wall (0.13%)
  runtime disk     13 KiB → 2.6 MiB              events 912 (662 KiB)
  tick latency     p50 1.5ms  p95 31.4ms  max 82.9ms
                   early median 1.42ms → late median 1.55ms
  all 9 invariants held
```

The numbers that matter:

- **48 runs, 48 plans, 48 executions, 48 grant uses, 48 task starts, and every
  task with `attempt_count == 1`.** Nothing ran twice, over a full day.
- **48 brain calls across 8 632 supervisor ticks.** The zero-cost-idle-tick
  invariant — the property the whole autonomy design is built on — held for 24
  hours of real operation. 8 584 ticks cost nothing.
- **RSS grew 128 KiB in a day (1.003×). File descriptors: 39 → 39.** No leak.
- **Tick latency did not creep** (1.42ms → 1.55ms median). The tick is O(1) in
  history, not O(n).
- Retention ran 144 times and archived 3 terminal workflows, correctly keeping 45
  back: 20 under the `keep_recent` floor, 25 still referenced by the objective's
  bounded attempt history. Exactly as designed.

**The first analysis reported FAILED — and it was the harness that was wrong.**
It counted workflows by listing directories in `workflows/`, and B1 retention had
archived 3 of them *while the soak was running*: 45 live + 3 archived = 48. The
analyser compared 48 grant uses against 45 workflows and called it a duplicate
launch. Two Phase B features colliding, caught in the one place built to catch
collisions. The analyser now counts from the event log — complete history, immune
to retention moving directories underneath it — and the same 24h data re-analyses
to PASSED. A regression test
(`test_archived_workflows_do_not_read_as_a_duplicate_launch`) pins it.

`health` ends as `warning` because the soak's scratch runtime has never run memory
maintenance (no graph, no insights, no child agents). Those are correct doctor
warnings about a fresh root, not soak findings.

**The 8-hour soak and the real-planner (`--planner real`) soak have not been run.**
The 24h run used `--planner fake`: everything else was real — real clock, real
scheduler, real subprocesses, real consent gate, real disk — but the workflow spec
came from a deterministic planner rather than `claude`.

---

## 5. Definition of done — verified

| # | Requirement | Status |
|---|---|---|
| 1 | One process exclusively owns workflow execution per runtime root | ✅ flock ownership; two-real-process test; live singleton + release verified |
| 2 | Historical event queries never silently truncate | ✅ `list_events()` is complete-history; `recent_events()` is the named window; the test that asserted the defect now asserts both properties |
| 3 | Disk/state growth has a configurable, auditable retention path | ✅ rotation + archival, dry-run, per-run persisted report, timeline event, nothing deleted unless explicitly asked |
| 4 | Recurring objectives fire correctly across restart and downtime | ✅ `last_fired_at` makes it at-most-once; missed-run policy bounds the backlog by config, not by outage length |
| 5 | Operators can understand health and blocked work without reading files | ✅ `mr1 status` — rollup, findings with fixes, stable JSON schema, exit codes 0/1/2 |
| 6 | Backpressure safely limits autonomous creation and execution | ✅ stops creating, keeps draining; disk/degraded/concurrency/health; edge-triggered events |
| 7 | Escalations have a reliable transport-neutral notification seam | ✅ sink interface, local + file + stdout, bounded retries, dedupe, and a broken transport cannot lose an escalation |
| 8 | Honest tooling for real 8h/24h validation | ✅ real harness, 9 falsifiable invariants, resume/interrupt, fake-planner mode — and **a real 24-hour soak that ran and passed** |
| 9 | Deterministic, runtime-QA, and virtual-soak tests remain green | ✅ 1492 / 50-0-0 (×2) / 49 |

---

## 6. Known limitations

1. **A 24-hour soak has been run and passed** (`--planner fake`). The **8-hour**
   soak and any **`--planner real`** soak have not: the real planner spawns
   `claude` on every objective firing, so a 24h run at a 30m cadence would be 48
   real LLM calls. Everything except the brain was real in the run that happened.
2. **The 24h soak never exercised event rotation.** It produced 912 events
   (662 KiB), under the 1 MiB default threshold, so `events.jsonl` never sealed a
   segment. Rotation is covered by the deterministic tests and by a dedicated
   real soak with a forced threshold — but not yet by a 24h run. A busier
   objective, or a lower `--events-max-live-bytes`, would close that.
3. **Cron is a 5-field subset.** `*`, ranges, lists, and steps — no `@yearly`, no
   `L`/`W`/`#`. Enough for every calendar recurrence an objective has needed.
4. **Archived workflows are not restorable through a CLI command.** They are
   plain directories under `archive/workflows/` and can be moved back by hand; a
   `mr1 maintenance restore <id>` would be a small addition.
5. **Event dedupe on append covers the cached window.** A duplicate `event_id`
   whose original has been archived would not be caught — practically impossible,
   since the id hashes the timestamp, but not structurally impossible.
6. **The degraded-mode failure counters are in-process.** A supervisor that
   crashes and is restarted by a process manager gets a fresh counter each time,
   so a *process-level* crash loop is not caught by B4 (the budget ledger, which is
   file-backed, is what bounds its spending). Detecting that belongs to whatever
   supervises the supervisor.
7. **A failed notification is not retried on a later tick.** Three attempts, then
   the alert is lost — recorded as `notification_failed`. The escalation itself is
   never lost.
8. **Ownership binds processes, not library callers.** A `Scheduler` constructed
   directly without an `ExecutionOwnership` is still the unconditional executor.
   That is deliberate (it is what keeps embedding and testing sane), but it means
   B8 is a guarantee about *processes*, not about arbitrary code.
9. Phase A's limitations that Phase B did not touch remain: at-most-once is
   enforced by classification rather than a commit token (Phase C), a hung workflow
   holds its concurrency slot until the stuck sweep escalates, and `CompilerPlanner`
   inherits the compiler agent's behaviour.

---

## 7. Exact commands

**Ownership / status inspection**
```bash
python -m mr1.workflow_cli status              # rollup + findings; exit 0=ok 1=warning 2=error
python -m mr1.workflow_cli status --json       # stable schema (schema_version, 14 sections)
python -m mr1.workflow_cli doctor
python -m mr1.workflow_cli timeline recent
```
`status` reports `execution: owned by pid N (service|repl|cli)` or `unowned`. A
REPL started while a service holds the root prints
`Execution: delegated to pid N (service)` at startup and will not launch tasks.

**Retention — dry run, then execute**
```bash
python -m mr1.workflow_cli maintenance status
python -m mr1.workflow_cli maintenance run --dry-run      # changes nothing; predicts exactly
python -m mr1.workflow_cli maintenance run

# thresholds (all optional; these are the defaults)
python -m mr1.workflow_cli maintenance run \
    --events-max-live-bytes 33554432 --events-keep-recent 1000 \
    --workflow-archive-after-days 30 --workflow-keep-recent 50 \
    --audit-archive-after-days 90 --snapshot-archive-after-days 90

# the only destructive option, and it never touches live state
python -m mr1.workflow_cli maintenance run --purge-archives-after-days 365
```

**Recurring objectives**
```bash
# interval, with a bounded backlog after downtime
python -m mr1.workflow_cli objective create "run the weekly genesis cycle" \
    --title Genesis --kind recurring --every 7d --idempotent \
    --missed-run-policy catch_up_once

# calendar, in a named timezone
python -m mr1.workflow_cli objective create "run the weekly genesis cycle" \
    --title Genesis --kind recurring \
    --cron "0 9 * * 1" --timezone America/New_York \
    --missed-run-policy bounded --max-catch-up-runs 2 --idempotent

python -m mr1.workflow_cli objective show <obj-id>     # shows next due / last fired / catch-up owed
```

**Backpressure**
```bash
python -m mr1.workflow_cli serve \
    --max-concurrent-workflows 4 \
    --max-plans-per-hour 20 \
    --max-workflows-per-objective-per-day 24 \
    --max-actions-per-hour 60 \
    --min-disk-free-bytes 536870912 \
    --max-consecutive-supervisor-errors 3 \
    --max-consecutive-scheduler-errors 5 \
    --retention-interval-s 21600
```

**Notification output**
```bash
python -m mr1.workflow_cli serve --notify file            # runtime_root/notifications/notifications.jsonl
python -m mr1.workflow_cli serve --notify file:/tmp/mr1-alerts.jsonl
python -m mr1.workflow_cli serve --notify stdout --notify file
# an external adapter is then just:  tail -f /tmp/mr1-alerts.jsonl | your-forwarder
```

**Real-time soak**
```bash
# validate the harness in under a minute — real everything, no LLM
python -m tests.soak.realtime --duration 60s --planner fake \
    --tick-interval 3s --objective-interval 10s --sample-interval 5s

# the 8-hour soak
python -m tests.soak.realtime --duration 8h --planner real \
    --objective-interval 30m --sample-interval 60s --dir soak-runs/8h

# the 24-hour soak, with retention active
python -m tests.soak.realtime --duration 24h --planner real \
    --objective-interval 30m --retention-interval 1h --dir soak-runs/24h

# Ctrl-C is a clean stop. Then:
python -m tests.soak.realtime --resume soak-runs/24h     # continue
python -m tests.soak.realtime --report soak-runs/24h     # re-analyse without running
```

**Suites**
```bash
python -m pytest tests/ --ignore=tests/runtime_qa    # 1492
python -m tests.runtime_qa.runner --jobs=1           # 50 / 0 findings
python -m tests.runtime_qa.runner --jobs=4           # 50 / 0 findings
python -m pytest tests/soak/                         # 49
python -m pytest tests/ --ignore=tests/runtime_qa -m "not slow"   # skip the real 8s soak
```
