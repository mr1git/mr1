
MR1 Autonomous Operation — Architecture & Roadmap
Context
Five QA sessions hardened MR1's orchestration runtime to the point where Session 5 concluded it "is no longer the primary engineering bottleneck" — 1133 tests green, 50 QA scenarios / 0 findings, routing probe 0/21. The next phase is autonomous operation: MR1 running continuously, holding long-lived objectives, recovering from failure, and asking for a human only when it genuinely needs one.

This plan does not redesign MR1 or rewrite the runtime. The scheduler's reconciliation loop is correct and stays untouched. What's missing is a thin layer above it, one governance primitive below it, and four blocking assumptions that must be retired.

The finding that changes the plan
Sessions 4 and 5 both report the "consented-autonomy tier" as built and verified. It is not built. Traced in the committed tree:

_approved_override_matches returns true only for approval_scope == "single_use" (capability_policy.py:1275). A grant_scope approval never bypasses the risk gate — it only appends a path to the agent's scope_roots (capability_policy.py:1074-1082).
An approval's ID is a SHA-256 over (actor, capability, invocation_mode, normalized_args, scope_roots, **workflow_id**, **task_id**) (capability_policy.py:887-912). It is bound to one exact invocation in one exact workflow.
shell_command is the only risk-1.0, workflow-allowed capability (capability_policy.py:458); MAX_AUTONOMOUS_CLEARANCE = 0.99 (scoped_agents.py:30).
Worse: apply_decision raises ValueError on grant_scope without a requested_scope_path (capability_policy.py:1074-1076) — and a risk block never populates that field (only a scope block does, via _check_scope). So approvals approve <id> --grant-scope on a risk-blocked shell task errors out today.
Consequence: a recurring objective that shells out mints a new workflow_id on every run → a new approval hash → a fresh human approval, forever. There is no standing consent. This is not a missing TTL on an existing mechanism; the mechanism does not exist. It is the single hardest gate on unattended operation.

Decisions taken (Marwan, this session)
Decision	Choice
Consent model	Objective-scoped — consent attaches to a mission, expires with it, revocable
Escalation posture	Bounded-autonomous — retry → replan → fallback within a budget; escalate on exhaustion
First real mission	Genesis weekly cycle (already spec'd; low-risk; naturally recurring)
1. Autonomous Runtime Architecture
Two loops at different frequencies. The fast one already exists and does not change.

┌─ Supervisor process  (mr1 serve — NEW, headless, singleton) ──────────┐
│                                                                        │
│  Supervisor.tick()   [slow: 60s default]            ← NEW             │
│    observe → gate → sweep → reconcile → plan → recover → escalate     │
│      • decides WHAT work exists                                        │
│      • calls the brain ONLY on a state change (never on a plain tick)  │
│      • executes nothing itself                                         │
│                                                                        │
│  Scheduler._run_loop() [fast: 1s]                   ← EXISTS, UNCHANGED│
│    tick() → reconcile active workflows → launch/poll/finalize tasks    │
│      • executes everything                                             │
│                                                                        │
│  InboxTriage._run_inbox_loop() [30s]                ← EXISTS, to govern│
└────────────────────────────────────────────────────────────────────────┘
         │ reads/writes
         ▼
   runtime_root/  objectives/  consent_grants/  control.json  health.json
                  workflows/   capability_approvals/  events/  agents/
The supervisor creates and classifies; the scheduler executes. Keeping that split is what makes this an additive layer rather than a rewrite.

2. Required Components
#	Component	New / Extend	Reuses
1	mr1/clock.py — Clock protocol, SystemClock, VirtualClock	New (pure refactor)	—
2	mr1/autonomy/supervisor.py — headless host + reconcile loop	New	MR1, Scheduler, run_doctor
3	mr1/autonomy/objectives.py — Objective, ObjectiveStore	New	WorkflowStore pattern (flock + atomic write + active index)
4	mr1/autonomy/consent.py — ConsentGrant, ConsentGrantStore	New	CapabilityAuditWriter, EventLog
5	PolicyEngine.evaluate — second override path	Extend (capability_policy.py:1228)	existing risk gate
6	CapabilityApprovalRequest.expires_at + expiry sweep	Extend	expire_requests_for_workflow (capability_policy.py:1087)
7	mr1/autonomy/recovery.py — failure classifier + ladder	New (pure function)	TaskStatus, WorkflowStatus
8	mr1/autonomy/control.py — control.json + pause/resume/stop/halt	New	atomic-replace pattern from state.py
9	mr1/autonomy/health.py — gauges + heartbeat + health.json	New	run_doctor() → DoctorReport (doctor.py:1433)
10	IntervalWatcher / CronWatcher	Extend	WatcherRegistry.register (watchers.py:314)
11	tests/soak/ — virtual clock + fake brain + fault injection	New	MockRunner (kazi_runner.py:503), auto_tick=False
12	Retention/GC service	New (Phase B)	—
Do not build: a health-check system (DoctorReport is one), a trigger engine (WatcherRegistry is one), an execution loop (scheduler.tick() is one), a blocked-work query (EventLog.blocked_now() is one), a bounded step loop (MRnRunPolicy is one), a cross-process store lock (WorkflowStore.locked() is one).

3. Execution Lifecycle
Supervisor.tick() — one pass, default 60s. Ordered so that safety gates run before any work is created.

1. OBSERVE
     control  = read control.json                       (mode: running|paused|stopping|halted)
     health   = run_doctor(runtime_root)                [REUSE — DoctorReport]
     gauges   = active wfs, oldest pending approval age, consecutive failures,
                disk free, events.jsonl bytes, plans this hour
     write heartbeat + health.json; emit supervisor_tick event

2. GATE  (backpressure — every one of these stops PLANNING, not draining)
     halted            → cancel running tasks, REVOKE ALL GRANTS, exit
     stopping          → no new plans; drain in-flight; exit when idle
     paused            → no new plans; keep draining
     health == error   → no new plans; escalate once; keep draining
     at concurrency cap / plan-rate budget exhausted → skip planning this tick

3. SWEEP  (time-based lifecycle — nothing does this today)
     approvals past expires_at        → expired  + emit approval_expired
     consent grants past expires_at   → expired  + emit grant_expired
     tasks BLOCKED > threshold        → stuck
     workflows RUNNING > 24h          → stuck    (precedent: _STUCK_RUNNING_THRESHOLD, doctor.py:85)
     each → feed RECOVER for the owning objective

4. RECONCILE  (per active objective)
     if obj.current_workflow_id:
         wf = load(...)
         wf not terminal   → continue (scheduler is working it)
         wf SUCCEEDED      → record success; satisfied (once) | schedule next (recurring) | idle (standing)
         wf FAILED         → RECOVER(obj, wf)
     else:
         trigger fires?    → PLAN(obj)          [WatcherRegistry.evaluate]

5. PLAN   ← the ONLY place the brain is called, and only on a state change
     brain(objective statement + history + grounding) → workflow spec
     validate spec
     check every capability the spec needs against obj's consent grants
       → needs authority it doesn't hold?  ESCALATE (ask for consent). DO NOT SUBMIT.
     submit with workflow_metadata = {objective_id: obj.id}
     scheduler picks it up on its next 1s tick

6. RECOVER   → §4

7. ESCALATE  → message to Marwan's inbox + timeline event + objective state change
A steady-state tick with nothing to do costs zero tokens. That property is what makes continuous operation affordable, and it must be preserved — never call the brain from OBSERVE, GATE, or SWEEP.

Objective model
@dataclass
class Objective:
    objective_id: str
    title: str
    statement: str                       # the NL goal
    kind: Literal["once", "recurring", "standing"]
    trigger: dict                        # a WatcherRegistry spec — reuse, don't invent
    status: Literal["active", "planning", "executing", "recovering",
                    "waiting_human", "satisfied", "quarantined", "abandoned", "paused"]
    owner_agent_id: str
    consent_grant_ids: list[str]         # the authority this mission holds
    failure_policy: FailurePolicy        # budgets, below
    current_workflow_id: str | None
    consecutive_failures: int
    history: list[Attempt]               # {workflow_id, outcome, classification, at}
once — a mission with a completion criterion; runs until satisfied.
recurring — fires on a trigger (interval/cron watcher, Phase B). Genesis is this.
standing — a continuously-reconciled responsibility.
4. Failure-Recovery Lifecycle
Bounded-autonomous (Marwan's choice). One rule governs everything:

A failure escalates exactly one level per exhausted budget. Every level terminates. Nothing retries forever; nothing fails silently.

Classification (recovery.py — a pure function over a failed workflow)
Class	Signals	Action
transient	timeout, infrastructure_failure, tool unavailable, subprocess OSError	backoff + resubmit same workflow
planning	task failed on its merits (bad command, wrong assumption, assertion)	replan — brain drafts a new workflow given the failure
blocked	approval_required, consent missing/expired, scope denied	escalate — MR1 cannot self-authorize. Ever.
fatal	budget exhausted, identical failure repeated, spec invalid after N replans	quarantine + escalate + stop
Propagation
capability fails
  → attempt retries                    [EXISTS — per-task retry semantics]
    → task terminal FAILED/TIMED_OUT/BLOCKED    [EXISTS]
      → workflow terminal FAILED                [EXISTS]
        → OBJECTIVE failure policy              [NEW]
             transient → backoff (30s→2m→8m, jittered), resubmit    ≤ 3
             planning  → replan                                      ≤ 2
             blocked   → ESCALATE, park WAITING_HUMAN
             fatal     → QUARANTINE, ESCALATE, stop trying
        consecutive_failures ≥ 5 → QUARANTINE regardless of class
The cases you named
task fails → attempt retries, then the ladder.
workflow fails → the ladder, from the objective.
agent fails → surfaces as a failed workflow task; normal ladder.
tool unavailable → transient; backoff.
approval expires → the task stays BLOCKED (fail-closed — correct). The SWEEP notices, classifies blocked, escalates, parks the objective. Today this parks forever with nobody watching.
Crash recovery is currently destructive — and must be handled here
Scheduler._handles is in-memory. On restart, a task persisted RUNNING with no recoverable result.json is force-failed as error_type="infrastructure_failure" (scheduler.py:1255-1277). A supervisor that restarts MR1 after a crash therefore converts in-flight work into permanent failures.

Phase A mitigation (cheap and correct): classify infrastructure_failure as transient, so the ladder retries it rather than treating it as a real defect. This is sufficient provided the objectives MR1 runs unattended are idempotent-ish — an explicit constraint on what missions are safe to enable, and it must be documented on Objective.

Phase C fix: a persisted idempotency/commit token, so restart can distinguish "never ran" from "ran, result lost."

5. Operational Model
Control is a file, not a socket: crash-safe, inspectable, cross-process, no new transport.

<runtime_root>/control.json — written with the same fsync + atomic-replace pattern as state.py, read every supervisor tick.

{"mode": "running|paused|stopping|halted", "reason": "...", "requested_by": "...", "requested_at": "..."}
Command	Effect
mr1 serve	Start the supervisor headless. Singleton via fcntl.flock on a pidfile. Installs a SIGTERM handler (today only SIGINT exists, root.py:3115).
mr1 pause	mode=paused. Stops planning. In-flight workflows keep draining.
mr1 resume	mode=running.
mr1 stop	Graceful. No new work; drain in-flight; then MR1.shutdown() and exit.
mr1 halt	Emergency stop. Cancel running tasks now, revoke every consent grant, pause all objectives, exit.
mr1 status	Heartbeat age, mode, health rollup, active objectives, oldest pending approval, budget usage.
The semantic that matters: pause stops new work; halt kills current work and removes MR1's authority. A stop that leaves standing consent in place has not actually stopped anything — which is exactly why halt revokes grants.

Backpressure (all config, all enforced in GATE)
max_concurrent_workflows · max_plans_per_hour (token/cost ceiling on brain calls) · max_workflows_per_objective_per_day · health error ⇒ planning halts automatically.

The already-autonomous surface nobody governs
The inbox-triage loop already runs unattended today — every 30s it makes an LLM call that may take up to 5 actions including agent_run, assign_mission, and create_workflow (root.py:3045, inbox_triage.py:34-44), with no budget and no control-plane hook. It must come under the same control plane: pause must pause it, and its actions must count against the same budget. Otherwise "paused" is not paused. (A9)

6. Monitoring Model
Reuse DoctorReport as the health rollup — it is already JSON, already categorized (runtime, events, memory, workflows, agents, capabilities, approvals, messages), already rolls up to ok|warning|error, and already has a --json CLI. Add a supervisor-computed gauge set written to <runtime_root>/health.json each tick and emitted as a supervisor_tick event.

Signal	Why it matters
supervisor_heartbeat_at	The most important one. Stale > 3 intervals ⇒ MR1 is dead or wedged. Today nothing would tell you.
supervisor_tick_errors	Non-zero ⇒ the autonomy loop is failing.
scheduler_tick_errors	Currently swallowed to /dev/null — see A1.
scheduler_tick_duration_ms	> tick interval ⇒ falling behind.
objectives_{active,waiting_human,quarantined}	quarantined > 0 ⇒ MR1 gave up on something.
oldest_pending_approval_age_s	The unattended-stall detector.
consecutive_failures[objective]	Approaching budget ⇒ about to escalate.
plans_this_hour	Runaway-planning / token-burn detector.
grants_{active,expiring_soon}	Authority currently outstanding.
unattended_executions[grant_id]	What MR1 did without asking. The accountability metric.
events_jsonl_bytes, disk_free_bytes	The months-of-uptime cliff.
event_cache_truncated	See below.
A correctness bug the QA sessions scored as closed
_MAX_CACHE_EVENTS = 50_000 (event_log.py:104-109) bounds memory — but list_events() and filter_events() read from the cache. Past 50k events, history queries silently return a truncated view with no error. Session 5 scored this "N-5 closed." For a long-running system whose recovery and planning logic read history, that is a correctness landmine, not an ops gauge. Fixing it belongs with retention (B1).

7. Roadmap
Phase A — required before autonomy can be switched on at all
#	Item
A0	Clock seam (mr1/clock.py). Injectable now() into supervisor, watchers, approval TTL, grant TTL. Nothing time-based is testable without it.
A1	Fail-loud loops. Scheduler._run_loop records the error + emits an event + increments a counter. Today except Exception: pass (scheduler.py:948-951) makes a wedged scheduler indistinguishable from an idle one. Reuse record_runtime_error (state.py:261).
A2	mr1 serve — headless supervisor: singleton flock + pidfile, SIGTERM handler, heartbeat, control.json, pause/resume/stop/halt.
A3	Approval wall-clock TTL — expires_at on CapabilityApprovalRequest + expiry sweep + escalate on expiry.
A4	Consent grants — the keystone. §8 below.
A5	Objective store + reconcile loop — observe/gate/sweep/reconcile/plan.
A6	Failure ladder + budgets + quarantine; classify infrastructure_failure as transient.
A7	Escalation — every "I need a human" writes to Marwan's inbox (MessageStore.send) + a timeline event. Never silent.
A8	Soak harness (virtual clock + fake brain + fault injection) + 10k-tick soak.
A9	Bring inbox-triage under the control plane + budget. It is already unattended and ungoverned.
Phase B — comfortable daily usage
#	Item
B1	Retention/GC: rotate events.jsonl, prune terminal workflow dirs, prune tasks state, prune snapshots + audit dirs. Fix the 50k silent read-truncation.
B2	Recurring objectives: register an interval/cron watcher into WatcherRegistry.
B3	mr1 status + health.json + the full gauge set.
B4	Backpressure caps + adaptive slowdown on degraded health.
B5	8h + 24h wall-clock soaks with the Genesis objective.
B6	Notification transport so escalations reach Marwan away from the terminal (ties into the messaging-channel project).
B7	Fix the _resume_blocked_workflow_task WorkflowStore race — it builds a fresh store instance (capability_policy.py:1123); WorkflowStore.locked() now has a real fcntl.flock (workflow_store.py:153), so just use it.
B8	Cross-process tick safety. Scheduler._tick_lock is in-process only; mr1 serve + an open REPL would both _launch_ready() the same task.
Phase C — future
C1 Persisted idempotency/commit token → true at-most-once, non-destructive crash recovery. C2 Objective decomposition (sub-objectives, inter-objective dependencies). C3 Learned failure classification; self-tuning budgets. C4 Week-long soak; multi-process supervisor. C5 Residual hygiene: title tombstones, cross-process event reader flock, direct-loader graceful degrade.

8. Consent Grants (A4) — the keystone, in detail
A second, parallel override path. Do not modify _approved_override_matches — single-use-keyed-to-one-invocation is the correct semantic for a one-off human approval. The grant is a different mechanism with different semantics: coarser, predicate-matched, TTL'd, revocable.

@dataclass(frozen=True)
class ConsentGrant:
    grant_id: str
    grantee_kind: Literal["objective"]     # objective-scoped, per decision
    grantee_id: str                        # obj-xxx
    capability_name: str                   # "shell_command"
    scope_roots: list[str]                 # path bound
    arg_predicate: dict                    # {"cmd": {"regex": "^(pytest|ruff|git status)"}}
    max_risk: float                        # 1.00
    granted_by: str                        # root only — existing rule: grant_scope requires clearance ≥ 1.0,
                                           # and max clearance is 0.99, so only root qualifies
    granted_at: str
    expires_at: str                        # REQUIRED. No immortal grants.
    revoked_at: str | None
    use_count: int
Wiring — one surgical change at capability_policy.py:1228:

# today
if metadata.risk_score > max_risk and not approved_override:
    → requires_approval

# new
if metadata.risk_score > max_risk and not approved_override and not consent_match:
    → requires_approval
consent_match(request, metadata, grants) is a predicate evaluation: grantee matches the request's owning objective, capability matches, request.scope ⊆ grant.scope_roots, args satisfy arg_predicate, metadata.risk_score ≤ grant.max_risk, not expired, not revoked.

Threading the objective through: submit_workflow already accepts workflow_metadata (scheduler.py:958). Stamp {objective_id: obj.id} there; the capability gate reads it back off the workflow. No new plumbing through CapabilityRequest.

Every grant-authorized execution must: emit capability_allowed with reason="consent_grant" + grant_id; write an audit record referencing the grant (the audit writer already exists — thread the id); increment use_count.

Operator surface:

mr1 grant create --objective obj-x --capability shell_command \
                 --scope ~/Projects/mr1 --allow '^pytest' --ttl 7d
mr1 grant list | mr1 grant show <id> | mr1 grant revoke <id> | mr1 grant revoke --all
Acceptance gate for A4: prove shell_command executes unattended under a matching grant, and is denied without one, with both outcomes visible in the audit log.

9. Soak Tests (A8)
Nothing exists today: no conftest.py, no fake clock, no fake brain, zero soak/bench tests. Build the seams first (A0 + the harness), then:

Soak	Clock	Asserts
10k supervisor ticks	virtual	No unbounded memory/disk growth; no stuck objective; heartbeat never stale; every escalation delivered; tick duration flat (no O(n) creep)
Fault injection (1k ticks, 10% failure)	virtual	Every failure reaches a terminal state; budgets respected; no infinite retry; quarantine fires
Crash/restart (kill mid-workflow ×50)	virtual	In-flight tasks recover or fail transiently — never permanently; no lost objectives
8h wall-clock	real	Genesis on a 30m cadence, real Claude. Watch memory / disk / token drift.
24h wall-clock	real	Same, with retention active. Assert disk stable.
Week-long	real	Phase C. Only meaningful once retention lands.
The 10k-tick soak is the one that must exist before autonomy ships. The wall-clock soaks are Phase B.

10. Implementation Order (Sonnet-executable)
Each step lands independently, is testable on its own, and does not require the next one to be useful.

#	Step	Notes
1	A0 clock seam	Pure refactor. All 1133 tests stay green. Zero behavior change.
2	A1 fail-loud scheduler loop	~20 lines. Immediately valuable. No design risk.
3	A2 mr1 serve + control plane	Shippable value with zero autonomy — MR1 stays alive headless and drains its queue. Also fixes today's bug that CLI-submitted workflows never advance (mrn_loop/inbox_triage build Scheduler(auto_tick=False) and never call tick()).
4	A3 approval TTL + sweep	Still no consent — just stop the infinite park, and escalate on expiry.
5	A4 consent grants	The governance keystone. Land with the grant CLI + the acceptance gate in §8.
6	A6 failure ladder	Before A5 — it's a pure function over a failed workflow, unit-testable standalone.
7	A5 objectives + reconcile loop	Everything it needs now exists.
8	A7 escalation to inbox	Wire into the ladder.
9	A9 inbox triage under control	Close the ungoverned unattended surface.
10	A8 soak harness + 10k-tick soak	Prove it.
11	Genesis as objective #1	The first real mission.
12	→ Phase B	
Steps 3 and 4 deliver user-visible value before any autonomy exists, which de-risks the sequence.

11. Verification
Per step: python -m pytest tests/ --ignore=tests/runtime_qa stays green (1133 baseline). A0 must not change a single assertion.
A2: mr1 serve in one terminal; mr1 submit spec.json in another; the workflow reaches succeeded with no REPL open. mr1 pause → a newly-submitted workflow does not start. SIGTERM → clean shutdown(), state saved.
A3: an approval past its TTL flips to expired, emits approval_expired, and produces an inbox message. Fail-closed: the task stays BLOCKED.
A4 (the gate): a workflow with shell_command under a matching grant runs unattended end-to-end; the same workflow without a grant blocks on approval. Both visible in capability_audit. mr1 grant revoke → next invocation blocks again. mr1 halt → all grants revoked.
A5/A6/A7: drive a fake objective through success, transient failure (retry), planning failure (replan), budget exhaustion (quarantine + inbox message), and a blocked failure (escalate, no self-authorization).
A8: 10k virtual ticks; assert flat tick duration, bounded memory/disk, zero stuck objectives, non-stale heartbeat.
End-to-end: Genesis runs a full weekly cycle unattended — trigger fires → plan → execute under grant → propose → escalate for approval → resume on approval — with mr1 status healthy throughout and the whole thing traceable in the timeline.
The answer
What is the shortest path from today's runtime to a genuinely continuously operating autonomous system?

Four things stand in the way, and only one of them is large:

Give the process a life of its own. MR1.run() blocks on input() — the entire runtime lives inside a human REPL. The scheduler loop already works; it just has no host. (mr1 serve + control plane. Small.)
Give MR1 standing authority. This is the big one, and it is the one the QA reports believed was already done. Without objective-scoped consent grants, "unattended" is a fiction: every shell invocation needs a human, forever. (Consent grants. Medium, and it is the gate.)
Give MR1 something to want. Nothing creates work today. Objectives + the reconcile loop are a thin layer above the existing scheduler that calls the brain only on state changes. (Medium.)
Give MR1 a way to give up. Without a failure ladder with budgets and an escalation path, autonomy means either infinite retry or silent death. (Medium.)
Everything else — retention, metrics, recurring triggers, notification transport — makes continuous operation comfortable, not possible.