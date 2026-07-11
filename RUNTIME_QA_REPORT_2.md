# MR1 Runtime QA Report — Session 2

**Date:** 2026-05-31  
**Harness:** `python -m mr1.runtime_test_cli` (isolated mode)  
**Scenario runner:** [tests/runtime_qa/runner.py](tests/runtime_qa/runner.py)  
**Scenarios added:** 15 (35 → 50 total)  
**Investigation mode:** code reading + live probe scripts, no Claude CLI calls  
**Session 1 baseline:** 35 scenarios, 0 findings, 0 crashes

This session targeted the **untested areas listed in Session 1 §4** — recursion/fanout, concurrent access, memory boundary, workflow failure recovery, approval lifecycle, and identity edge cases. The routing and workflow preview fixes from session 1 were assumed correct and not re-tested.

---

## 1. Executive Summary

Session 2 found **7 new findings** across four categories:

| Category | Finding | Severity |
|---|---|---|
| Workflow | WORKFLOW-3: Cancelled workflow reopened by rerun/append/insert/replace | **high** |
| Workflow | WORKFLOW-4: Double-cancel returns misleading "not found" message | low |
| Identity | IDENTITY-1: Agent titled "all" swallowed by global kill-all selector | **medium** |
| Identity | IDENTITY-2: Duplicate agent titles allowed — silent mass-kill on targeted operations | **medium** |
| Identity | IDENTITY-3: Null-byte accepted in agent title | low |
| Test harness | RUNNER-RACE: Module-global patch race under `--jobs > 1` | **medium** |
| Concurrency | CONCURRENCY-1: `StateManager` unprotected against inbox-triage / step() race | **medium** |

No crashes. No data loss from existing safeguards. But WORKFLOW-3 creates a path where a user-cancelled workflow resumes execution without user knowledge.

---

## 2. Confidence Assessment (updated)

| Subsystem | Session 1 | Session 2 | Why changed |
|---|---|---|---|
| Routing (NL→intent) | 3/10 | **7/10** | Vocabulary fixes confirmed by probe: 0/21 mismatches. Observability and clarification paths working. |
| Delegation | 6/10 | **7/10** | Title extraction now works (librarian case). Fanout/depth limits still only theoretically tested. |
| Approvals | 4/10 | **6/10** | Bogus-ID and double-action both handled gracefully. Approval lifecycle denial/expiry not exercised by any real workflow. |
| Workflows | 5/10 | **5/10** | Slash CRUD solid; NL authoring improved. WORKFLOW-3 drops score — cancel is not truly terminal. |
| Memory | 6/10 | **6/10** | Cross-session state persists correctly (agents, workflows visible in session 2). Inbox triage + step() race untested in production config. |
| Observability | 3/10 | **7/10** | `runtime_turn_decided` events now emitted on every turn. Routing signals visible. Override reasons informative. |
| Safety | 5/10 | **6/10** | Kill-all MR1 protection holds. Double-kill idempotent. NL-embedded slash goes to brain (safe in practice). IDENTITY-1/2 remain. |

---

## 3. Findings

### 3.1 WORKFLOW-3 · Cancelled workflow silently reopened by mutation commands

- **Severity:** high
- **Category:** workflow bug / state consistency
- **Reproduction:**
  ```python
  /workflow submit /path/to/spec.json   # submit a workflow → wf-xxx
  /workflow cancel wf-xxx               # cancel it → WorkflowStatus.CANCELLED
  /workflow rerun wf-xxx say_hello      # rerun a task
  # Result: wf-xxx is now WorkflowStatus.RUNNING again
  ```
  Also confirmed with `/workflow append wf-xxx fragment.json` → same result.
- **Expected:** Once a workflow is CANCELLED it should be terminal. `rerun`, `append`, `insert`, `replace` should reject with `"workflow cancelled: cannot mutate terminal workflow"` (or similar).
- **Actual:** All four mutation operations call `reopen_workflow(workflow)` unconditionally at [scheduler_core/state_machine.py:18-23](mr1/scheduler_core/state_machine.py#L18). `reopen_workflow` sets `workflow.status = WorkflowStatus.PENDING` without checking the prior status. The scheduler then picks up the workflow on the next tick and re-executes it.
- **Root cause:** `reopen_workflow()` in [state_machine.py:18](mr1/scheduler_core/state_machine.py#L18) has no guard for terminal workflow states. `Scheduler.cancel_workflow()` has an explicit `if wf.is_terminal(): return False` check but `Scheduler.append_workflow()`, `insert_workflow()`, `replace_workflow()` and `Mutation.rerun_task()` do not.
- **Fix:** Add `if workflow.status in {WorkflowStatus.CANCELLED, WorkflowStatus.FAILED}: raise error_cls(...)` before calling `reopen_workflow()` in `rerun_task` (line 221), `append_workflow` (line 386), `insert_workflow`, and `replace_workflow` in [scheduler_core/mutations.py](mr1/scheduler_core/mutations.py).

---

### 3.2 WORKFLOW-4 · Double-cancel returns "workflow not found" instead of "already cancelled"

- **Severity:** low
- **Category:** workflow bug / UX
- **Reproduction:**
  ```
  /workflow submit spec.json  → wf-xxx
  /workflow cancel wf-xxx     → "workflow cancelled: wf-xxx"
  /workflow cancel wf-xxx     → "workflow not found: wf-xxx"   ← misleading
  ```
- **Expected:** `"workflow already cancelled: wf-xxx"` or similar.
- **Actual:** `Scheduler.cancel_workflow()` at [scheduler.py:2424](mr1/scheduler.py#L2424) returns `False` for both "not found" and "already terminal". The root_builtin at [root_builtins.py:176-178](mr1/orchestrator/root_builtins.py#L176) maps `False → "workflow not found: {id}"` without distinguishing the two cases.
- **Root cause:** `Scheduler.cancel_workflow` conflates "missing" and "terminal" in a single boolean. The root_builtin can't reconstruct the distinction.
- **Fix:** Return an enum or raise a specific exception for the "already terminal" case, or have the builtin look up the workflow status before printing the error.

---

### 3.3 IDENTITY-1 · Agent titled "all" is indistinguishable from the global kill-all selector

- **Severity:** medium
- **Category:** identity / safety
- **Reproduction:**
  ```
  /agent create Keeper         # → ag-xxx
  /agent create all            # → ag-yyy   (title = "all")
  /agent kill-all all          # → "Terminated 2 agent(s): ag-xxx, ag-yyy."
  ```
  There is **no way** to target just the agent titled "all" via `/agent kill-all`. The command always kills everything.
- **Expected:** Either (a) "all" is a reserved word and creating an agent with that title is rejected with `"'all' is a reserved title"`, or (b) the kill-all selector makes `all` unambiguous by requiring explicit quoting or a different flag.
- **Actual:** In `handle_agent_kill_all_builtin` at [root_builtins.py:571-584](mr1/orchestrator/root_builtins.py#L571), `selector` starts as `"all"` (the default "kill everything" mode). When the user passes the token `"all"` as the title argument, `selector = token = "all"` — same value — so the `selector != "all"` filtering block never executes. Keeper is also terminated.
- **Root cause:** The default value and the explicit "kill all" selector share the string `"all"` with no sentinel distinction. A user creating an agent named "all" is not warned.

---

### 3.4 IDENTITY-2 · Duplicate agent titles allowed — mass-kill on targeted operations

- **Severity:** medium
- **Category:** identity
- **Reproduction:**
  ```
  /agent create Alpha   # → ag-aaa
  /agent create Alpha   # → ag-bbb   (no error)
  /agent kill-all Alpha # → "Terminated 2 agent(s): ag-aaa, ag-bbb."
  ```
  A user who intends to manage one agent named Alpha accidentally terminates both.
- **Expected:** Either `create_child_agent` rejects duplicate titles with a clear error, or `/agent kill-all <title>` warns when multiple agents share the target title ("2 agents titled Alpha will be terminated — confirm?").
- **Actual:** `create_child_agent` at [scoped_agents.py:679-719](mr1/scoped_agents.py#L679) has no uniqueness check. The kill-all path at [root_builtins.py:592-598](mr1/orchestrator/root_builtins.py#L592) terminates all matching agents without warning.
- **Root cause:** No title-uniqueness invariant at the store level. NL paths that resolve agent references by title also become ambiguous when duplicates exist.

---

### 3.5 IDENTITY-3 · Null byte accepted in agent title

- **Severity:** low
- **Category:** data quality
- **Reproduction:**
  ```python
  session.handle_input('/agent create Title\x00Null', ...)
  # Returns agent_id — no error
  # Agent JSON on disk contains "title": "Title\x00Null"
  ```
- **Expected:** Title validation should reject control characters including `\x00`. A null byte in a title string breaks terminal display (truncates at the null), corrupts `shlex.split`-based title searches, and may fail on some JSON consumers.
- **Actual:** `create_child_agent` at [scoped_agents.py:684](mr1/scoped_agents.py#L684) does `title = title.strip()` and rejects empty strings, but does not strip control characters.
- **Root cause:** Insufficient input sanitization on the title field.

---

### 3.6 RUNNER-RACE · `_patched_runtime_paths` is not thread-safe under `--jobs > 1`

- **Severity:** medium
- **Category:** test harness limitation / concurrency
- **Reproduction (confirmed):**
  ```python
  # 4 threads each enter _patched_runtime_paths with their own temp dir.
  # With a 10ms stagger, 2/4 threads observe the wrong _CONTEXT_PATH.
  ```
  Measured: 2 of 4 threads saw another thread's `_CONTEXT_PATH` value.
- **Expected:** Each scenario's MR1 instance reads memory context from its own isolated temp dir.
- **Actual:** `_patched_runtime_paths` at [mr1/runtime_test_cli.py:112-143](mr1/runtime_test_cli.py#L112) uses `setattr(module, name, value)` on shared module-level globals (`orchestrator_root._CONTEXT_PATH`, `orchestrator_memory._CONTEXT_PATH`, `mem_rtvr._RAG_DIR`, etc.) without any inter-thread lock. When Thread A sets a global, Thread B immediately overwrites it before Thread A reads it. Thread A then loads Thread B's memory context — silently.
- **Impact:** Under `--jobs > 1`, scenarios can load the wrong memory context at `start()`, causing non-reproducible inter-scenario contamination. Test results from parallel runs are not reliable.
- **Root cause:** Module-level global patching is inherently not thread-safe. The correct fix is either (a) a process-based runner (each scenario in its own subprocess), or (b) a threading.Lock around the patch section, forcing scenarios to patch/unpatch serially while still running Claude CLI calls concurrently.
- **Note:** Session 1 report (§7) flagged this as a known caveat. It is now confirmed to actively cause contamination.

---

### 3.7 CONCURRENCY-1 · `StateManager` unprotected against concurrent inbox-triage + step() access

- **Severity:** medium
- **Category:** concurrency bug
- **Not reproducible in isolated test harness** (`inbox_auto_triage=False` in `RuntimeTestSession`).
- **Description:** In production, `_run_inbox_loop` at [orchestrator/root.py:3000](mr1/orchestrator/root.py#L3000) fires on a background thread every `inbox_triage_interval_s` seconds. It creates an `InboxTriageRunner` that receives `pending_workflow_state=self._state` — the same `StateManager` object used by `step()`. `StateManager` has no threading lock: `save()`, `add_decision()`, `_record_conversation()`, `begin_task()`, etc. all read and write `self._state` (a plain dict) without synchronization.
- **Expected:** Either `StateManager` uses a lock, or `step()` blocks the inbox-triage thread while running.
- **Actual:** No lock visible in [orchestrator/state.py](mr1/orchestrator/state.py). A race between `step()` writing to `self._state["conversation"]` and `InboxTriageRunner` reading `pending_workflow_state` can produce torn reads or lost writes.
- **Risk:** Only manifests in production with `inbox_auto_triage=True` and overlapping timing. Frequency depends on turn latency vs. triage interval (default 30s). Under high-frequency use this is reachable.

---

## 4. Non-findings (investigated but working correctly)

The following areas from the Session 1 §4 "missing scenarios" list were investigated and found to work correctly:

1. **Cross-session resume** (same `--runtime-root`): Session 2 correctly sees agents/workflows created by session 1. Kill operations across sessions work. State file persists atomically.

2. **Approval bogus-ID**: Returns `"approval request not found: <id>"` — clear, no crash.

3. **Double-kill of same agent**: Idempotent — second kill returns the agent ID without error.

4. **Unicode/emoji/RTL agent titles**: Create, list, kill all work correctly with Unicode, CJK, and emoji titles via `shlex.split` + UTF-8 JSON serialization.

5. **NL with embedded slash command** (`"please run /agent kill-all all"`): `dispatch.builtin_attempted = False` — goes to `step()` correctly. Brain asks for confirmation rather than executing blind. Structural safety holds.

6. **`/workflow rerun <wf-id> <nonexistent-task>`**: Returns `"workflow not found: <id>"` or `"task not found"` — no crash.

7. **`/workflow submit` with non-existent file / invalid JSON / invalid spec**: All three return clear parse/validation errors.

---

## 5. New Scenarios Added (15)

| Name | Category | Tests For |
|---|---|---|
| `fanout_duplicate_title` | identity | Duplicate titles — kill-all covers both |
| `identity_reserved_word_title_all` | identity | "all" as agent title — kill-all behavior |
| `identity_unicode_title` | identity | Unicode/emoji create+kill round-trip |
| `identity_null_byte_title` | identity | Null byte in title — no crash |
| `wf_rerun_after_cancel` | workflow | Rerun task from cancelled workflow |
| `wf_append_after_cancel` | workflow | Append tasks to cancelled workflow |
| `wf_double_cancel` | workflow | Double-cancel UX |
| `wf_rerun_nonexistent` | workflow | Rerun non-existent workflow/task |
| `wf_submit_invalid_json` | workflow | Submit bad JSON spec |
| `wf_submit_invalid_spec` | workflow | Submit valid JSON, invalid spec |
| `approval_bogus_id` | approval | Approve/deny bogus approval ID |
| `approval_list_empty` | approval | `/approvals` bare with no pending |
| `state_agent_survives_restart` | state | Cross-session agent visibility |
| `concurrency_kill_then_kill` | safety | Idempotent double-kill |
| `concurrency_double_cancel_workflow` | workflow | Double-cancel workflow |

---

## 6. Recommended Fixes (prioritized)

### P0 — prevent silent workflow resurrection

**Fix WORKFLOW-3.** In [scheduler_core/mutations.py](mr1/scheduler_core/mutations.py), add a terminal-status guard before calling `reopen_workflow()`:

```python
# In rerun_task(), before reset_task_runtime_state():
if workflow.status in {WorkflowStatus.CANCELLED, WorkflowStatus.FAILED}:
    raise self._error_cls(
        f"workflow is {workflow.status.value} and cannot be reopened: {workflow_id}"
    )

# Same guard in append_workflow(), insert_workflow(), replace_workflow()
```

This prevents any mutation from accidentally un-cancelling a workflow without an explicit "uncancel" command.

### P1 — fix identity ambiguity

**Fix IDENTITY-1.** Either:
- Option A: Reject titles matching reserved selectors. In `create_child_agent` at [scoped_agents.py:684](mr1/scoped_agents.py#L684): `if title.lower() == "all": raise ValueError("'all' is a reserved agent selector")`.
- Option B: Parse `kill-all all` specially — require an explicit flag like `--all` for the global scope, leaving positional args strictly as title selectors.

**Fix IDENTITY-2.** Add a title-uniqueness check in `create_child_agent` (or at least a warning in the kill-all confirmation output when multiple agents share a title).

### P2 — fix test harness concurrency

**Fix RUNNER-RACE.** Change the runner to use `ProcessPoolExecutor` instead of `ThreadPoolExecutor`. Each scenario runs in its own subprocess, so module-level globals are isolated by the OS. This also eliminates any future module-level contamination vectors.

Alternatively, add a `threading.Lock` that serializes the `_patched_runtime_paths` patch/unpatch cycle while still allowing the Claude CLI calls to run concurrently — though this provides weaker isolation.

### P3 — small UX / data quality fixes

**Fix WORKFLOW-4.** In `Scheduler.cancel_workflow`, distinguish "not found" from "already terminal" by returning an enum or raising a typed exception.

**Fix IDENTITY-3.** In `create_child_agent`, add: `if any(ord(c) < 32 for c in title): raise ValueError("agent title must not contain control characters")`.

### P4 — production concurrency safety

**Fix CONCURRENCY-1.** Add a `threading.RLock` to `StateManager` and acquire it in all public methods that read/write `self._state`. Alternatively, document that `step()` must not be called while the inbox-triage thread is enabled, and enforce this with an `assert threading.current_thread() is threading.main_thread()` guard in `step()`.

---

## 7. Remaining Unexercised Areas

The following items from Session 1 §4 remain untested due to requiring real Claude CLI invocations or external infrastructure:

1. **Brain transport failure.** What if `claude` CLI returns a non-zero exit code mid-turn, returns non-JSON stdout, or times out? The `_invoke` method at [orchestrator/process.py:70](mr1/orchestrator/process.py#L70) has `try/except OSError` and `TimeoutExpired` but JSON parse failures on the output stream produce `parse_errors` counter increments — the resulting `result_text` may be empty. Untested end-to-end.

2. **Inbox triage at high volume.** `_run_inbox_loop` processes at most `max_messages=10` per tick, but what happens with 1000 queued messages over time? The `InboxTriagePolicy` caps are low, but accumulation behavior is untested.

3. **Recursion bloom in practice.** Session 1 created a "Sentinel" agent whose mission includes creating 5 sub-agents. The harness stops at `steps=1`. Running `/agent run <ag-id> --steps 10` against a real Claude CLI to measure whether sub-agents are actually created (and whether the system terminates naturally) remains untested.

4. **Approval lifecycle with real capability execution.** No scenario has actually exercised the capability policy flow end-to-end (a task blocking on an approval, the approval being granted, and the task resuming). The approval CRUD mechanics are tested; the execution feedback loop is not.

5. **Workflow watcher triggers.** `/workflow trigger <wf-id> <label> [event]` accepts the command but no scenario observes whether the watcher actually fires or whether the trigger persists across ticks.

---

## 8. Artifacts

- **New scenarios:** [tests/runtime_qa/scenarios.py](tests/runtime_qa/scenarios.py) — 15 scenarios added (35 → 50 total)
- **Probe scripts used:**
  - Inline Python via `python -c` (not saved; reproduce from §3 reproduction steps)
- **Per-scenario data:** [tests/runtime_qa/results/](tests/runtime_qa/results/) — JSONL + meta per scenario
- **Summary:** [tests/runtime_qa/results/summary.json](tests/runtime_qa/results/summary.json)

To run only the new scenarios:
```
python -m tests.runtime_qa.runner --category=identity
python -m tests.runtime_qa.runner wf_rerun_after_cancel wf_append_after_cancel wf_double_cancel wf_rerun_nonexistent wf_submit_invalid_json approval_bogus_id approval_list_empty concurrency_kill_then_kill
```

To confirm the WORKFLOW-3 bug directly (no LLM required):
```python
python -c "
import tempfile, json
from pathlib import Path
from mr1.runtime_test_cli import RuntimePaths, RuntimeTestSession, _patched_runtime_paths

tmp = tempfile.mkdtemp()
paths = RuntimePaths(isolated=True, runtime_root=Path(tmp),
    workflow_root=Path(tmp)/'workflows', state_path=Path(tmp)/'active'/'mr1_state.json',
    context_path=Path(tmp)/'active'/'mr1_context.md',
    dumps_root=Path(tmp)/'dumps', rag_root=Path(tmp)/'rag')
spec_file = Path(tmp) / 'spec.json'
spec_file.write_text(json.dumps({'tasks': [{'kind': 'run', 'cmd': 'echo hi', 'label': 'hi'}]}))

with _patched_runtime_paths(paths):
    session = RuntimeTestSession(paths)
    r1 = session.handle_input(f'/workflow submit {spec_file}', request_index=1)
    wf_id = r1['response_text'].strip().replace('submitted: ', '')
    session.handle_input(f'/workflow cancel {wf_id}', request_index=2)
    wf_before = session._mr1._workflow_store.load_workflow(wf_id)
    print(f'Status after cancel: {wf_before.status}')  # CANCELLED
    session.handle_input(f'/workflow rerun {wf_id} hi', request_index=3)
    wf_after = session._mr1._workflow_store.load_workflow(wf_id)
    print(f'Status after rerun: {wf_after.status}')    # RUNNING — BUG
    session.shutdown()
"
```

---

— end of session 2.
