# Agent Ontology

Every agent in MR1 — from the root down to the shortest-lived one-shot
worker — is described by four independent properties:

| Property     | Answers                          | Values |
|--------------|-----------------------------------|--------|
| `mr_level`   | Where is it in the tree?          | `1`, `2`, `3`, ... (an integer, unbounded except by `height_limit` in `config.yml`) |
| `role`       | What can it do?                   | `worker` \| `orchestrator` |
| `lifecycle`  | How long is it expected to exist? | `ephemeral` \| `task_scoped` \| `project_scoped` \| `standing` |
| `clearance`  | What risk may it authorize?       | `0.0`–`1.0` (`security_clearance`, unchanged by this document) |

These are stored (or computed) independently. None of them may be inferred
from another — an agent's title, its `mr_level`, and its `role` never imply
one another.

This document is the canonical reference. If other docs (README, model
tables) describe the ontology differently, this one wins.

---

## 1. MRN hierarchy semantics

`MRN` is depth notation, not a type name. `N` is the agent's distance from
the root:

```
MR1  = the unique root agent, mr_level = 1
MR2  = any agent one level below MR1, mr_level = 2
MR3  = any agent one level below an MR2, mr_level = 3
MRN  = the general family — an agent at arbitrary level N
```

**MR2 does not name one agent.** There can be — and typically are — many
agents at `mr_level = 2` simultaneously, each with its own title. "MR2" is
never used as an agent's actual identity; it is metadata describing its
position in the tree. See [§6 Naming rules](#6-naming-rules).

Invariant, enforced at creation and checked by `mr1 doctor`:

```
child.mr_level == parent.mr_level + 1
MR1.mr_level == 1
MR1.parent_agent_id is None
```

`mr_level` is stored explicitly on every persisted agent record
(`AgentRecord.mr_level` in `mr1/scoped_agents.py`) — it is derived once at
creation time from `parent.mr_level + 1` and persisted, not recomputed by
walking the tree at read time. This makes it possible to validate the
invariant independently (a corrupted or hand-edited record whose
`mr_level` no longer matches `parent.mr_level + 1` is a doctor-reported
error, not silently trusted).

The ephemeral orchestrator-recursion path (`mr1/mrn.py`) computes levels
the same way at each recursive spawn (`child_level = level + 1`), gated by
`Dispatcher.validate_spawn_level`, which also enforces the `height_limit`
ceiling from `config.yml`.

## 2. Role semantics

```
worker        — performs a focused, bounded unit of work.
                Does not coordinate children. May be ephemeral or
                longer-lived. Used aggressively to isolate context and
                give a task focused attention without polluting a
                parent's context.

orchestrator  — reasons about and coordinates work. May create workflows,
                message other agents, and create or supervise children
                when permitted. Includes MR1 itself and every subordinate
                orchestration agent.
```

Role is stored explicitly (`AgentRecord.role`) and is **never** inferred
from `mr_level`, `lifecycle`, or title. MR1 is `role="orchestrator"`
because that is what is recorded for it, not because it happens to be
`mr_level=1`.

Today, only `role="orchestrator"` agents are persisted as `AgentRecord`s
(both MR1 and every child created via `AgentStore.create_child_agent`).
Workers (`mr1/worker.py`) run as one-shot subprocesses and are never
written to the agent store — there is no `AgentRecord` for a worker to
inspect after it finishes; its result is returned directly to its caller.
The ontology does not require this — the ontology's ordinary requirement is
just that if role=worker were persisted, it wouldn't imply anything about
`mr_level` or `lifecycle` — but no code path exercises that combination
today, so it is not covered by the test suite.

## 3. Lifecycle semantics

```
ephemeral       — one bounded action, then the agent is gone. No record
                  persists beyond a log line. This is a worker's normal
                  shape.

task_scoped     — bounded to one task tree: may coordinate children before
                  finishing, but exits when the tree completes and is not
                  persisted. This is the ephemeral orchestrator-recursion
                  path's shape (mr1/mrn.py, levels 2+).

project_scoped  — persisted; lives until explicitly terminated, typically
                  because it owns an area, domain, or project. The default
                  lifecycle for a persisted child created via
                  create_child_agent.

standing        — persisted; expected to run indefinitely (an ongoing
                  monitor, or MR1 itself). Not auto-assigned — a caller
                  opts a child into it explicitly via
                  create_child_agent(..., lifecycle="standing").
```

Lifecycle is a design-time expectation about duration, stored on the
record (`AgentRecord.lifecycle`). It is a **different axis** from the
runtime activity status derived by `derive_lifecycle_state()`
(`status/run_status` → `active`/`idle`/`working`/`terminated`/...). The two
are easy to conflate because both used to be shown under a single
"lifecycle:" label in CLI/TUI output — that label now reads `lifecycle:`
for the ontology field and `activity:` for the derived runtime status. See
`mr1/scoped_agents.py` for the exact split.

Terminal behavior: `AgentStore.terminate_agent` works the same regardless
of `lifecycle` — a `standing` agent can be terminated just like a
`project_scoped` one; `lifecycle` is not itself a state machine with its
own transitions. MR1 (`mr_level=1`) is the one hard exception:
`terminate_agent` refuses to terminate the root agent at all.

## 4. Orthogonality

```
mr_level     answers: where is it in the tree?
role         answers: what can it do?
lifecycle    answers: how long does it exist?
clearance    answers: what risk may it authorize or execute?
```

None of the four may be derived from another when deciding what an agent
is. The one place they are deliberately *combined* — not derived, combined
— is `mr1.scoped_agents.actor_category(role, mr_level)`, used only by
`capability_policy.py`'s clearance-threshold lookup:

```python
def actor_category(role: str, mr_level: int) -> str:
    if role == "orchestrator" and mr_level == 1:
        return "root_orchestrator"
    return role
```

This exists because MR1 has always carried a higher default direct-action
clearance threshold than any other orchestrator (`0.50` vs. `0.20`), and a
worker a lower one still (`0.00`) — a real, pre-existing policy distinction
this refactor preserves exactly, not a new one. `actor_category` is a
downstream *policy* concern, not part of the ontology itself: `role` is
still stored and read independently of `mr_level` everywhere else.

## 5. Examples

```
MR1 root orchestrator:
    mr_level=1, role=orchestrator, lifecycle=standing
    title="MR1" (the one agent where the title is also its conventional
    display name — every other agent gets a real, human-chosen title)

MR2 project-scoped orchestrator:
    mr_level=2, role=orchestrator, lifecycle=project_scoped
    title="Repository Inspector"  (chosen by the user or MR1's brain;
    never defaults to "MR2")

MR3 ephemeral worker:
    role=worker, lifecycle=ephemeral
    Spawned by an MR2 or MR3 orchestrator to do one bounded, scoped job
    (read a file, run a check, summarize something) and exit. Workers are
    not currently persisted, so "mr_level=3" here describes where in the
    delegation chain it was spawned from, not a stored AgentRecord field.

A standing MR2 monitor:
    mr_level=2, role=orchestrator, lifecycle=standing
    title="Uptime Watchtower" — created via
    create_child_agent(parent, "Uptime Watchtower", lifecycle="standing")
    for a task that's explicitly indefinite (e.g. "watch this and tell me
    if it breaks"), rather than the project_scoped default.
```

## 6. Naming rules

- `mr_level` is metadata. It is never a substitute for a title.
- An agent's title is semantic and human-readable, chosen explicitly by
  the user or generated meaningfully by MR1's brain (e.g. "Sentinel",
  "Architect", "Repository Inspector") — never defaulted to `"MR2"`,
  `"MR3"`, or any other level-shaped string.
- Where a title is genuinely unset (a caller didn't supply one), the
  fallback is the literal string `"Unnamed agent"` — not a level or role
  token. See `build_assignment_packet` / `render_assignment_mission` in
  `mr1/scoped_agents.py`.
- Explicit user-supplied titles always win, including a title that happens
  to look like `"MR2"` — a user is free to name an agent whatever they
  want; the rule is about the *system* never generating that as a
  default.
- Titles are case-insensitively unique and permanently reserved once
  taken, even after the agent that held them is terminated
  (`AgentStore`'s title index) — this refactor does not change that.

## 7. Creation and delegation invariants

- Only `role="orchestrator"` agents may create children today
  (`AgentStore.create_child_agent` requires a caller that resolves through
  the same store; there is no code path for a `role="worker"` agent to
  create anything).
- `child.mr_level = parent.mr_level + 1`, always. No agent may create a
  grandchild directly, and no `mr_level` may be chosen freely by a caller
  or by the LLM brain — it is computed from the parent, never passed in as
  a free-form number.
- `child.security_clearance <= parent.security_clearance`, unchanged by
  this refactor.
- The `[DELEGATE]` block MR1's brain (and every ephemeral MRn orchestrator)
  emits uses role tokens, not level or legacy-type tokens:
  `{"agent": "orchestrator", ...}` or `{"agent": "worker", ...}`. The
  actual numeric level for an `"orchestrator"` delegation is always
  computed as `caller.mr_level + 1` — the model never states a level
  directly. Older `{"agent": "mr2", ...}` / `{"agent": "kazi", ...}`
  directives are no longer recognized (see Migration below).
- Height limit (`config.yml: height_limit`) still bounds how deep the
  ephemeral orchestrator-recursion path (`mr1/mrn.py`) may go;
  unchanged by this refactor.

## 8. Persistence and migration behavior

**Schema.** An `AgentRecord` (`mr1/scoped_agents.py`) is stored as JSON
under `<runtime_root>/agents/<agent_id>.json` with `role`, `mr_level`, and
`lifecycle` as first-class fields, replacing the old flat `agent_type`
(`"mr1"` | `"mrn"` | `"kazi"`) and `tree_level` fields.

**Legacy data.** `AgentRecord.from_dict` recognizes the pre-refactor shape
automatically (`migrate_legacy_agent_dict` in the same module) and maps it
on load:

```
agent_type="mr1" | "mrn"  -> role="orchestrator"
agent_type="kazi"         -> role="worker"
tree_level                -> mr_level (same value, renamed)
lifecycle (absent)        -> "standing" if mr_level==1 else "project_scoped"
```

An `agent_type` this mapping doesn't recognize (e.g. the long-retired
`"kami"`) raises rather than silently guessing — a corrupted or
unrecognized legacy record is surfaced, not dropped.

**One-time migration.** `mr1 migrate-ontology` (implemented by
`migrate_agent_store_ontology()` in `mr1/scoped_agents.py`) rewrites every
agent file under a store's root to the current shape. It is idempotent —
a file already in the new shape is left byte-for-byte untouched, so
running it twice is a no-op the second time — and never deletes or drops
an agent: a file it can't interpret is left exactly as found and reported
separately. Runtime code doesn't require this command to be run first
(`from_dict` reads old-shape files transparently), but running it converts
a store to the canonical, alias-free shape going forward.

**No permanent aliases.** Once a store has been migrated (or was always
on the new schema), nothing in the active code path reads or writes
`agent_type`/`tree_level` again — `migrate_legacy_agent_dict` is the sole,
explicitly-labeled legacy-compatibility surface, covered by
`tests/test_ontology_migration.py`.
