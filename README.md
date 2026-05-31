# MR1 — Multi-Agent Orchestration System

MR1 is a persistent, terminal-based multi-agent system built on the Claude CLI. A top-level orchestrator (`MR1`) maintains a live conversation with the user and routes tasks to specialised sub-agents: `Kami` handles complex, multi-step work with broad tool access, while `Kazi` handles scoped one-shot jobs. A set of lightweight mini-agents manages memory, context packaging, and communication summarisation without ever invoking an LLM — all deterministic, no surprises.

## How to run

```
python main.py
```

`python main.py` runs the plain chat loop.

To launch the read-only runtime TUI instead:

```bash
python main.py --tui
```

or:

```bash
python -m mr1.tui
```

The plain chat path also remains available explicitly:

```bash
python main.py --plain
```

## Runtime TUI

The new TUI is a separate read-only runtime viewer over the persisted MR1 state.

It provides:

- a live MR1/MRn tree with MR1 pinned at the top
- keyboard-first navigation by parent, child, and sibling
- timeline mode for recent runtime events
- a right-side detail panel for the selected agent or event
- dimmed terminated agents and a dead-agent visibility toggle
- live-follow mode that can be frozen while inspecting history

Inside the plain loop, `/vizualize`, `/visualize`, and `/visualize-web` now print guidance for the TUI rather than launching legacy UI paths.

For synthetic workload generation in the plain loop, use:

```text
/test spawn agents 3
/test kill agents
```

## Agent hierarchy

```
MR1 (persistent orchestrator — haiku)
 ├── Kami  (senior autonomous agent — haiku)
 │    └── Kazi  (task worker — haiku)
 └── Kazi  (task worker — haiku)

Mini agents (no LLM calls, deterministic):
 ├── mem_dltr   memory distillation & garbage collection
 ├── mem_rtvr   memory retrieval (chromadb RAG + dump search)
 ├── ctx_pkgr   context packaging for Kazi prompts
 └── com_smrzr  communication summarisation → RAG ingestion
```

MR1 decides per-turn whether to answer directly, delegate to Kami (complex), or delegate to Kazi (simple). All spawns pass through the `Dispatcher` permission gate before any subprocess is created.

## Built-in commands (while MR1 is running)

| Command    | Effect                                      |
|------------|---------------------------------------------|
| `/status`  | Show session ID, active tasks, recent decisions |
| `/tasks`   | List all tasks with status icons            |
| `/kill`    | Terminate all running agents                |
| `/history` | Show recent conversation turns              |
| `/vizualize` | Show how to launch the runtime TUI |
| `/visualize-web` | Alias that now points to the runtime TUI |
| `/test spawn agents <h>` | Spawn a synthetic full binary tree of worker processes |
| `/test kill agents` | Kill all synthetic worker processes |
| `exit`     | Save session state and quit                 |

## Phase 1: Workflows

Phase 1 adds a deterministic workflow scheduler that runs inside the MR1 process. Workflow control does not invoke MR1 reasoning: submission, discovery, scheduling, event logging, and inspection all go through the store, scheduler, and workflow CLI only.

Supported commands in the plain loop:

| Command | Effect |
|---------|--------|
| `/workflows` | List all known workflows |
| `/workflow <id>` | Show one workflow and its tasks |
| `/workflow submit <path>` | Load a JSON spec from disk and submit it |
| `/workflow rerun <id> <task>` | Reset one task for another execution attempt |
| `/workflow cancel <id>` | Cancel one workflow |
| `/workflow append <id> <path>` | Append task(s) from a JSON fragment |
| `/workflow insert <id> <after_task> <path>` | Insert one task after an existing task |
| `/workflow replace [-r] <id> <task> <path>` | Replace one unstarted, failed, cancelled, or skipped task, optionally rerunning immediately |
| `/workflow trigger <id> <label-or-task-id> [event_name]` | Trigger a manual watcher |
| `/task cancel <task_id>` | Cancel one task |
| `/task <id>` | Show one task's detail |
| `/result <task_id>` | Show one task's normalized output |
| `/inputs <task_id>` | Show one task's resolved workflow inputs |
| `/artifacts <workflow_id>` | List registered artifacts for a workflow |
| `/jobs` | List live workflow tasks |
| `/watchers` | List active watcher tasks |
| `/agents` | List persistent scoped agents visible to the caller |
| `/agent create <title>` | Create a scoped MRn child agent |
| `/agent <ag-id>` | Show one scoped agent record and its reports |
| `/agent run <ag-id> --steps N` | Run a bounded multi-step MRn loop under explicit policy |
| `/agent kill <ag-id>` | Terminate a scoped agent |
| `/inbox` | List MR1/root inbox messages |
| `/inbox --archived` | List MR1/root inbox including archived messages |
| `/inbox triage --max-actions N --max-messages N` | Run one bounded inbox triage pass |
| `/outbox` | List MR1/root sent messages |
| `/message <msg-id>` | Show one persistent message |
| `/message read <msg-id>` | Mark one persistent message as read |
| `/message archive <msg-id>` | Archive one persistent message |
| `/message send <ag-id> <subject> <body-file>` | Send one root-scoped persistent message |
| `/timeline recent` | Show recent unified runtime timeline events |
| `/timeline show <event_id>` | Show one unified timeline event |
| `/timeline trace <correlation_id>` | Trace one causal chain |
| `/timeline blocked` | Show currently blocked timeline items |
| `/timeline approvals` | Show approval lifecycle events |
| `/agent kazi` | Show one runtime agent profile |
| `/capabilities` | List all registered capabilities across tools, watchers, and agents |
| `/capability <name>` | Show one capability contract |
| `/schema [section]` | Show workflow schema metadata (`workflow`, `task`, `inputs`, `refs`, `task-kinds`) |
| `/tools` | List registered deterministic workflow tools |
| `/tool <type>` | Show one tool contract |
| `/events <workflow_id>` | Show recent workflow events |
| `/scheduler tick` | Force one deterministic scheduler pass |

Phase 1 started with DAGs of Kazi agent tasks. Phase 2 adds deterministic watcher tasks that gate downstream work without invoking an LLM.

A minimal agent-only spec looks like:

```json
{
  "title": "Example workflow",
  "tasks": [
    {
      "label": "a",
      "title": "First task",
      "prompt": "Inspect the repository state"
    },
    {
      "label": "b",
      "title": "Second task",
      "prompt": "Summarize the findings from task a",
      "depends_on": ["a"]
    }
  ]
}
```

You can submit the same spec without entering MR1 by using the deterministic CLI:

```bash
python -m mr1.workflow_cli submit path/to/workflow.json
python -m mr1.workflow_cli compile-workflow path/to/request.txt
python -m mr1.workflow_cli compile-workflow path/to/request.txt --submit
python -m mr1.workflow_cli rerun <workflow_id> <task_label_or_id>
python -m mr1.workflow_cli cancel-task <task_id>
python -m mr1.workflow_cli cancel-workflow <workflow_id>
python -m mr1.workflow_cli append-workflow <workflow_id> path/to/fragment.json
python -m mr1.workflow_cli insert-workflow <workflow_id> <after_task> path/to/task.json
python -m mr1.workflow_cli replace-workflow [-r] <workflow_id> <task_label_or_id> path/to/task.json
python -m mr1.workflow_cli workflows
python -m mr1.workflow_cli workflow <workflow_id>
python -m mr1.workflow_cli capabilities
python -m mr1.workflow_cli capability shell_command --json
python -m mr1.workflow_cli schema
python -m mr1.workflow_cli schema inputs --json
python -m mr1.workflow_cli tools
python -m mr1.workflow_cli tool shell_command --example
python -m mr1.workflow_cli agents
python -m mr1.workflow_cli agent create research
python -m mr1.workflow_cli agent-assign <ag-id> path/to/mission.txt
python -m mr1.workflow_cli agent-step <ag-id>
python -m mr1.workflow_cli agent-run <ag-id> --steps 5
python -m mr1.workflow_cli agent <ag-id>
python -m mr1.workflow_cli agent kill <ag-id>
python -m mr1.workflow_cli inbox
python -m mr1.workflow_cli inbox --archived
python -m mr1.workflow_cli outbox
python -m mr1.workflow_cli message <message_id>
python -m mr1.workflow_cli message-read <message_id>
python -m mr1.workflow_cli message-archive <message_id>
python -m mr1.workflow_cli message-send <ag-id> "subject" path/to/body.txt
python -m mr1.workflow_cli timeline recent
python -m mr1.workflow_cli timeline show <event_id>
python -m mr1.workflow_cli timeline trace <correlation_id>
python -m mr1.workflow_cli timeline blocked
python -m mr1.workflow_cli timeline approvals
python -m mr1.workflow_cli agent kazi
python -m mr1.workflow_cli agent kazi health
python -m mr1.workflow_cli result <task_id>
python -m mr1.workflow_cli inputs <task_id>
python -m mr1.workflow_cli artifacts <workflow_id>
```

## Workflow Control

MR1 now distinguishes between a logical task and an execution attempt:

- `Task` = the logical node in the workflow DAG.
- `Attempt` = one concrete execution of that task.

Attempt history is stored per task under:

```text
mr1/memory/workflows/<wf_id>/tasks/<task_id>/attempts/<attempt_id>/
  stdout.log
  stderr.log
  result.json
```

`attempt_id` starts at `1`, increases strictly by `1`, is never reused, and always matches the directory name under `attempts/`.

Task control semantics:

- `rerun` resets a task in `failed`, `timed_out`, `cancelled`, `succeeded`, or `skipped` state back to `waiting` or `ready` without deleting prior attempts.
- the next real launch allocates the next `attempt_id`; rerun itself does not consume an attempt number.
- `cancel-task` cancels a running task or marks a queued task as cancelled.
- `cancel-workflow` cancels every non-terminal task in the workflow.

Workflow mutation semantics:

- `append` adds new task nodes without changing existing task definitions.
- `insert` adds one task after an existing task and rewires that task's direct children through the inserted node.
- `replace` keeps the same `task_id` and label, but swaps the task definition for an unstarted task or a failed/timed-out/cancelled/skipped task.
- plain `replace` stops with the replaced task in `ready` or `waiting`; `replace -r` immediately ticks once so execution resumes.

Output semantics:

- `output.json` remains the canonical normalized output for the latest successful attempt only.
- failed, timed-out, and cancelled attempts keep their own `result.json` under the attempt directory and do not overwrite `output.json`.

## Phase 8: Conditional Branching

Phase 8 adds deterministic branching to workflow tasks.

- `run_if` is an optional task-level condition evaluated only after the dependency gate passes.
- if `run_if` evaluates true, the task becomes `ready` and may run.
- if `run_if` evaluates false, the task becomes `skipped`.
- `skipped` is terminal but non-failing. It does not count as `succeeded` and it does not create an execution attempt.
- `dependency_policy` controls how joins interpret upstream branch outcomes.

Supported `run_if` operators:

- `eq`
- `ne`
- `contains`
- `exists`
- `missing`
- `gt`
- `gte`
- `lt`
- `lte`
- `truthy`
- `falsy`

Supported dependency policies:

- `all_succeeded` (default): every dependency must succeed; a skipped dependency skips the task and a failed dependency blocks it.
- `any_succeeded`: wait for all dependencies to become terminal, then continue if at least one succeeded; otherwise skip.

Branch example:

```json
{
  "title": "Conditional branch",
  "tasks": [
    {
      "label": "check",
      "title": "Check exit code",
      "task_kind": "agent",
      "agent_type": "kazi",
      "prompt": "Run the check."
    },
    {
      "label": "success_path",
      "title": "Success path",
      "task_kind": "agent",
      "agent_type": "kazi",
      "depends_on": ["check"],
      "run_if": {
        "ref": "check.result.data.exit_code",
        "op": "eq",
        "value": 0
      },
      "prompt": "Handle the success case."
    },
    {
      "label": "failure_path",
      "title": "Failure path",
      "task_kind": "agent",
      "agent_type": "kazi",
      "depends_on": ["check"],
      "run_if": {
        "ref": "check.result.data.exit_code",
        "op": "ne",
        "value": 0
      },
      "prompt": "Handle the failure case."
    },
    {
      "label": "final",
      "title": "Join",
      "task_kind": "agent",
      "agent_type": "kazi",
      "depends_on": ["success_path", "failure_path"],
      "dependency_policy": "any_succeeded",
      "prompt": "Summarize the branch result."
    }
  ]
}
```

Phase 2 watcher tasks use `task_kind: "watcher"` plus a watcher-specific `watcher_type` and `watch_config`:

```json
{
  "label": "wait_file",
  "title": "Wait for file",
  "task_kind": "watcher",
  "watcher_type": "file_exists",
  "watch_config": {
    "path": "/tmp/some_file.txt"
  }
}
```

Supported watcher types:

| Watcher | Required config | Meaning |
|---------|------------------|---------|
| `file_exists` | `path` | Succeeds when the path exists |
| `time_reached` | `at` | Succeeds when current time reaches the timestamp |
| `manual_event` | `event` | Succeeds only after an explicit trigger |
| `condition_script` | `path` | Runs a deterministic script where exit `0/1/other` means `satisfied/not_satisfied/failed` |

Watcher inspection and manual trigger are available both inside MR1 and via the deterministic CLI:

```bash
python -m mr1.workflow_cli watchers
python -m mr1.workflow_cli trigger <workflow_id> <label-or-task-id> [event_name]
```

An example watcher workflow is available at `examples/workflows/watcher_demo.json`. Manual smoke test:

1. Submit the example workflow.
2. Run `/scheduler tick` until `/watchers` shows `wait_file` as running.
3. Create the file with `touch /tmp/mr1_watcher_demo.txt`.
4. Run `/scheduler tick` again.
5. Confirm the watcher succeeded and the downstream Kazi task unlocked.

## Phase 3: Workflow Dataflow + Artifacts

Phase 3 standardises task outputs and lets downstream tasks consume upstream results through deterministic references. `depends_on` still controls scheduling only. `inputs` controls data passing.

Normalized task outputs are written to:

```text
mr1/memory/workflows/<wf_id>/tasks/<task_id>/output.json
```

The normalized schema is:

```json
{
  "task_id": "tk-...",
  "workflow_id": "wf-...",
  "status": "succeeded",
  "summary": "Short human-readable summary",
  "text": "Main textual output",
  "data": {},
  "metrics": {},
  "artifacts": [],
  "created_at": "...",
  "metadata": {}
}
```

Supported input references:

```text
<label>.result
<label>.result.summary
<label>.result.text
<label>.result.data
<label>.result.data.<key>
<label>.result.metrics
<label>.result.metrics.<key>
<label>.stdout
<label>.stderr
<label>.artifact.<artifact_name>
```

Artifact metadata is stored by path and never inlined into `workflow.json`. Artifact names are exact-match and must be unique per task.

Example Phase 3 workflow:

```json
{
  "title": "Dataflow demo",
  "tasks": [
    {
      "label": "producer",
      "title": "Produce text",
      "task_kind": "agent",
      "agent_type": "kazi",
      "prompt": "Write hello world."
    },
    {
      "label": "consumer",
      "title": "Consume producer output",
      "task_kind": "agent",
      "agent_type": "kazi",
      "depends_on": ["producer"],
      "inputs": [
        {"name": "producer_text", "from": "producer.result.text"}
      ],
      "prompt": "Summarize the producer text."
    }
  ]
}
```

Phase 3 adds deterministic inspection commands:

```bash
python -m mr1.workflow_cli result <task_id>
python -m mr1.workflow_cli inputs <task_id>
python -m mr1.workflow_cli artifacts <workflow_id>
```

Manual smoke test with `examples/workflows/dataflow_demo.json`:

1. Submit the workflow.
2. Run `/scheduler tick` until the producer succeeds.
3. Confirm `tasks/<producer_task_id>/output.json` exists.
4. Confirm the consumer has `inputs.json` and `materialized_prompt.txt`.
5. Run `/result <producer_task_id>`.
6. Run `/inputs <consumer_task_id>`.
7. Run `/artifacts <workflow_id>` and confirm it behaves deterministically even when no artifacts are present.

## Phase 4: Deterministic Tool Tasks

Phase 4 adds first-class deterministic tool tasks. Tools are bounded capabilities and actions. They are not agents and they do not invoke an LLM.

Tool tasks use `task_kind: "tool"` plus a `tool_type` and `tool_config`:

```json
{
  "label": "read_notes",
  "title": "Read notes",
  "task_kind": "tool",
  "tool_type": "read_file",
  "tool_config": {
    "path": "notes.txt"
  }
}
```

Supported built-in tools:

| Tool | Required config | Meaning |
|------|------------------|---------|
| `read_file` | `path` | Read a file and expose contents through normalized output |
| `write_file` | `path`, `content` | Write UTF-8 text to a file and register it as an artifact |
| `shell_command` | `argv` | Run a bounded argv command with `shell=False` and structured captured output |

MR1 now exposes a global capability layer:

- Capabilities define what the system can do.
- Workflow schema defines how a workflow must be expressed.
- Tools, watchers, and agents define how each capability is implemented.
- Capability contracts document config schema, minimal valid examples, and explicit downstream output references.

The distinction matters during workflow authoring:

- `capabilities` = what MR1 can do
- `workflow schema` = how to express workflows

## Scoped Agents

MR1 now persists a scoped agent tree alongside workflows.

- MR1 is the root agent and can see every workflow and agent branch.
- MRn agents are persistent child agents with their own `ag-...` identity, title, parent, level, lifecycle state, and owned workflows.
- Workflows are not global. Each workflow stores `owner_agent_id`, `owner_agent_title`, and `parent_agent_id`.
- MRn visibility is limited to self plus descendants. Sibling and parent branches are hidden.
- Terminating an agent blocks future workflow creation for that agent, but its existing workflows and reports remain on disk.

Scoped agent commands:

```text
/agents
/agent create research
/agent assign <ag-id> path/to/mission.txt
/agent step <ag-id>
/agent run <ag-id> --steps 5
/agent <ag-id>
/agent kill <ag-id>
```

Scoped workflow rules:

- MR1 can inspect and mutate every workflow.
- MRn can inspect and mutate only workflows it owns or workflows owned by descendant agents.
- Workflow-id commands return `access denied: workflow not in agent scope` when the caller is outside the owning branch.
- Task-id lookups resolve only inside the caller's visible workflows.

## Persistent MRn Execution Step

MR1 is the root MRn at `tree_level=1`. Persistent MRn agents can now be assigned a mission and advanced one bounded step at a time.

- `/agent step <ag-id>` runs exactly one reasoning/action iteration.
- MRn actions are structured and deterministic: create a scoped workflow, inspect a scoped workflow, write a report, ask the parent for clarification, or stay idle.
- Reports are written under `mr1/memory/agents/<agent_id>/reports/`.
- Step logs are appended to `mr1/memory/agents/<agent_id>/logs/steps.jsonl`.
- Messaging delivery, infinite loops, and autonomous background execution are not part of this phase.

## Controlled MRn Runs

MRn can now execute multiple bounded steps under an explicit MR1-controlled run policy.

- `step` runs one iteration and returns immediately.
- `run` executes up to a fixed number of iterations and stops deterministically when policy requires it.
- There is still no background daemon, scheduler loop, or unbounded autonomy.

Default run policy:

- `max_steps=3`
- `max_workflows_created=2`
- `stop_on_parent_message=true`
- `stop_on_blocked=true`
- `stop_on_waiting=true`
- `stop_on_idle=false`
- `stop_on_workflow_running=true`
- `require_confirmation_for_workflows=true`

Stop reasons:

- `max_steps`
- `waiting`
- `blocked`
- `idle`
- `parent_message`
- `workflow_running`
- `workflow_limit`
- `runtime_limit`
- `disallowed_action`
- `confirmation_required`

Commands:

```text
/agent run <ag-id> --steps 5
```

```bash
python -m mr1.workflow_cli agent-run <ag-id> --steps 5
python -m mr1.workflow_cli agent-run <ag-id> --max-workflows 1 --no-confirm-workflows
```

Run logs are written under:

```text
mr1/memory/agents/<agent_id>/logs/runs/<run_id>.json
mr1/memory/agents/<agent_id>/logs/runs.jsonl
```

Workflow validation and scoped ownership rules are unchanged. A bounded run can request workflow creation, but it cannot bypass workflow confirmation policy, runtime validation, or agent scope.

## Persistent Agent Messaging

MR1 now persists local durable agent messages under:

```text
mr1/memory/messages/<message_id>.json
```

- Messages are local coordination records, not external notifications.
- MRn agents do not contact the user directly.
- MRn agents report upward through MR1/root, and MR1 decides what reaches the user.
- Root can read all messages and send to any persistent agent.
- MRn agents can read only their own inbox/outbox and can send only to their parent or owned descendants.
- Archived messages remain on disk and are hidden from inbox/outbox listings unless explicitly requested.

Inspection and control commands:

```text
/inbox
/inbox --archived
/inbox triage --max-actions 2 --max-messages 5
/outbox
/message <msg-id>
/message read <msg-id>
/message archive <msg-id>
/message send <ag-id> <subject> <body-file>
```

And via the deterministic CLI:

```bash
python -m mr1.workflow_cli inbox
python -m mr1.workflow_cli inbox --archived
python -m mr1.workflow_cli inbox-triage --max-actions 2 --max-messages 5
python -m mr1.workflow_cli outbox
python -m mr1.workflow_cli message <message_id>
python -m mr1.workflow_cli message-read <message_id>
python -m mr1.workflow_cli message-archive <message_id>
python -m mr1.workflow_cli message-send <ag-id> "subject" path/to/body.txt
```

## Inbox Triage

Inbox triage is a single bounded coordination pass over unread root inbox messages. MR1 reads a capped slice of unread messages, produces a JSON-only triage plan, executes a small number of safe local actions, then stops.

- There is no daemon, background worker, or autonomous loop.
- Triage can summarize messages, mark or archive them, reply locally, advance scoped MRn agents, assign missions, or prepare a workflow.
- Workflow creation keeps the existing confirmation path: if the compiler requires confirmation, MR1 stores a pending workflow draft instead of submitting immediately.
- Triage never sends email, SMS, or any other external message.

Commands:

```text
/inbox triage
/inbox triage --max-actions 2 --max-messages 5
```

```bash
python -m mr1.workflow_cli inbox-triage
python -m mr1.workflow_cli inbox-triage --max-actions 2 --max-messages 5
```

The result prints a short summary, the actions taken, and bounded counts for reads, archives, replies, agent advances, and created workflows.

## Agent Runtime

Agent profiles describe the runtime contract for an agent: config schema, CLI binary, invocation shape, supported JSON output, and example workflow usage.

Agent runs are actual executions of that profile. They are runtime-managed workers with validated config, controlled invocation, health checks, structured output parsing, and classified failures.

The distinction from the other capability types is:

- tools are deterministic functions
- watchers are event gates
- agents are runtime workers

You can inspect the registered agent profiles directly:

```text
/agent kazi
/agent kazi --json
/agent kazi health
```

And through the deterministic CLI:

```bash
python -m mr1.workflow_cli agent kazi
python -m mr1.workflow_cli agent kazi --json
python -m mr1.workflow_cli agent kazi health
```

`/agent kazi health` validates the binary path, version response, runtime config, dispatcher-approved flags, non-interactive prompt execution, auth state, and JSON envelope parsing.

## WorkflowCompiler Agentic Tool

MR1 and MRn own workflow intent. The `workflow_compiler` agentic tool owns workflow spec construction.

- MR1/MRn validate the compiler's natural-language preview rather than reasoning over raw workflow JSON by default.
- The compiler returns a structured envelope containing `preview`, `spec`, `assumptions`, `risks`, `needs_confirmation`, and `confidence`.
- Runtime validation remains the authority on exact workflow JSON correctness.
- Workflow submission still flows through the normal scheduler/authoring path, which stamps `owner_agent_id` and enforces scoped ownership.
- `show json` is still available when the caller explicitly wants the raw workflow spec.

Tool tasks write the same normalized `output.json` schema as agent and watcher tasks, so downstream references work without new syntax:

```text
read_notes.result.text
shell.result.data.exit_code
shell.result.data.stdout
shell.artifact.stdout
write_file.artifact.written_file
```

You can inspect those contracts directly:

```text
/capabilities
/capability shell_command --json
/schema
/schema inputs --json
/tools
/tool shell_command --example
/agents
/agent kazi
/agent kazi health
```

Example tool to agent handoff:

```json
{
  "title": "Tool read demo",
  "tasks": [
    {
      "label": "read_notes",
      "title": "Read notes",
      "task_kind": "tool",
      "tool_type": "read_file",
      "tool_config": {
        "path": "notes.txt"
      }
    },
    {
      "label": "summarize",
      "title": "Summarize notes",
      "task_kind": "agent",
      "agent_type": "kazi",
      "depends_on": ["read_notes"],
      "inputs": [
        {"name": "notes", "from": "read_notes.result.text"}
      ],
      "prompt": "Summarize these notes."
    }
  ]
}
```

Deterministic inspection surfaces:

```bash
python -m mr1.workflow_cli tools
python -m mr1.workflow_cli task <task_id>
python -m mr1.workflow_cli result <task_id>
```

Inside MR1 you can use:

```text
/tools
/task <task_id>
/result <task_id>
```

Examples are available at:

```text
examples/workflows/tool_read_demo.json
examples/workflows/tool_shell_demo.json
```

Manual smoke test:

1. Create `/tmp/mr1_notes.txt`.
2. Submit `examples/workflows/tool_read_demo.json` with its `path` updated to that file.
3. Run `/scheduler tick`.
4. Confirm the read task wrote `output.json`.
5. Run `/result <read_task_id>` and inspect `result.text`.
6. Run `/task <downstream_task_id>` and inspect `materialized_prompt.txt`.

Shell smoke test:

1. Submit `examples/workflows/tool_shell_demo.json`.
2. Run `/scheduler tick`.
3. Run `/result <shell_task_id>`.
4. Confirm `result.data.exit_code`, `result.data.stdout`, and the `stdout` artifact are present.

## Running mem_dltr manually

Distils old decisions and completed tasks out of active memory into `memory/dumps/` and the RAG store:

```python
python -c "from mr1.mini.mem_dltr import distill; distill()"
```

## Project layout

```
mr1/
├── main.py                  entry point
├── mr1/
│   ├── mr1.py               persistent orchestrator
│   ├── kami.py              senior autonomous agent
│   ├── kazi.py              ephemeral task worker
│   ├── core/
│   │   ├── dispatcher.py    permission gate (no LLM)
│   │   ├── spawner.py       subprocess lifecycle manager
│   │   └── logger.py        structured JSONL logging
│   ├── mini/
│   │   ├── mem_dltr.py      memory distillation
│   │   ├── mem_rtvr.py      memory retrieval (chromadb)
│   │   ├── ctx_pkgr.py      context packager
│   │   └── com_smrzr.py     communication summariser
│   ├── agents/              YAML agent definitions
│   ├── memory/              active state, dumps, RAG store
│   ├── tasks/               per-task logs and comms
│   └── permissions/
│       └── allowlist.yml    what each agent may do
└── tests/                   pytest test suite
```
