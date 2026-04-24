# MR1 — Multi-Agent Orchestration System

MR1 is a persistent, terminal-based multi-agent system built on the Claude CLI. A top-level orchestrator (`MR1`) maintains a live conversation with the user and routes tasks to specialised sub-agents: `Kami` handles complex, multi-step work with broad tool access, while `Kazi` handles scoped one-shot jobs. A set of lightweight mini-agents manages memory, context packaging, and communication summarisation without ever invoking an LLM — all deterministic, no surprises.

## How to run

```
python main.py
```

`python main.py` now launches the Ink timeline/tree UI by default. That UI is the primary MR1 interface: you talk to MR1 in the same screen while the background task tree animates above the conversation.

There is also an experimental `TermUI` renderer with a simpler block-style MR1 anchor:

```bash
python main.py --termui
```

There is now also a browser-based path that avoids terminal rendering entirely:

```bash
python main.py --web
```

To use the legacy plain-text loop instead:

```bash
python main.py --plain
```

## Timeline UI

The Ink UI is no longer a detached observer. It is the main MR1 frontend.

It provides:

- `R` root MR1
- `O/Y/G/B/I/V` deeper or later tree positions
- a moving MR1 anchor with timeline-follow behavior
- a lower conversation lane and an upper system lane
- child agents that stay attached while alive, then freeze and dim when dead
- a built-in prompt composer so you can talk to MR1 directly inside the UI

Install the Node dependencies once from the project root:

```bash
npm install
```

Then either launch it directly:

```bash
npm run viz
```

To try the experimental `TermUI` prototype directly:

```bash
npm run viz:termui
```

For a non-interactive smoke render of the current timeline snapshot:

```bash
npm run viz:once
```

For a static `TermUI` smoke render against the current snapshot:

```bash
npm run viz:termui:once
```

For deterministic frontend checks:

```bash
npm run viz:test
```

And for the deterministic `TermUI` render checks:

```bash
npm run viz:termui:test
```

Inside the legacy plain loop, `/vizualize` and `/visualize` now act as handoff hints toward the primary Ink UI rather than launching a detached observer window.

For the browser-based visualizer inside the plain loop, use `/visualize-web`.

For synthetic workload generation in either the plain loop or the web prompt, use:

```text
/test spawn agents 3
/test kill agents
```

## Bridge

The Ink app talks to a Python bridge process instead of scraping terminal output. The bridge owns a persistent MR1 instance and emits structured JSON lines for:

- conversation turns
- task lifecycle events
- periodic timeline snapshots
- command results and errors

You can run it directly for debugging:

```bash
python -m mr1.ui_bridge
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
| `/vizualize` | Explain how to switch to the primary Ink UI |
| `/visualize-web` | Launch the browser-based web visualizer |
| `/test spawn agents <h>` | Spawn a synthetic full binary tree of worker processes |
| `/test kill agents` | Kill all synthetic worker processes |
| `exit`     | Save session state and quit                 |

## Phase 1: Workflows

Phase 1 adds a deterministic workflow scheduler that runs inside the MR1 process. Workflow control does not invoke MR1 reasoning: submission, discovery, scheduling, event logging, and inspection all go through the store, scheduler, and workflow CLI only.

Supported commands in the plain loop, UI bridge, and web UI:

| Command | Effect |
|---------|--------|
| `/workflows` | List all known workflows |
| `/workflow <id>` | Show one workflow and its tasks |
| `/workflow submit <path>` | Load a JSON spec from disk and submit it |
| `/workflow trigger <id> <label-or-task-id> [event_name]` | Trigger a manual watcher |
| `/task <id>` | Show one task's detail |
| `/result <task_id>` | Show one task's normalized output |
| `/inputs <task_id>` | Show one task's resolved workflow inputs |
| `/artifacts <workflow_id>` | List registered artifacts for a workflow |
| `/jobs` | List live workflow tasks |
| `/watchers` | List active watcher tasks |
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
python -m mr1.workflow_cli workflows
python -m mr1.workflow_cli workflow <workflow_id>
python -m mr1.workflow_cli result <task_id>
python -m mr1.workflow_cli inputs <task_id>
python -m mr1.workflow_cli artifacts <workflow_id>
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
