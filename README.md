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
