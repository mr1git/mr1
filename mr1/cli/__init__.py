"""CLI package for `python -m mr1.workflow_cli`.

`mr1.workflow_cli` is the historical entry point and remains a thin
facade re-exporting everything from this package, so existing
`from mr1.workflow_cli import _format_X` calls in tests and orchestrator
modules keep working.

Module map:
    formatting.py   — small shared formatting/render helpers
    context.py      — workflow-store/visibility/audit helpers used by handlers
    messages.py     — message formatters + `_cmd_inbox/_cmd_message/...`
    agents.py       — agent formatters + `_cmd_agent/_cmd_agents/...`
    memory.py       — memory formatters + `_cmd_memory_*` + snapshot/doctor
    capabilities.py — capability/tool/approval formatters + handlers
    workflows.py    — workflow/task/result/inputs/artifacts/watchers handlers
    events.py       — events + timeline handlers
    main.py         — parser construction and `main()` entry point
"""
