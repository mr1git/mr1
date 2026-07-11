"""Slash-command (`/`-prefixed) dispatch for the MR1 root orchestrator.

Originally defined as `_handle_*_builtin` and `_submit_workflow_from_path`
methods on the `MR1` class in `mr1.orchestrator.root`. Extracted to keep
`root.py` focused on lifecycle/state/routing. `MR1` retains thin wrapper
methods that delegate here, so existing `@patch("mr1.mr1.MR1._handle_*")`
and `/`-command dispatch from `step()` continue to work unchanged.
"""

import json
import shlex
import threading
from pathlib import Path
from typing import Optional

from mr1.scoped_agents import AgentScopeError, is_agent_live
from mr1.inbox_triage import InboxTriagePolicy
from mr1.mrn_run import MRnRunPolicy, MRnRunRunner
from mr1.mrn_loop import MRnStepRunner
from mr1.workflow_models import Provenance
from mr1.scheduler import WatcherTriggerError, WorkflowSpecError
from mr1 import workflow_cli

_KNOWN_COMMAND_HINT = (
    "Known commands: /status, /tasks, /kill, /stop, /history, /memdltr, "
    "/workflows, /watchers, /workflow, /task, /result, /inputs, /artifacts, "
    "/jobs, /events, /scheduler tick, /agents, /agent, /inbox, /outbox, "
    "/message, /approvals, /tools, /capabilities, /schema, /help."
)

_WORKFLOW_USAGE_BY_SUBCOMMAND = {
    "append": "Usage: /workflow append <workflow_id> <path>",
    "cancel": "Usage: /workflow cancel <workflow_id>",
    "insert": "Usage: /workflow insert <workflow_id> <after_task> <path>",
    "replace": "Usage: /workflow replace [-r] <workflow_id> <task> <path>",
    "rerun": "Usage: /workflow rerun <workflow_id> <task>",
    "submit": "Usage: /workflow submit <path>",
    "trigger": "Usage: /workflow trigger <workflow_id> <label-or-task-id> [event_name]",
}


def _workflow_usage(subcommand: Optional[str] = None) -> str:
    if subcommand:
        return _WORKFLOW_USAGE_BY_SUBCOMMAND[subcommand]
    return "\n".join([
        "Usage: /workflow <workflow_id>",
        _WORKFLOW_USAGE_BY_SUBCOMMAND["submit"],
        _WORKFLOW_USAGE_BY_SUBCOMMAND["rerun"],
        _WORKFLOW_USAGE_BY_SUBCOMMAND["cancel"],
        _WORKFLOW_USAGE_BY_SUBCOMMAND["append"],
        _WORKFLOW_USAGE_BY_SUBCOMMAND["insert"],
        _WORKFLOW_USAGE_BY_SUBCOMMAND["replace"],
        _WORKFLOW_USAGE_BY_SUBCOMMAND["trigger"],
    ])


def handle_builtin(root, cmd: str) -> Optional[str]:
    """
    Handle slash commands locally. Returns the output string,
    or None if the input is not a built-in command.
    """
    cmd = cmd.strip()
    if cmd == "/status":
        return root._state.format_status()
    if cmd == "/tasks":
        return root._state.format_tasks()
    if cmd == "/kill":
        running_synthetic = root._running_test_agent_count()
        killed = root._spawner.kill_all("user_kill")
        root.kill_test_agents()
        synthetic_killed = running_synthetic
        # Mark all running tasks as killed.
        for tid in list(root._state.active_tasks):
            root._state.complete_task(tid, "killed")
        total = killed + synthetic_killed
        if total == 0:
            return (
                "No running spawned processes/test agents to kill. "
                "Persistent agents are not affected. "
                "Use /agent kill <ag-id> to terminate a persistent agent."
            )
        return (
            f"Killed {total} spawned process/test agent(s). "
            "Persistent MRn agents are not affected. "
            "Use /agent kill <ag-id> to terminate a persistent agent."
        )
    if cmd == "/history":
        recent = root._state._state["decisions"][-10:]
        if not recent:
            return "No recent decisions."
        lines = []
        for d in recent:
            lines.append(
                f"  {d['timestamp'][:19]}  {d['action']}  "
                f"{d.get('input_summary', '')[:60]}"
            )
        return "\n".join(lines)
    if cmd == "/memdltr":
        return root.trigger_memdltr()
    if cmd.startswith("/test spawn agents"):
        parts = cmd.split()
        if len(parts) != 4:
            return "Usage: /test spawn agents <height>"
        try:
            height = int(parts[3])
        except ValueError:
            return "Usage: /test spawn agents <height>"
        return root.spawn_test_agents(height)
    if cmd == "/test kill agents":
        return root.kill_test_agents()
    if cmd in ("/vizualize", "/visualize"):
        return root.launch_visualizer()
    if cmd in ("/visualize-web", "/vizualize-web"):
        return root.launch_web_visualizer()
    if cmd == "/workflows":
        return workflow_cli._format_workflows_table(
            root._scheduler.list_workflows()
        )
    if cmd == "/watchers":
        return workflow_cli._format_watchers(root._scheduler.list_workflows())
    if cmd == "/agents" or cmd.startswith("/agents "):
        return root._handle_agent_builtin(cmd)
    if cmd.startswith("/agent"):
        return root._handle_agent_builtin(cmd)
    if cmd == "/inbox" or cmd.startswith("/inbox "):
        return root._handle_message_builtin(cmd)
    if cmd == "/outbox" or cmd.startswith("/outbox "):
        return root._handle_message_builtin(cmd)
    if cmd.startswith("/message"):
        return root._handle_message_builtin(cmd)
    if cmd == "/approvals" or cmd.startswith("/approvals "):
        return root._handle_approval_builtin(cmd)
    if cmd == "/tools" or cmd.startswith("/tools "):
        return root._handle_capability_builtin(cmd)
    if cmd == "/capabilities" or cmd.startswith("/capabilities "):
        return root._handle_capability_builtin(cmd)
    if cmd.startswith("/capability"):
        return root._handle_capability_builtin(cmd)
    if cmd.startswith("/tool"):
        return root._handle_capability_builtin(cmd)
    if cmd == "/schema" or cmd.startswith("/schema "):
        return root._handle_schema_builtin(cmd)
    if cmd == "/workflow" or cmd.startswith("/workflow "):
        rest = cmd[len("/workflow"):].strip()
        if not rest:
            return _workflow_usage()
        if rest == "submit":
            return _workflow_usage("submit")
        if rest.startswith("submit "):
            path_str = rest[len("submit "):].strip()
            return root._submit_workflow_from_path(path_str)
        if rest == "rerun":
            return _workflow_usage("rerun")
        if rest.startswith("rerun "):
            parts = rest.split(maxsplit=2)
            if len(parts) != 3:
                return _workflow_usage("rerun")
            try:
                task_id = root._scheduler.rerun_task(parts[1], parts[2])
            except WorkflowSpecError as exc:
                return str(exc)
            root._scheduler.tick()
            return f"rerun scheduled: {task_id}"
        if rest == "cancel":
            return _workflow_usage("cancel")
        if rest.startswith("cancel "):
            parts = rest.split(maxsplit=1)
            if len(parts) != 2:
                return _workflow_usage("cancel")
            try:
                cancelled = root._scheduler.cancel_workflow(parts[1])
            except WorkflowSpecError as exc:
                return str(exc)
            root._scheduler.tick()
            return (
                f"workflow cancelled: {parts[1]}"
                if cancelled else f"workflow not found: {parts[1]}"
            )
        if rest == "append":
            return _workflow_usage("append")
        if rest.startswith("append "):
            parts = rest.split(maxsplit=2)
            if len(parts) != 3:
                return _workflow_usage("append")
            spec, error = workflow_cli._load_json_file(parts[2])
            if error:
                return error
            try:
                workflow_id = root._scheduler.append_workflow(parts[1], spec)
            except WorkflowSpecError as exc:
                return str(exc)
            root._scheduler.tick()
            return f"workflow updated: {workflow_id}"
        if rest == "insert":
            return _workflow_usage("insert")
        if rest.startswith("insert "):
            parts = rest.split(maxsplit=3)
            if len(parts) != 4:
                return _workflow_usage("insert")
            spec, error = workflow_cli._load_json_file(parts[3])
            if error:
                return error
            try:
                workflow_id = root._scheduler.insert_workflow(parts[1], parts[2], spec)
            except WorkflowSpecError as exc:
                return str(exc)
            root._scheduler.tick()
            return f"workflow updated: {workflow_id}"
        if rest == "replace" or rest == "replace -r":
            return _workflow_usage("replace")
        if rest.startswith("replace "):
            try:
                parts = shlex.split(rest)
            except ValueError:
                return _workflow_usage("replace")
            rerun_after_replace = False
            if len(parts) > 1 and parts[1] == "-r":
                rerun_after_replace = True
                parts = [parts[0], *parts[2:]]
            if len(parts) != 4:
                return _workflow_usage("replace")
            spec, error = workflow_cli._load_json_file(parts[3])
            if error:
                return error
            try:
                workflow_id = root._scheduler.replace_workflow(parts[1], parts[2], spec)
            except WorkflowSpecError as exc:
                return str(exc)
            if rerun_after_replace:
                root._scheduler.tick()
                return f"workflow updated and rerun: {workflow_id}"
            return f"workflow updated: {workflow_id}"
        if rest == "trigger":
            return _workflow_usage("trigger")
        if rest.startswith("trigger "):
            parts = rest.split(maxsplit=3)
            if len(parts) < 3:
                return _workflow_usage("trigger")
            wf_id = parts[1]
            label_or_task_id = parts[2]
            event_name = parts[3] if len(parts) > 3 else None
            try:
                task_id = root._scheduler.trigger_watcher(
                    wf_id,
                    label_or_task_id,
                    event_name=event_name,
                )
            except WatcherTriggerError as exc:
                return str(exc)
            root._scheduler.tick()
            return f"triggered watcher: {task_id}"
        wf_id = rest
        wf = root._scheduler.get_workflow(wf_id)
        if wf is None:
            return f"workflow not found: {wf_id}"
        return workflow_cli._format_workflow_detail(wf)
    if cmd.startswith("/task "):
        rest = cmd[len("/task "):].strip()
        if rest.startswith("cancel "):
            task_id = rest[len("cancel "):].strip()
            if not task_id:
                return "Usage: /task cancel <task_id>"
            try:
                cancelled = root._scheduler.cancel_task(task_id)
            except WorkflowSpecError as exc:
                return str(exc)
            root._scheduler.tick()
            return f"task cancelled: {cancelled}"
        task_id = rest
        wf, task = workflow_cli._find_workflow_for_task(
            root._workflow_store, task_id
        )
        if wf is None or task is None:
            return f"task not found: {task_id}"
        return workflow_cli._format_task_detail(wf, task)
    if cmd.startswith("/result "):
        task_id = cmd[len("/result "):].strip()
        wf, task = workflow_cli._find_workflow_for_task(
            root._workflow_store, task_id
        )
        if wf is None or task is None:
            return f"task not found: {task_id}"
        output = root._workflow_store.load_task_output(wf.workflow_id, task.task_id)
        return workflow_cli._format_result(task, output)
    if cmd.startswith("/inputs "):
        task_id = cmd[len("/inputs "):].strip()
        wf, task = workflow_cli._find_workflow_for_task(
            root._workflow_store, task_id
        )
        if wf is None or task is None:
            return f"task not found: {task_id}"
        inputs = root._workflow_store.load_task_inputs(wf.workflow_id, task.task_id)
        return workflow_cli._format_inputs(task, inputs)
    if cmd.startswith("/artifacts "):
        wf_id = cmd[len("/artifacts "):].strip()
        wf = root._scheduler.get_workflow(wf_id)
        if wf is None:
            return f"workflow not found: {wf_id}"
        return workflow_cli._format_artifacts(wf)
    if cmd == "/jobs":
        return workflow_cli._format_jobs(root._scheduler.list_workflows())
    if cmd.startswith("/events "):
        wf_id = cmd[len("/events "):].strip()
        if root._scheduler.get_workflow(wf_id) is None:
            return f"workflow not found: {wf_id}"
        events = root._workflow_store.load_events(wf_id, limit=50)
        return workflow_cli._format_events(events)
    if cmd == "/scheduler tick":
        root._scheduler.tick()
        return "scheduler ticked."
    if cmd == "/help":
        return (
            "Commands: /status  /tasks  /kill  /stop  /history  /memdltr  "
            "/workflows  /watchers  /capabilities  /capability <name>  "
            "/tools  /tool <type>  /agents  /agent <ag-id>  /agent create <title>  "
            "/agent kill-all [all|<title>] [--exclude <agent-id-or-title>]...  "
            "/agent step <ag-id>  /agent run <ag-id> --steps N  "
            "/inbox  /outbox  /message <id>  /approvals  /schema  "
            "/vizualize  /visualize-web  /clear  /help  /exit  "
            "/test spawn agents <h>  /test kill agents"
        )
    if cmd == "/clear":
        print("\033[2J\033[H", end="", flush=True)
        return ""
    if cmd == "/stop":
        running_synthetic = root._running_test_agent_count()
        killed = root._spawner.kill_all("user_stop")
        root.kill_test_agents()
        synthetic_killed = running_synthetic
        for tid in list(root._state.active_tasks):
            root._state.complete_task(tid, "killed")
        total = killed + synthetic_killed
        if total == 0:
            return (
                "No running spawned processes/test agents to stop. "
                "Persistent agents are not affected. "
                "Use /agent kill <ag-id> to terminate a persistent agent."
            )
        return (
            f"Stopped {total} spawned process/test agent(s). "
            "Persistent MRn agents are not affected."
        )
    if cmd.startswith("/"):
        command = cmd.split(maxsplit=1)[0]
        return f"Unknown slash command: {command}. {_KNOWN_COMMAND_HINT}"
    return None

def handle_capability_builtin(root, cmd: str) -> str:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        if cmd.startswith("/capability"):
            return "usage: /capability <name> [--json] [--example] [--brief]"
        if cmd.startswith("/tool"):
            return "usage: /tool <tool_type> [--json] [--example] [--brief]"
        if cmd.startswith("/capabilities"):
            return "usage: /capabilities [--json] [--brief]"
        return "usage: /tools [--json] [--brief]"

    command = parts[0]
    flags = {part for part in parts[1:] if part.startswith("--")}
    positionals = [part for part in parts[1:] if not part.startswith("--")]
    allowed_flags = {"--json", "--brief"}
    if command in {"/capability", "/tool"}:
        allowed_flags.add("--example")
    if any(flag not in allowed_flags for flag in flags):
        if command == "/capability":
            return "usage: /capability <name> [--json] [--example] [--brief]"
        if command == "/tool":
            return "usage: /tool <tool_type> [--json] [--example] [--brief]"
        if command == "/capabilities":
            return "usage: /capabilities [--json] [--brief]"
        return "usage: /tools [--json] [--brief]"
    if "--example" in flags and "--brief" in flags:
        return "invalid flag combination"

    if command == "/capabilities":
        if positionals:
            return "usage: /capabilities [--json] [--brief]"
        return workflow_cli._format_capabilities(
            json_output="--json" in flags,
            brief="--brief" in flags,
        )
    if command == "/capability":
        if positionals and positionals[0] == "call":
            if len(positionals) != 3 or flags:
                return "usage: /capability call <name> <config-json>"
            capability_name = positionals[1]
            config_json = positionals[2]
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                return "error: invalid JSON config"
            if not isinstance(config, dict):
                return "error: config must be a JSON object"
            from mr1.capability_runner import CapabilityRunner
            runner = CapabilityRunner(
                scoped_agent_store=root._scoped_agents,
                message_store=root._message_store,
                workspace_root=root._workflow_store.root.parent,
            )
            try:
                result = runner.run_capability(capability_name, config, root._root_agent_id)
            except ValueError as exc:
                return f"error: {exc}"
            lines = [
                f"capability:   {result.capability}",
                f"status:       {result.status}",
                f"duration_ms:  {result.duration_ms}",
                "output:",
                json.dumps(result.output, indent=2, sort_keys=True),
            ]
            if result.error is not None:
                lines.append(f"error:        {result.error}")
            return "\n".join(lines)
        if len(positionals) != 1:
            return "usage: /capability <name> [--json] [--example] [--brief]"
        try:
            return workflow_cli._format_capability(
                positionals[0],
                json_output="--json" in flags,
                example_only="--example" in flags,
                brief="--brief" in flags,
            )
        except ValueError:
            return f"capability not found: {positionals[0]}"
    if command == "/tools":
        if positionals:
            return "usage: /tools [--json] [--brief]"
        return workflow_cli._format_tools(
            json_output="--json" in flags,
            brief="--brief" in flags,
        )
    if len(positionals) != 1:
        return "usage: /tool <tool_type> [--json] [--example] [--brief]"
    try:
        return workflow_cli._format_tool(
            positionals[0],
            json_output="--json" in flags,
            example_only="--example" in flags,
            brief="--brief" in flags,
        )
    except ValueError:
        return f"tool not found: {positionals[0]}"

def handle_agent_builtin(root, cmd: str) -> str:
    usage = (
        "usage: /agent <create <title>|kill <ag-id>|kill-all [all|<title>] [--exclude <agent-id-or-title>]...|assign <ag-id> <mission-file>|"
        "step <ag-id>|run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]|"
        "<ag-id>|kazi [health]> [--json] [--brief]"
    )
    try:
        parts = shlex.split(cmd)
    except ValueError:
        if cmd.startswith("/agent"):
            return usage
        return "usage: /agents [--json] [--brief]"

    command = parts[0]
    if command == "/agent" and len(parts) > 1 and parts[1] == "run":
        return root._handle_agent_run_builtin(parts[2:], usage)
    if command == "/agent" and len(parts) > 1 and parts[1] == "kill-all":
        return root._handle_agent_kill_all_builtin(parts[2:], usage)
    flags = {part for part in parts[1:] if part.startswith("--")}
    positionals = [part for part in parts[1:] if not part.startswith("--")]
    allowed_flags = {"--json", "--brief"}
    if any(flag not in allowed_flags for flag in flags):
        if command == "/agent":
            return usage
        return "usage: /agents [--json] [--brief]"

    if command == "/agents":
        if positionals:
            return "usage: /agents [--json] [--brief]"
        return workflow_cli._format_agents(
            root._scoped_agents.list_visible_agents(root._root_agent_id),
            json_output="--json" in flags,
            brief="--brief" in flags,
        )

    if not positionals:
        return usage
    if positionals[0] == "create":
        title = " ".join(positionals[1:]).strip()
        if not title:
            return "usage: /agent create <title>"
        try:
            agent = root._scoped_agents.create_child_agent(root._root_agent_id, title)
        except ValueError as exc:
            return str(exc)
        return agent.agent_id
    if positionals[0] == "kill":
        if len(positionals) != 2:
            return "usage: /agent kill <ag-id>"
        try:
            agent = root._scoped_agents.terminate_agent(root._root_agent_id, positionals[1])
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        return agent.agent_id
    if positionals[0] == "assign":
        if len(positionals) != 3:
            return "usage: /agent assign <ag-id> <mission-file>"
        mission_path = Path(positionals[2])
        if not mission_path.exists():
            return f"error: mission file not found: {mission_path}"
        try:
            mission = mission_path.read_text(encoding="utf-8")
        except OSError:
            return f"error: mission file not found: {mission_path}"
        try:
            agent = root._scoped_agents.assign_mission(
                root._root_agent_id,
                positionals[1],
                mission,
            )
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        return agent.agent_id
    if positionals[0] == "step":
        if len(positionals) != 2:
            return "usage: /agent step <ag-id>"
        runner = MRnStepRunner(
            workflow_store=root._workflow_store,
            scoped_agent_store=root._scoped_agents,
            message_store=root._message_store,
            workflow_authoring_service=root._workflow_authoring,
        )
        try:
            result = runner.step(positionals[1], caller_agent_id=root._root_agent_id)
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        return workflow_cli._format_mrn_step_result(result)

    agent_name = positionals[0]
    action = positionals[1] if len(positionals) > 1 else None
    if agent_name.startswith("ag-"):
        if len(positionals) != 1:
            return "usage: /agent <ag-id>"
        try:
            agent = root._runtime_access.read_agent(
                agent_name,
                caller_agent_id=root._root_agent_id,
            )
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        return workflow_cli._format_agent(
            agent,
            json_output="--json" in flags,
            brief="--brief" in flags,
        )

    if len(positionals) > 2 or (action is not None and action != "health"):
        return usage
    try:
        if action == "health":
            return workflow_cli._format_runtime_agent_health(
                agent_name,
                json_output="--json" in flags,
            )
        return workflow_cli._format_runtime_agent(
            agent_name,
            json_output="--json" in flags,
            brief="--brief" in flags,
        )
    except ValueError:
        return f"agent not found: {agent_name}"


def handle_agent_kill_all_builtin(root, parts: list[str], usage: str) -> str:
    del usage
    selector = "all"
    exclusions: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if token == "--exclude":
            if index + 1 >= len(parts):
                return "usage: /agent kill-all [all|<title>] [--exclude <agent-id-or-title>]..."
            exclusions.append(parts[index + 1])
            index += 2
            continue
        if selector != "all":
            return "usage: /agent kill-all [all|<title>] [--exclude <agent-id-or-title>]..."
        selector = token
        index += 1

    live_agents = [
        agent
        for agent in root._scoped_agents.list_visible_agents(root._root_agent_id)
        if agent.agent_id != root._root_agent_id and is_agent_live(agent)
    ]
    if selector != "all":
        normalized_selector = selector.strip().lower()
        live_agents = [
            agent
            for agent in live_agents
            if agent.agent_id == selector or agent.title.strip().lower() == normalized_selector
        ]
    if not live_agents:
        return "No live agents matched the requested kill-all scope."

    normalized_exclusions = {item.strip().lower() for item in exclusions if item.strip()}
    excluded = [
        agent
        for agent in live_agents
        if agent.agent_id.lower() in normalized_exclusions
        or agent.title.strip().lower() in normalized_exclusions
    ]
    targets = [agent for agent in live_agents if agent not in excluded]

    terminated: list[str] = []
    errors: list[str] = []
    for agent in targets:
        try:
            root._scoped_agents.terminate_agent(root._root_agent_id, agent.agent_id)
            terminated.append(agent.agent_id)
        except (ValueError, AgentScopeError) as exc:
            errors.append(f"{agent.agent_id}: {exc}")

    parts_out: list[str] = []
    if terminated:
        parts_out.append(f"Terminated {len(terminated)} agent(s): {', '.join(terminated)}.")
    if excluded:
        excluded_text = ", ".join(f"{agent.title} ({agent.agent_id})" for agent in excluded)
        parts_out.append(f"Excluded {len(excluded)} agent(s): {excluded_text}.")
    unmatched_exclusions = [
        item
        for item in exclusions
        if item.strip()
        and item.strip().lower() not in {
            *[agent.agent_id.lower() for agent in live_agents],
            *[agent.title.strip().lower() for agent in live_agents],
        }
    ]
    if unmatched_exclusions:
        parts_out.append(f"Unmatched exclusions: {', '.join(unmatched_exclusions)}.")
    if errors:
        parts_out.append(f"Errors: {'; '.join(errors)}")
    return " ".join(parts_out) if parts_out else "No agents terminated."

def handle_agent_run_builtin(root, parts: list[str], usage: str) -> str:
    if not parts:
        return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
    agent_id = parts[0]
    if agent_id.startswith("--"):
        return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
    steps = 3
    max_workflows = 2
    require_confirmation_for_workflows = True
    index = 1
    while index < len(parts):
        token = parts[index]
        if token == "--steps":
            if index + 1 >= len(parts):
                return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
            try:
                steps = int(parts[index + 1])
            except ValueError:
                return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
            index += 2
            continue
        if token == "--max-workflows":
            if index + 1 >= len(parts):
                return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
            try:
                max_workflows = int(parts[index + 1])
            except ValueError:
                return "usage: /agent run <ag-id> [--steps N] [--max-workflows N] [--no-confirm-workflows]"
            index += 2
            continue
        if token == "--no-confirm-workflows":
            require_confirmation_for_workflows = False
            index += 1
            continue
        return usage

    runner = MRnRunRunner(
        workflow_store=root._workflow_store,
        scoped_agent_store=root._scoped_agents,
        message_store=root._message_store,
    )
    policy = MRnRunPolicy(
        max_steps=steps,
        max_workflows_created=max_workflows,
        require_confirmation_for_workflows=require_confirmation_for_workflows,
    )
    try:
        result = runner.run(agent_id, policy, caller_agent_id=root._root_agent_id)
    except (ValueError, AgentScopeError) as exc:
        return str(exc)
    return workflow_cli._format_mrn_run_result(result)

def handle_message_builtin(root, cmd: str) -> str:
    inbox_usage = "usage: /inbox [--archived] | /inbox triage [--max-actions N] [--max-messages N] | /inbox triage on | /inbox triage off | /inbox triage status"
    try:
        parts = shlex.split(cmd)
    except ValueError:
        if cmd.startswith("/message"):
            return "usage: /message <message_id> | /message read <message_id> | /message archive <message_id> | /message send <agent_id> <subject> <body-file>"
        if cmd.startswith("/outbox"):
            return "usage: /outbox"
        return inbox_usage

    command = parts[0]
    flags = {part for part in parts[1:] if part.startswith("--")}
    positionals = [part for part in parts[1:] if not part.startswith("--")]
    if command == "/inbox":
        if positionals and positionals[0] == "triage" and len(positionals) == 2:
            sub = positionals[1]
            if sub == "on":
                if root._inbox_thread and root._inbox_thread.is_alive():
                    return "inbox auto-triage already running"
                root._inbox_stop.clear()
                root._inbox_thread = threading.Thread(
                    target=root._run_inbox_loop,
                    name="inbox-triage",
                    daemon=True,
                )
                root._inbox_thread.start()
                return "inbox auto-triage started"
            if sub == "off":
                root._inbox_stop.set()
                if root._inbox_thread:
                    root._inbox_thread.join(timeout=2.0)
                    root._inbox_thread = None
                return "inbox auto-triage stopped"
            if sub == "status":
                running = bool(root._inbox_thread and root._inbox_thread.is_alive())
                return f"inbox auto-triage: {'running' if running else 'stopped'} (interval: {root._inbox_triage_interval_s}s)"
            return inbox_usage
        if positionals and positionals[0] == "triage":
            if "--archived" in flags:
                return inbox_usage
            max_messages = 10
            max_actions = 3
            index = 1
            while index < len(parts):
                token = parts[index]
                if token == "triage":
                    index += 1
                    continue
                if token == "--max-actions":
                    if index + 1 >= len(parts):
                        return inbox_usage
                    try:
                        max_actions = int(parts[index + 1])
                    except ValueError:
                        return inbox_usage
                    index += 2
                    continue
                if token == "--max-messages":
                    if index + 1 >= len(parts):
                        return inbox_usage
                    try:
                        max_messages = int(parts[index + 1])
                    except ValueError:
                        return inbox_usage
                    index += 2
                    continue
                return inbox_usage
            try:
                runner = root._make_inbox_triage_runner()
                result = runner.run(
                    InboxTriagePolicy(
                        max_messages=max_messages,
                        max_actions=max_actions,
                    ),
                    caller_agent_id=root._root_agent_id,
                )
            except ValueError as exc:
                return f"error: {exc}"
            return workflow_cli._format_inbox_triage_result(result)
        if flags - {"--archived"} or positionals:
            return inbox_usage
        return workflow_cli._format_messages_table(
            root._message_store.list_inbox(
                root._root_agent_id,
                include_archived="--archived" in flags,
            ),
            mode="inbox",
        )
    if command == "/outbox":
        if flags:
            return "usage: /outbox"
        if positionals:
            return "usage: /outbox"
        return workflow_cli._format_messages_table(
            root._message_store.list_outbox(root._root_agent_id),
            mode="outbox",
        )

    if flags:
        return "usage: /message <message_id> | /message read <message_id> | /message archive <message_id> | /message send <agent_id> <subject> <body-file>"
    if not positionals:
        return "usage: /message <message_id> | /message read <message_id> | /message archive <message_id> | /message send <agent_id> <subject> <body-file>"
    if positionals[0] == "read":
        if len(positionals) != 2:
            return "usage: /message read <message_id>"
        try:
            workflow_cli._require_message(
                root._message_store,
                positionals[1],
                root._root_agent_id,
            )
            message = root._message_store.mark_read(positionals[1])
            if message is None:
                return f"message not found: {positionals[1]}"
            return message.message_id
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
    if positionals[0] == "archive":
        if len(positionals) != 2:
            return "usage: /message archive <message_id>"
        try:
            workflow_cli._require_message(
                root._message_store,
                positionals[1],
                root._root_agent_id,
            )
            message = root._message_store.archive_message(positionals[1])
            if message is None:
                return f"message not found: {positionals[1]}"
            return message.message_id
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
    if positionals[0] == "send":
        if len(positionals) != 4:
            return "usage: /message send <agent_id> <subject> <body-file>"
        body_path = Path(positionals[3])
        if not body_path.exists():
            return f"error: message body file not found: {body_path}"
        try:
            body = body_path.read_text(encoding="utf-8")
        except OSError:
            return f"error: message body file not found: {body_path}"
        if not root._message_store.can_agent_send_message(root._root_agent_id, positionals[1]):
            return "access denied: recipient not in agent scope"
        message = root._message_store.create_message(
            from_agent_id=root._root_agent_id,
            to_agent_id=positionals[1],
            kind="request",
            subject=positionals[2],
            body=body,
        )
        return message.message_id
    if len(positionals) != 1:
        return "usage: /message <message_id>"
    try:
        workflow_cli._require_message(
            root._message_store,
            positionals[0],
            root._root_agent_id,
        )
        message = root._runtime_access.read_message(
            positionals[0],
            caller_agent_id=root._root_agent_id,
        )
    except (ValueError, AgentScopeError) as exc:
        return str(exc)
    return workflow_cli._format_message_detail(message)

def handle_approval_builtin(root, cmd: str) -> str:
    usage = (
        "usage: /approvals list | /approvals show <approval_request_id> | "
        "/approvals approve <approval_request_id> [--grant-scope] [--reason TEXT] | "
        "/approvals deny <approval_request_id> [--reason TEXT]"
    )
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return usage
    if len(parts) == 1:
        approvals = workflow_cli._visible_approvals(
            root._approval_store,
            root._scoped_agents,
            root._root_agent_id,
        )
        return workflow_cli._format_approvals_table(approvals)
    if len(parts) < 2:
        return usage

    subcommand = parts[1]
    if subcommand == "list":
        if len(parts) != 2:
            return usage
        approvals = workflow_cli._visible_approvals(
            root._approval_store,
            root._scoped_agents,
            root._root_agent_id,
        )
        return workflow_cli._format_approvals_table(approvals)

    if subcommand == "show":
        if len(parts) != 3:
            return usage
        try:
            approval = workflow_cli._require_visible_approval(
                root._approval_store,
                parts[2],
                root._scoped_agents,
                root._root_agent_id,
            )
        except (ValueError, AgentScopeError) as exc:
            return str(exc)
        return workflow_cli._format_approval(approval)

    if subcommand not in {"approve", "deny"}:
        return usage
    if len(parts) < 3:
        return usage

    approval_request_id = parts[2]
    grant_scope = False
    reason = "approved" if subcommand == "approve" else "denied"
    index = 3
    while index < len(parts):
        token = parts[index]
        if token == "--grant-scope":
            if subcommand != "approve":
                return usage
            grant_scope = True
            index += 1
            continue
        if token == "--reason":
            if index + 1 >= len(parts):
                return usage
            reason = parts[index + 1]
            index += 2
            continue
        return usage

    try:
        updated = root._apply_approval_decision(
            approval_request_id,
            decision_text="approved" if subcommand == "approve" else "denied",
            reason=reason,
            grant_scope=grant_scope,
        )
    except (ValueError, AgentScopeError) as exc:
        return str(exc)
    return workflow_cli._format_approval_decision_result(
        updated,
        grant_scope=grant_scope,
    )

def handle_schema_builtin(root, cmd: str) -> str:
    usage = "usage: /schema [workflow|task|inputs|refs|task-kinds] [--json] [--brief]"
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return usage

    flags = {part for part in parts[1:] if part.startswith("--")}
    positionals = [part for part in parts[1:] if not part.startswith("--")]
    allowed_flags = {"--json", "--brief"}
    if any(flag not in allowed_flags for flag in flags):
        return usage
    if len(positionals) > 1:
        return usage
    try:
        return workflow_cli._format_schema(
            positionals[0] if positionals else None,
            json_output="--json" in flags,
            brief="--brief" in flags,
        )
    except ValueError as exc:
        return f"error: {exc}"

def submit_workflow_from_path(root, path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return f"spec file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    try:
        wf_id = root._scheduler.submit_workflow(
            spec,
            Provenance(type="agent", id="MR1"),
            caller_agent_id=root._root_agent_id,
            owner_agent_id=root._root_agent_id,
        )
    except WorkflowSpecError as exc:
        return f"invalid workflow: {exc}"
    return f"submitted: {wf_id}"
