from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO

from mr1.event_log import SystemEvent
from mr1.mr1 import MR1, StateManager
from mr1.workflow_store import WorkflowStore


@dataclass(frozen=True)
class RuntimePaths:
    isolated: bool
    runtime_root: Path
    workflow_root: Path
    state_path: Path
    context_path: Path
    dumps_root: Path
    rag_root: Path


@dataclass(frozen=True)
class RuntimeSnapshot:
    agents: dict[str, dict[str, Any]]
    workflows: dict[str, dict[str, Any]]
    approvals: dict[str, dict[str, Any]]
    events: dict[str, dict[str, Any]]
    turn_artifacts: dict[str, dict[str, Any]]
    max_event_index: Optional[int]


@dataclass(frozen=True)
class ParsedJsonlInput:
    input_text: str
    request_id: Optional[str]


def _default_live_runtime_root() -> Path:
    return WorkflowStore().root.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mr1.runtime_test_cli",
        description="Drive the real MR1 runtime through a safe JSON test harness.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--isolated",
        action="store_true",
        help="Use an isolated runtime root. This is the default.",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Use the normal MR1 runtime roots instead of an isolated test root.",
    )
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Explicit runtime root for isolated mode.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Read line-delimited inputs from stdin and emit one JSON object per line.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="One-shot input to process.",
    )
    return parser


def _runtime_paths(args: argparse.Namespace) -> tuple[RuntimePaths, Optional[tempfile.TemporaryDirectory[str]]]:
    if args.live and args.runtime_root:
        raise ValueError("--runtime-root cannot be combined with --live")

    isolated = not args.live
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if isolated:
        if args.runtime_root:
            runtime_root = Path(args.runtime_root).expanduser().resolve()
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="mr1-runtime-test-")
            runtime_root = Path(temp_dir.name)
    else:
        runtime_root = _default_live_runtime_root().resolve()

    return (
        RuntimePaths(
            isolated=isolated,
            runtime_root=runtime_root,
            workflow_root=runtime_root / "workflows",
            state_path=runtime_root / "active" / "mr1_state.json",
            context_path=runtime_root / "active" / "mr1_context.md",
            dumps_root=runtime_root / "dumps",
            rag_root=runtime_root / "rag",
        ),
        temp_dir,
    )


@contextlib.contextmanager
def _patched_runtime_paths(paths: RuntimePaths):
    if not paths.isolated:
        yield
        return

    import mr1.orchestrator.memory as orchestrator_memory
    import mr1.orchestrator.root as orchestrator_root
    import mr1.mini.mem_dltr as mem_dltr
    import mr1.mini.mem_rtvr as mem_rtvr

    patches = [
        (orchestrator_root, "_CONTEXT_PATH", paths.context_path),
        (orchestrator_memory, "_CONTEXT_PATH", paths.context_path),
        (mem_dltr, "_ACTIVE_DIR", paths.state_path.parent),
        (mem_dltr, "_DUMPS_DIR", paths.dumps_root),
        (mem_dltr, "_RAG_DIR", paths.rag_root),
        (mem_rtvr, "_RAG_DIR", paths.rag_root),
        (mem_rtvr, "_DUMPS_DIR", paths.dumps_root),
        (mem_rtvr, "_CHROMA_DIR", paths.rag_root / ".chroma"),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    try:
        for module, name, value in patches:
            setattr(module, name, value)
        mem_rtvr._chroma_client = None
        mem_rtvr._collection = None
        yield
    finally:
        for module, name, value in originals:
            setattr(module, name, value)
        mem_rtvr._chroma_client = None
        mem_rtvr._collection = None


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def _map_by_id(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {
        str(item[key]): item
        for item in items
        if key in item
    }


def _turn_artifacts_dir(state_path: Path) -> Path:
    return state_path.parent / "mr1_turns"


def _load_turn_artifacts(state_path: Path) -> dict[str, dict[str, Any]]:
    artifacts_dir = _turn_artifacts_dir(state_path)
    if not artifacts_dir.exists():
        return {}
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(artifacts_dir.glob("*.json")):
        try:
            payloads[str(path)] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def _event_map(events: list[SystemEvent]) -> dict[str, dict[str, Any]]:
    return {
        event.event_id: event.to_dict()
        for event in events
    }


def _diff_entities(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    created = [after[item_id] for item_id in sorted(after.keys() - before.keys())]
    updated = [
        after[item_id]
        for item_id in sorted(after.keys() & before.keys())
        if before[item_id] != after[item_id]
    ]
    return created, updated


def _new_items(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    sort_key: str,
) -> list[dict[str, Any]]:
    items = [after[item_id] for item_id in after.keys() - before.keys()]
    return sorted(items, key=lambda item: (item.get(sort_key), json.dumps(item, sort_keys=True)))


class RuntimeTestSession:
    def __init__(self, paths: RuntimePaths) -> None:
        self._paths = paths
        self._mr1 = MR1(
            state_manager=StateManager(state_path=paths.state_path),
            workflow_store=WorkflowStore(root=paths.workflow_root),
            inbox_auto_triage=False,
        )
        self._started = False

    @property
    def paths(self) -> RuntimePaths:
        return self._paths

    def ensure_started(self) -> None:
        if self._started:
            return
        self._mr1.start()
        self._started = True

    def shutdown(self) -> None:
        self._mr1.shutdown("runtime_test_cli")

    def snapshot(self) -> RuntimeSnapshot:
        caller = self._mr1._root_agent_id
        runtime_access = self._mr1._runtime_access
        events = self._mr1._event_log.list_events()
        max_event_index = max((event.event_index for event in events), default=None)
        return RuntimeSnapshot(
            agents=_map_by_id(runtime_access.list_agents(caller_agent_id=caller), "agent_id"),
            workflows=_map_by_id(runtime_access.list_workflows(caller_agent_id=caller), "workflow_id"),
            approvals=_map_by_id(runtime_access.list_pending_approvals(caller_agent_id=caller), "approval_request_id"),
            events=_event_map(events),
            turn_artifacts=_load_turn_artifacts(self._paths.state_path),
            max_event_index=max_event_index,
        )

    def handle_input(
        self,
        user_input: str,
        *,
        request_index: int,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        before = self.snapshot()
        builtin_attempted = user_input.strip().startswith("/")
        builtin_handled = False
        step_called = False
        started_before = self._started
        response_text = ""
        errors: list[dict[str, Any]] = []
        ok = True

        try:
            if builtin_attempted:
                if user_input.strip() == "/memdltr":
                    self.ensure_started()
                builtin_result = self._mr1._handle_builtin(user_input)
                if builtin_result is not None:
                    builtin_handled = True
                    response_text = builtin_result
                else:
                    self.ensure_started()
                    step_called = True
                    response_text = self._mr1.step(user_input, announce=True)
            else:
                self.ensure_started()
                step_called = True
                response_text = self._mr1.step(user_input, announce=True)
        except Exception as exc:
            ok = False
            errors.append(_error_payload(exc))

        after = self.snapshot()
        created_agents, updated_agents = _diff_entities(before.agents, after.agents)
        created_workflows, updated_workflows = _diff_entities(before.workflows, after.workflows)
        new_events = _new_items(before.events, after.events, sort_key="event_index")
        new_turn_artifacts = _new_items(before.turn_artifacts, after.turn_artifacts, sort_key="timestamp")
        approvals_required = _new_items(before.approvals, after.approvals, sort_key="created_at")

        return {
            "ok": ok,
            "request_index": request_index,
            "request_id": request_id,
            "input": user_input,
            "input_kind": "slash_command" if builtin_attempted else "natural_language",
            "response_text": response_text,
            "dispatch": {
                "builtin_attempted": builtin_attempted,
                "builtin_handled": builtin_handled,
                "step_called": step_called,
                "brain_started": self._started,
                "brain_started_for_request": self._started and not started_before,
            },
            "runtime": {
                "isolated": self._paths.isolated,
                "runtime_root": str(self._paths.runtime_root),
                "workflow_root": str(self._paths.workflow_root),
                "state_path": str(self._paths.state_path),
                "session_id": self._mr1._state.session_id,
            },
            "agents": {
                "created": created_agents,
                "updated": updated_agents,
            },
            "workflows": {
                "created": created_workflows,
                "updated": updated_workflows,
            },
            "approvals_required": approvals_required,
            "timeline": {
                "new_event_ids": [event["event_id"] for event in new_events if event.get("event_id")],
                "events": new_events,
                "max_event_index": after.max_event_index,
            },
            "turn_artifacts": new_turn_artifacts,
            "errors": errors,
        }


def _write_payload(payload: dict[str, Any], *, stdout: TextIO, jsonl: bool) -> None:
    if jsonl:
        stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    else:
        stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    stdout.flush()


def _parse_jsonl_input(raw_line: str) -> ParsedJsonlInput:
    stripped = raw_line.strip()
    if not stripped:
        raise ValueError("empty input line")
    if not stripped.startswith("{"):
        return ParsedJsonlInput(input_text=stripped, request_id=None)

    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("JSONL input must be an object or plain text line")
    input_text = payload.get("input")
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError('JSONL input object must include a non-empty "input" string')
    request_id = payload.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise ValueError('"request_id" must be a string when provided')
    return ParsedJsonlInput(input_text=input_text, request_id=request_id)


def _run_jsonl_loop(
    session: RuntimeTestSession,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> int:
    exit_code = 0
    for index, raw_line in enumerate(stdin, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit"}:
            break
        try:
            parsed = _parse_jsonl_input(raw_line)
        except Exception as exc:
            exit_code = 1
            _write_payload(
                {
                    "ok": False,
                    "request_index": index,
                    "request_id": None,
                    "input": stripped,
                    "input_kind": "jsonl_control",
                    "response_text": "",
                    "dispatch": {
                        "builtin_attempted": False,
                        "builtin_handled": False,
                        "step_called": False,
                        "brain_started": session._started,
                        "brain_started_for_request": False,
                    },
                    "runtime": {
                        "isolated": session.paths.isolated,
                        "runtime_root": str(session.paths.runtime_root),
                        "workflow_root": str(session.paths.workflow_root),
                        "state_path": str(session.paths.state_path),
                        "session_id": session._mr1._state.session_id,
                    },
                    "agents": {"created": [], "updated": []},
                    "workflows": {"created": [], "updated": []},
                    "approvals_required": [],
                    "timeline": {
                        "new_event_ids": [],
                        "events": [],
                        "max_event_index": session.snapshot().max_event_index,
                    },
                    "turn_artifacts": [],
                    "errors": [_error_payload(exc if isinstance(exc, Exception) else Exception(str(exc)))],
                },
                stdout=stdout,
                jsonl=True,
            )
            continue

        payload = session.handle_input(
            parsed.input_text,
            request_index=index,
            request_id=parsed.request_id,
        )
        if not payload["ok"]:
            exit_code = 1
        _write_payload(payload, stdout=stdout, jsonl=True)
    return exit_code


def main(
    argv: Optional[list[str]] = None,
    *,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    stdin = stdin or sys.stdin

    if args.jsonl and args.input is not None:
        parser.error("positional input cannot be combined with --jsonl")
    if not args.jsonl and args.input is None:
        parser.error("one-shot input is required unless --jsonl is set")

    try:
        paths, temp_dir = _runtime_paths(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        with contextlib.ExitStack() as stack:
            if temp_dir is not None:
                stack.callback(temp_dir.cleanup)
            stack.enter_context(_patched_runtime_paths(paths))
            session = RuntimeTestSession(paths)
            stack.callback(session.shutdown)
            if args.jsonl:
                return _run_jsonl_loop(session, stdin=stdin, stdout=stdout)
            payload = session.handle_input(args.input, request_index=1, request_id=None)
            _write_payload(payload, stdout=stdout, jsonl=False)
            return 0 if payload["ok"] else 1
    except Exception as exc:
        payload = {
            "ok": False,
            "request_index": 1,
            "request_id": None,
            "input": args.input or "",
            "input_kind": "bootstrap",
            "response_text": "",
            "dispatch": {
                "builtin_attempted": False,
                "builtin_handled": False,
                "step_called": False,
                "brain_started": False,
                "brain_started_for_request": False,
            },
            "runtime": {
                "isolated": not args.live,
                "runtime_root": str(paths.runtime_root),
                "workflow_root": str(paths.workflow_root),
                "state_path": str(paths.state_path),
                "session_id": None,
            },
            "agents": {"created": [], "updated": []},
            "workflows": {"created": [], "updated": []},
            "approvals_required": [],
            "timeline": {
                "new_event_ids": [],
                "events": [],
                "max_event_index": None,
            },
            "turn_artifacts": [],
            "errors": [_error_payload(exc)],
        }
        _write_payload(payload, stdout=stdout, jsonl=False)
        stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
