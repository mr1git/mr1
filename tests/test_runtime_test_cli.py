from __future__ import annotations

import json
from io import StringIO

from mr1 import runtime_test_cli
from mr1.mr1 import MR1
from mr1.routing_advisor import RouteAdvice


def _force_clarification(*_args, **_kwargs) -> RouteAdvice:
    return RouteAdvice(
        route="ask_clarification",
        required_refs=[],
        side_effects_allowed=False,
        recommended_commands=["ask_clarification"],
        confidence=1.0,
        reason="test route",
    )


def test_isolated_mode_does_not_touch_live_runtime_root(tmp_path, monkeypatch):
    live_root = tmp_path / "live-runtime"
    isolated_root = tmp_path / "isolated-runtime"
    live_state_path = live_root / "active" / "mr1_state.json"
    live_state_path.parent.mkdir(parents=True, exist_ok=True)
    live_state_path.write_text('{"sentinel":"live"}', encoding="utf-8")

    monkeypatch.setattr(runtime_test_cli, "_default_live_runtime_root", lambda: live_root)
    monkeypatch.setattr(MR1, "start", lambda self: None)

    stdout = StringIO()
    rc = runtime_test_cli.main(
        ["--isolated", "--runtime-root", str(isolated_root), "/status"],
        stdout=stdout,
        stderr=StringIO(),
    )

    assert rc == 0
    assert live_state_path.read_text(encoding="utf-8") == '{"sentinel":"live"}'
    assert (isolated_root / "active" / "mr1_state.json").exists()


def test_natural_language_input_uses_real_step_path(tmp_path, monkeypatch):
    calls: list[tuple[str, bool]] = []
    original_step = MR1.step

    def step_spy(self, user_input: str, announce: bool = False) -> str:
        calls.append((user_input, announce))
        return original_step(self, user_input, announce=announce)

    monkeypatch.setattr(MR1, "start", lambda self: None)
    monkeypatch.setattr(MR1, "step", step_spy)
    monkeypatch.setattr("mr1.orchestrator.root.build_route_advice", _force_clarification)

    stdout = StringIO()
    rc = runtime_test_cli.main(
        ["--isolated", "--runtime-root", str(tmp_path / "runtime"), "create a test agent"],
        stdout=stdout,
        stderr=StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert rc == 0
    assert calls == [("create a test agent", True)]
    assert payload["dispatch"]["step_called"] is True
    assert payload["response_text"] == (
        "I want to make sure I do the right thing here — what exactly do you want me to act on?"
    )
    assert payload["timeline"]["events"][-1]["event_type"] == "runtime_turn_decided"
    assert payload["turn_artifacts"][-1]["routing_decision"]["final_action"] == "ask_clarification"


def test_slash_command_uses_builtin_path(tmp_path, monkeypatch):
    calls: list[str] = []
    original_builtin = MR1._handle_builtin

    def builtin_spy(self, cmd: str):
        calls.append(cmd)
        return original_builtin(self, cmd)

    monkeypatch.setattr(MR1, "start", lambda self: None)
    monkeypatch.setattr(MR1, "_handle_builtin", builtin_spy)

    stdout = StringIO()
    rc = runtime_test_cli.main(
        ["--isolated", "--runtime-root", str(tmp_path / "runtime"), "/status"],
        stdout=stdout,
        stderr=StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert rc == 0
    assert calls == ["/status"]
    assert payload["dispatch"]["builtin_attempted"] is True
    assert payload["dispatch"]["builtin_handled"] is True
    assert payload["dispatch"]["step_called"] is False
    assert "Session:" in payload["response_text"]


def test_unknown_slash_command_stays_on_builtin_path(tmp_path, monkeypatch):
    calls: list[str] = []
    original_builtin = MR1._handle_builtin

    def builtin_spy(self, cmd: str):
        calls.append(cmd)
        return original_builtin(self, cmd)

    monkeypatch.setattr(MR1, "start", lambda self: None)
    monkeypatch.setattr(MR1, "_handle_builtin", builtin_spy)

    stdout = StringIO()
    rc = runtime_test_cli.main(
        ["--isolated", "--runtime-root", str(tmp_path / "runtime"), "/notacommand"],
        stdout=stdout,
        stderr=StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert rc == 0
    assert calls == ["/notacommand"]
    assert payload["dispatch"]["builtin_attempted"] is True
    assert payload["dispatch"]["builtin_handled"] is True
    assert payload["dispatch"]["step_called"] is False
    assert payload["response_text"].startswith("Unknown slash command: /notacommand.")


def test_workflow_bare_slash_stays_on_builtin_path_and_returns_usage(tmp_path, monkeypatch):
    builtin_calls: list[str] = []
    step_calls: list[str] = []
    original_builtin = MR1._handle_builtin
    original_step = MR1.step

    def builtin_spy(self, cmd: str):
        builtin_calls.append(cmd)
        return original_builtin(self, cmd)

    def step_spy(self, user_input: str, announce: bool = False) -> str:
        step_calls.append(user_input)
        return original_step(self, user_input, announce=announce)

    monkeypatch.setattr(MR1, "start", lambda self: None)
    monkeypatch.setattr(MR1, "_handle_builtin", builtin_spy)
    monkeypatch.setattr(MR1, "step", step_spy)

    stdout = StringIO()
    rc = runtime_test_cli.main(
        ["--isolated", "--runtime-root", str(tmp_path / "runtime"), "/workflow"],
        stdout=stdout,
        stderr=StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert rc == 0
    assert builtin_calls == ["/workflow"]
    assert step_calls == []
    assert payload["dispatch"]["builtin_attempted"] is True
    assert payload["dispatch"]["builtin_handled"] is True
    assert payload["dispatch"]["step_called"] is False
    assert payload["response_text"].startswith("Usage: /workflow <workflow_id>")


def test_jsonl_mode_emits_parseable_json(tmp_path, monkeypatch):
    monkeypatch.setattr(MR1, "start", lambda self: None)
    monkeypatch.setattr("mr1.orchestrator.root.build_route_advice", _force_clarification)

    stdout = StringIO()
    rc = runtime_test_cli.main(
        ["--isolated", "--runtime-root", str(tmp_path / "runtime"), "--jsonl"],
        stdin=StringIO('/status\n{"input":"create a test agent","request_id":"req-2"}\n'),
        stdout=stdout,
        stderr=StringIO(),
    )

    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert rc == 0
    assert len(lines) == 2
    assert lines[0]["input"] == "/status"
    assert lines[1]["request_id"] == "req-2"
    assert lines[1]["dispatch"]["step_called"] is True


def test_errors_are_structured_not_tracebacks(tmp_path, monkeypatch):
    def explode(self, cmd: str):
        raise RuntimeError(f"boom: {cmd}")

    monkeypatch.setattr(MR1, "start", lambda self: None)
    monkeypatch.setattr(MR1, "_handle_builtin", explode)

    stdout = StringIO()
    stderr = StringIO()
    rc = runtime_test_cli.main(
        ["--isolated", "--runtime-root", str(tmp_path / "runtime"), "/status"],
        stdout=stdout,
        stderr=stderr,
    )

    payload = json.loads(stdout.getvalue())
    assert rc == 1
    assert payload["ok"] is False
    assert payload["errors"] == [{"type": "RuntimeError", "message": "boom: /status"}]
    assert "Traceback" not in stdout.getvalue()
    assert "Traceback" not in stderr.getvalue()
