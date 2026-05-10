"""MR1 Claude session process wrapper.

`MR1Process` is the thin shim that drives MR1's Claude CLI per turn.
Claude Code does not expose a long-lived interactive JSON mode for
this workflow, so each turn invokes `claude --print` with stream-json
I/O and resumes the prior Claude session ID when one is known.
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional


class MR1Process:
    """
    Manages MR1's Claude Code session.

    Claude Code does not expose a stable long-lived interactive JSON mode
    for this workflow. Instead, each turn is executed with `claude --print`
    using stream-json I/O and the prior Claude session ID is resumed when
    available.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str,
        tools: list[str],
        session_id: Optional[str] = None,
    ):
        self._system_prompt = system_prompt
        self._model = model
        self._tools = tools
        self._session_id = session_id
        self._available = False

    def start(self) -> None:
        """Verify Claude Code is available for per-turn session use."""
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"claude CLI is unavailable: {detail}")
        self._available = True

    def send(self, message: str) -> str:
        """
        Execute a single Claude turn and return the final result text.
        """
        if not self.alive:
            return "[MR1 ERROR] Process is not running."

        result_text, error_text = self._invoke(message, resume=bool(self._session_id))
        if error_text and self._session_id:
            self._session_id = None
            result_text, error_text = self._invoke(message, resume=False)

        if error_text:
            return f"[MR1 ERROR] {error_text}"

        self._available = True
        return result_text

    def _invoke(self, message: str, resume: bool) -> tuple[str, Optional[str]]:
        try:
            payload = json.dumps(
                {"type": "user", "message": {"role": "user", "content": message}}
            )
        except TypeError as exc:
            return "", f"failed to encode input: {exc}"

        cmd = [
            "claude",
            "--print",
            "--verbose",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--replay-user-messages",
        ]
        if self._model:
            cmd.extend(["--model", self._model])
        if self._tools:
            cmd.extend(["--allowedTools", ",".join(self._tools)])
        if resume and self._session_id:
            cmd.extend(["--resume", self._session_id])
        else:
            cmd.extend(["--append-system-prompt", self._system_prompt])

        try:
            result = subprocess.run(
                cmd,
                input=payload + "\n",
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return "", "claude turn timed out"
        except OSError as exc:
            return "", f"could not run claude: {exc}"

        stdout = result.stdout or ""
        stderr = (result.stderr or "").strip()
        parsed_text = ""
        parsed_session_id = None
        parse_errors = 0

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if event.get("session_id"):
                parsed_session_id = event["session_id"]
            if event.get("type") == "result":
                parsed_text = event.get("result", "")
                parsed_session_id = event.get("session_id", parsed_session_id)

        if parsed_session_id:
            self._session_id = parsed_session_id

        if result.returncode != 0:
            detail = stderr or parsed_text.strip()
            if not detail and parse_errors:
                detail = "received malformed stream-json output"
            return "", detail or f"claude exited with code {result.returncode}"

        if not parsed_text:
            detail = stderr or "claude returned no result text"
            return "", detail

        return parsed_text, None

    def kill(self) -> None:
        """Forget the current Claude session handle."""
        self._session_id = None
        self._available = False

    @property
    def pid(self) -> Optional[int]:
        return None

    @property
    def alive(self) -> bool:
        return self._available

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id
