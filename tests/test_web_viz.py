import json
from urllib.request import Request, urlopen

from mr1.web_viz import WebVizServer


class FakeMR1:
    def __init__(self):
        self.inputs = []

    def build_timeline_snapshot(self):
        return {
            "generated_at": "2026-04-20T14:00:00Z",
            "session": {
                "session_id": "sess-web",
                "started_at": "2026-04-20T13:00:00Z",
                "status": "running",
            },
            "summary": {
                "task_count": 0,
                "running_count": 0,
                "decision_count": 0,
            },
            "root": {
                "id": "mr1",
                "name": "MR1",
                "status": "running",
            },
            "tasks": [],
            "events": [],
            "conversation": [],
            "recent_decisions": [],
        }

    def _handle_builtin(self, text: str):
        if text == "/status":
            return "status ok"
        return None

    def step(self, text: str):
        self.inputs.append(text)
        return f"echo:{text}"


def test_web_viz_serves_snapshot_and_input():
    server = WebVizServer(FakeMR1())
    url = server.start(open_browser=False)

    try:
        with urlopen(f"{url}/api/snapshot") as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["session"]["session_id"] == "sess-web"

        request = Request(
            f"{url}/api/input",
            data=json.dumps({"text": "hello web"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result["ok"] is True
        assert result["text"] == "echo:hello web"

        builtin_request = Request(
            f"{url}/api/input",
            data=json.dumps({"text": "/status"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(builtin_request) as response:
            builtin_result = json.loads(response.read().decode("utf-8"))
        assert builtin_result["kind"] == "command_result"
        assert builtin_result["text"] == "status ok"
    finally:
        server.stop()
