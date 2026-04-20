"""
Local web visualization server for MR1.

Serves a browser UI that polls live timeline snapshots and can submit
messages back to the active MR1 instance.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


_PKG_ROOT = Path(__file__).resolve().parent
_WEB_ROOT = _PKG_ROOT / "webui"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


class WebVizServer:
    def __init__(self, mr1_instance: Any):
        self._mr1 = mr1_instance
        self._request_lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._url: Optional[str] = None

    @property
    def url(self) -> Optional[str]:
        return self._url

    def _snapshot(self) -> dict[str, Any]:
        return self._mr1.build_timeline_snapshot()

    def _handle_input(self, text: str) -> dict[str, Any]:
        with self._request_lock:
            builtin_result = self._mr1._handle_builtin(text)
            if builtin_result is not None:
                return {"ok": True, "kind": "command_result", "text": builtin_result}
            return {"ok": True, "kind": "response", "text": self._mr1.step(text)}

    def start(self, host: str = "127.0.0.1", port: int = 0, open_browser: bool = True) -> str:
        if self._httpd is not None and self._url is not None:
            return self._url

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def _send_bytes(
                self,
                data: bytes,
                content_type: str,
                status: int = HTTPStatus.OK,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _serve_file(self, relative_path: str, content_type: str) -> None:
                target = (_WEB_ROOT / relative_path).resolve()
                if not str(target).startswith(str(_WEB_ROOT.resolve())) or not target.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(target.read_bytes(), content_type)

            def _read_json_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return {}

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    self._serve_file("index.html", "text/html; charset=utf-8")
                    return
                if self.path == "/app.js":
                    self._serve_file("app.js", "text/javascript; charset=utf-8")
                    return
                if self.path == "/styles.css":
                    self._serve_file("styles.css", "text/css; charset=utf-8")
                    return
                if self.path == "/api/snapshot":
                    self._send_bytes(
                        _json_bytes(server_ref._snapshot()),
                        "application/json; charset=utf-8",
                    )
                    return
                if self.path == "/api/health":
                    self._send_bytes(
                        _json_bytes({"ok": True, "url": server_ref.url}),
                        "application/json; charset=utf-8",
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/api/input":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = self._read_json_body()
                text = str(payload.get("text", "")).strip()
                if not text:
                    self._send_bytes(
                        _json_bytes({"ok": False, "error": "Missing input text"}),
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    result = server_ref._handle_input(text)
                except Exception as exc:  # pragma: no cover - defensive
                    self._send_bytes(
                        _json_bytes({"ok": False, "error": str(exc)}),
                        "application/json; charset=utf-8",
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return
                self._send_bytes(_json_bytes(result), "application/json; charset=utf-8")

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        bind_host, bind_port = self._httpd.server_address[:2]
        self._url = f"http://{bind_host}:{bind_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        if open_browser:
            webbrowser.open(self._url)

        return self._url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None


def serve_standalone(
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> int:
    from mr1.mr1 import MR1

    mr1 = MR1()
    mr1.start()
    server = WebVizServer(mr1)
    url = server.start(host=host, port=port, open_browser=open_browser)
    print(f"MR1 web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        mr1.shutdown("web_visualizer_shutdown")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MR1 web visualization UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        serve_standalone(host=args.host, port=args.port, open_browser=not args.no_browser)
    )


if __name__ == "__main__":
    main()
