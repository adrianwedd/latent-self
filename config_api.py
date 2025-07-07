"""Simple HTTP API to trigger configuration reloads."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from services import ConfigManager


class _Handler(BaseHTTPRequestHandler):
    config: ConfigManager | None = None

    def do_POST(self) -> None:  # pragma: no cover - runtime behavior
        if self.path == "/reload" and self.config:
            try:
                self.config.reload()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"reloaded")
            except Exception as exc:  # noqa: BLE001 - external interface
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(exc).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
        return  # suppress default logging


class ConfigAPIServer:
    """Minimal HTTP server exposing a ``/reload`` endpoint."""

    def __init__(self, config: ConfigManager, host: str = "127.0.0.1", port: int = 5001) -> None:
        self._server = HTTPServer((host, port), _Handler)
        _Handler.config = config
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        """Start serving requests in a background thread."""
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the server and join the thread."""
        self._server.shutdown()
        self._thread.join()
