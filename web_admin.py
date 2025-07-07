from __future__ import annotations

"""Simple Flask-based admin API for reading and updating config."""

from threading import Thread
from typing import Any

import yaml
from flask import Flask, request, jsonify, abort
from werkzeug.security import check_password_hash
from werkzeug.serving import make_server

from services import ConfigManager


class WebAdmin:
    """Expose config values over HTTP with basic auth."""

    def __init__(self, config: ConfigManager, host: str = "0.0.0.0", port: int = 8001) -> None:
        self.config = config
        self.app = Flask(__name__)
        self._server = make_server(host, port, self.app)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._setup_routes()

    # ------------------------------------------------------------------
    # Flask route handlers
    # ------------------------------------------------------------------

    def _auth(self) -> bool:
        """Return True if request provides valid credentials."""
        password_hash = self.config.data.get("admin_password_hash", "")
        if not password_hash:
            return True
        auth = request.authorization
        return bool(auth and check_password_hash(password_hash, auth.password or ""))

    def _setup_routes(self) -> None:
        @self.app.get("/config")
        def get_config():
            if not self._auth():
                abort(401)
            return jsonify(self.config.data)

        @self.app.post("/config")
        def update_config() -> Any:
            if not self._auth():
                abort(401)
            updates = request.get_json(silent=True) or {}
            with self.config._lock:  # type: ignore[attr-defined]
                self.config.data.update(updates)
                with self.config.config_path.open("w") as f:
                    yaml.dump(self.config.data, f, default_flow_style=False)
            self.config.reload()
            return jsonify({"status": "ok"})

        @self.app.post("/reload")
        def reload_config() -> Any:
            if not self._auth():
                abort(401)
            try:
                self.config.reload()
                return jsonify({"status": "ok"})
            except Exception as exc:  # noqa: BLE001 - runtime
                return jsonify({"status": "error", "error": str(exc)}), 500

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._thread.join()
