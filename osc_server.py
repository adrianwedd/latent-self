from __future__ import annotations

"""OSC server integration for live control of morphing parameters.

The server listens for messages that mirror the application hotkeys:

```
/direction       "AGE"|"GENDER"|"SMILE"|...
/blend/age       0.0 - 1.0
/cycle_duration  seconds
```

These updates are applied immediately to the running :class:`VideoProcessor`.
"""

import logging
from threading import Thread
from typing import Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from services import ConfigManager, VideoProcessor
from directions import Direction


class OSCServer:
    """Run a minimal OSC server in a background thread."""

    def __init__(self, config: ConfigManager, video: VideoProcessor, host: str = "0.0.0.0", port: int = 9000) -> None:
        self.config = config
        self.video = video
        self._dispatcher = Dispatcher()
        self._dispatcher.map("/direction", self._on_direction)
        # Support dynamic blend weight paths like /blend/age 0.5
        self._dispatcher.map("/blend/*", self._on_blend)
        self._dispatcher.map("/cycle_duration", self._on_cycle)
        self._server = ThreadingOSCUDPServer((host, port), self._dispatcher)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        logging.info("OSC listening on %s:%s", host, port)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------
    def _on_direction(self, addr: str, *args: Any) -> None:
        if not args:
            return
        value = str(args[0])
        try:
            direction = Direction.from_str(value)
            self.video.enqueue_direction(direction)
        except Exception as e:  # noqa: BLE001 - invalid input
            logging.warning("OSC: invalid direction %s - %s", value, e)

    def _on_blend(self, addr: str, *args: Any) -> None:
        if not args:
            return
        name = addr.split("/")[-1]
        try:
            val = float(args[0])
        except (TypeError, ValueError):
            logging.warning("OSC: invalid blend weight for %s", name)
            return
        with self.config._lock:  # type: ignore[attr-defined]
            self.config.data.setdefault("blend_weights", {})[name] = val
        self.video.blend_weights[name] = val

    def _on_cycle(self, addr: str, *args: Any) -> None:
        if not args:
            return
        try:
            val = float(args[0])
        except (TypeError, ValueError):
            logging.warning("OSC: invalid cycle duration")
            return
        with self.config._lock:  # type: ignore[attr-defined]
            self.config.data["cycle_duration"] = val
        self.video.cycle_seconds = val

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._thread.join()
