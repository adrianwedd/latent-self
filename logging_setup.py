"""Logging utilities for console and kiosk modes."""

import json
import logging
from logging import LogRecord
from typing import Any

class JsonFormatter(logging.Formatter):
    """Output logs as JSON objects."""

    def format(self, record: LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }
        if record.name != "root":
            log_record["logger"] = record.name
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def configure_logging(kiosk: bool, level: int = logging.INFO) -> None:
    """Configure root logging handler."""
    handler = logging.StreamHandler()
    formatter = JsonFormatter() if kiosk else logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler])


from contextlib import contextmanager
from time import time


class FrameTimer:
    """Aggregate frame processing durations and emit periodic stats."""

    def __init__(self, interval: float = 10.0) -> None:
        self.interval = interval
        self._durations: list[float] = []
        self._last_emit = time()

    def record(self, dur: float) -> None:
        self._durations.append(dur)
        now = time()
        if now - self._last_emit >= self.interval and self._durations:
            total = sum(self._durations)
            count = len(self._durations)
            fps = count / total if total else 0.0
            avg_latency = total / count
            logging.info("metrics.fps %.2f metrics.latency %.3fs", fps, avg_latency)
            self._durations.clear()
            self._last_emit = now

    @contextmanager
    def track(self) -> Any:
        start = time()
        try:
            yield
        finally:
            self.record(time() - start)

@contextmanager
def log_timing(label: str) -> Any:
    """Context manager to log timing metrics with DEBUG level."""
    start = time()
    try:
        yield
    finally:
        duration = time() - start
        logging.debug("timing.%s %.3fs", label, duration)

