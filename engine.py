from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from services import VideoProcessor

try:
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtGui import QImage
    from ui.fullscreen import MirrorWindow
    QT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    QT_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from latent_self import LatentSelf


if QT_AVAILABLE:
    class VideoWorker(QThread):
        """QThread worker for video processing."""

        new_frame = pyqtSignal(QImage)
        preview_frame = pyqtSignal(QImage)

        def __init__(self, processor: VideoProcessor) -> None:
            super().__init__()
            self.processor = processor

        def run(self) -> None:  # pragma: no cover - Qt event loop
            self.processor.start(self.new_frame, self.preview_frame)


class Engine:
    """Run the application event loop and UI."""

    def __init__(self, app: LatentSelf) -> None:
        self.app = app

    def _run_cv2(self) -> None:
        logging.info(
            "Using cv2 UI. Controls: [q]uit | [y]age | [g]gender | [h]smile | "
            "[s]pecies | [u]beauty | [1]happy | [2]angry | [3]sad | [4]fear | "
            "[5]disgust | [6]surprise | [b]blend"
        )
        self.app.video.start()
        self.app.video.join()

    def _run_qt(self) -> None:
        from PyQt6.QtWidgets import QApplication

        app = QApplication(sys.argv)
        window = MirrorWindow(self.app)
        worker = VideoWorker(self.app.video)
        worker.new_frame.connect(window.update_frame)
        worker.start()
        if self.app.kiosk:
            window.show_fullscreen()
        else:
            window.show()
        app.exec()
        self.app.video.stop()
        worker.wait()

    def run(self) -> None:
        """Start subsystems and launch the chosen UI."""

        logging.info("Starting Latent Self…")
        self.app.memory.start()
        self.app.audio.start()
        self.app.scheduler.start()
        if self.app.config.data.get("osc", {}).get("enabled"):
            try:
                from osc_server import OSCServer

                port = int(self.app.config.data["osc"].get("port", 9000))
                self.app._osc_server = OSCServer(
                    self.app.config, self.app.video, port=port
                )
                self.app._osc_server.start()
                logging.info("OSC server started")
            except Exception as e:  # noqa: BLE001 - runtime
                logging.warning("Failed to start OSC server: %s", e)
        if self.app._start_web_admin:
            try:
                from web_admin import WebAdmin

                self.app._web_server = WebAdmin(self.app.config)
                self.app._web_server.start()
                logging.info("Web admin started")
            except Exception as e:  # noqa: BLE001 - runtime
                logging.warning("Failed to start web admin: %s", e)
        try:
            if self.app.ui == "qt":
                if not QT_AVAILABLE:
                    logging.error(
                        "Qt UI requested, but PyQt6 is not installed. Please run: pip install PyQt6"
                    )
                    sys.exit(1)
                self._run_qt()
            else:
                self._run_cv2()
        except Exception:  # pragma: no cover - runtime
            logging.exception("Unhandled exception")
            if QT_AVAILABLE and self.app.ui == "qt":
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.critical(
                    None,
                    "Latent Self Error",
                    "An unexpected error occurred. Check logs.",
                )
        finally:
            self.app.video.stop()
            if self.app._osc_server:
                self.app._osc_server.shutdown()
            if self.app._web_server:
                self.app._web_server.shutdown()
            if self.app.telemetry:
                self.app.telemetry.shutdown()
            self.app.model_manager.unload()
            self.app.audio.stop()
            self.app.memory.stop()
            self.app.scheduler.shutdown()
            logging.info("Application shut down gracefully.")
