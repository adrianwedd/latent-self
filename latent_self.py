# latent_self.py
"""Latent Self – interactive face-morphing mirror installation.

This prototype captures live webcam frames, encodes faces into the latent space of
StyleGAN-FFHQ via the e4e encoder, walks along pre-computed latent directions for
age, gender, and ethnicity, then projects the modified latent back to the image
space in a smooth forward-and-return cycle.

Heavy ML models are **loaded lazily** – you can start the program without a GPU,
though real-time performance strongly benefits from one.

Dependencies (ℹ tested on Python 3.10):
    pip install opencv-python torch torchvision numpy pillow PyYAML appdirs
    pip install PyQt6  # For --ui qt

Model weights required (download separately):
    * ffhq-1024-stylegan2.pkl      – StyleGAN2-ADA generator
    * e4e_ffhq_encode.pt           – e4e encoder
    * latent_directions.npz        – numpy file with unit vectors for AGE, GENDER, ETHNICITY

Place them under a directory specified by --weights (default: ./models).

Usage:
    python latent_self.py --camera 0 --resolution 512 --cuda
    python latent_self.py --ui qt          # windowed
    python latent_self.py --ui qt --kiosk  # fullscreen kiosk
    
Interactive Controls:
    q       : Quit gracefully
    a       : Morph along the 'Age' axis
    g       : Morph along the 'Gender' axis
    h       : Morph along the 'Smile' axis
    e       : Morph along the 'Ethnicity' axis
    s       : Morph along the 'Species' axis
    u       : Morph along the 'Beauty' axis
    b       : Morph along a blend of all axes (default)
    F12     : (Qt only) Open admin panel
"""

from __future__ import annotations

import argparse
import logging
from logging_setup import configure_logging
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import time
from typing import Any, Dict
import platform

import appdirs
import cv2
import numpy as np
import torch
import yaml
from services import ConfigManager, ModelManager, VideoProcessor, TelemetryClient, asset_path, MemoryMonitor

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - runtime fail hard
    raise RuntimeError(
        "mediapipe is required for face tracking. Install it with 'pip install mediapipe'."
    ) from exc

@dataclass
class DirectionUI:
    """Helper structure binding a UI slider and label to a direction."""
    name: str
    slider: "QSlider"
    label: "QLabel"



try:
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QSlider, QLabel
    from ui.fullscreen import MirrorWindow
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False





class LatentSelf:
    """Orchestrates models, video processing and UI components."""
    def __init__(
        self,
        config: ConfigManager,
        camera_index: int,
        resolution: int,
        device: str,
        weights_dir: Path,
        ui: str,
        kiosk: bool,
        demo: bool = False,
        low_power: bool = False,
        model_manager: ModelManager | None = None,
        video_processor: VideoProcessor | None = None,
        telemetry: TelemetryClient | None = None,
    ) -> None:
        """Construct the application core.

        Args:
            config: Loaded application configuration manager.
            camera_index: Index of the webcam to use.
            resolution: Square output resolution.
            device: Torch device string (``"cpu"`` or ``"cuda"``).
            weights_dir: Directory containing model weights.
            ui: UI backend to use.
            kiosk: Whether to enable fullscreen kiosk mode.
            demo: Use prerecorded media from ``data/`` instead of a webcam.
            low_power: Enable adaptive resolution and frame skipping.
            model_manager: Optional pre-created :class:`ModelManager`.
            video_processor: Optional pre-created :class:`VideoProcessor`.
            telemetry: Optional :class:`TelemetryClient` for metrics.
        """
        self.config = config
        self.device = torch.device(device)
        self.ui = ui
        self.kiosk = kiosk
        self.model_manager = model_manager or ModelManager(weights_dir, self.device)
        self.telemetry = telemetry or TelemetryClient(config)
        self.low_power = low_power
        self.video = video_processor or VideoProcessor(
            self.model_manager,
            config,
            self.device,
            camera_index,
            resolution,
            ui,
            self.telemetry,
            demo,
            low_power,
        )

        self.memory = MemoryMonitor(config)

    def run(self) -> None:
        """Start the application UI and processing loop."""
        logging.info("Starting Latent Self…")
        self.memory.start()
        try:
            if self.ui == "qt":
                if not QT_AVAILABLE:
                    logging.error("Qt UI requested, but PyQt6 is not installed. Please run: pip install PyQt6")
                    sys.exit(1)
                self._run_qt()
            else:
                self._run_cv2()
        except Exception:
            logging.exception("Unhandled exception")
            if QT_AVAILABLE and self.ui == "qt":
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(None, "Latent Self Error", "An unexpected error occurred. Check logs.")
        finally:
            self.video.stop()
            if self.telemetry:
                self.telemetry.shutdown()
            self.model_manager.unload()
            self.memory.stop()
            logging.info("Application shut down gracefully.")

    def _run_cv2(self) -> None:
        """Display output using OpenCV windows."""
        logging.info(
            "Using cv2 UI. Controls: [q]uit | [y]age | [g]ender | [h]smile | "
            "[s]pecies | [u]beauty | [1]happy | [2]angry | [3]sad | [4]fear | [5]disgust | [6]surprise | [b]blend"
        )
        self.video.start()
        self.video.join()

    def _run_qt(self) -> None:
        """Launch the Qt based user interface."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        self.window = MirrorWindow(self)
        worker = VideoWorker(self.video)
        worker.new_frame.connect(self.window.update_frame)
        worker.start()
        if self.kiosk:
            self.window.show_fullscreen()
        else:
            self.window.show()
        app.exec()
        self.video.stop()
        worker.wait()

if QT_AVAILABLE:
    class VideoWorker(QThread):
        """QThread worker for video processing."""
        new_frame = pyqtSignal(QImage)

        def __init__(self, processor: VideoProcessor):
            super().__init__()
            self.processor = processor

        def run(self):
            self.processor.start(self.new_frame)


# -------------------------------------------------------------------------------------------------
# CLI entry-point ---------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate and sanitize CLI arguments.

    Args:
        args: Namespace of parsed CLI values.
        parser: Argument parser used for reporting errors.
    """
    if args.camera < 0 or args.camera > 10:
        parser.error("--camera must be between 0 and 10")
    if args.resolution < 64 or args.resolution > 2048:
        parser.error("--resolution must be between 64 and 2048")
    if args.fps is not None and not (1 <= args.fps <= 120):
        parser.error("--fps must be between 1 and 120")
    if args.cycle_duration is not None and args.cycle_duration <= 0:
        parser.error("--cycle-duration must be positive")
    for name in ("blend_age", "blend_gender", "blend_smile", "blend_species"):
        val = getattr(args, name)
        if val is not None and not (0.0 <= val <= 1.0):
            parser.error(f"--{name.replace('_','-')} must be between 0 and 1")
    if args.emotion and args.emotion.lower() not in {"happy", "angry", "sad", "fear", "disgust", "surprise"}:
        parser.error("--emotion must be one of happy, angry, sad, fear, disgust, surprise")
    if args.max_cpu_mem is not None and args.max_cpu_mem <= 0:
        parser.error("--max-cpu-mem must be positive")
    if args.max_gpu_mem is not None and args.max_gpu_mem <= 0:
        parser.error("--max-gpu-mem must be positive")
    args.weights = Path(args.weights).expanduser().resolve()
    if not args.weights.exists():
        parser.error(f"Weights directory does not exist: {args.weights}")

def main(argv: list[str] | None = None) -> None:
    """Entry point for the command line executable."""
    parser = argparse.ArgumentParser(
        description="Latent Self interactive mirror",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--resolution", type=int, default=512, help="Square frame size (px)")
    parser.add_argument("--fps", type=int, default=None, help="Target frames per second (overrides config)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--max-cpu-mem", type=int, default=None, help="Max CPU memory MB before warning")
    parser.add_argument("--max-gpu-mem", type=float, default=None, help="Max GPU memory GB before warning")
    parser.add_argument("--weights", type=Path, default=asset_path("models"), help="Directory for model weights")
    parser.add_argument("--ui", type=str, default="cv2", choices=["cv2", "qt"], help="UI backend to use")
    parser.add_argument("--kiosk", action="store_true", help="Hide cursor and launch fullscreen (Qt only)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--low-power", action="store_true", dest="low_power", help="Enable adaptive low power mode")
    parser.add_argument(
        "--demo",
        "--test",
        action="store_true",
        dest="demo",
        help="Use prerecorded media from data/ instead of webcam",
    )

    g = parser.add_argument_group("Morphing Controls (overrides config)")
    g.add_argument("--cycle-duration", type=float, default=None, help="Duration of one morph cycle (seconds)")
    g.add_argument("--blend-age", type=float, default=None, help="Weight for age in blended mode")
    g.add_argument("--blend-gender", type=float, default=None, help="Weight for gender in blended mode")
    g.add_argument("--blend-smile", type=float, default=None, help="Weight for smile in blended mode")
    g.add_argument("--blend-species", type=float, default=None, help="Weight for species in blended mode")
    g.add_argument("--emotion", type=str, default=None, help="Start with given emotion (happy, angry, sad, fear, disgust, surprise)")

    args = parser.parse_args(argv)

    _validate_args(args, parser)

    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(args.kiosk, level=log_level)

    config = ConfigManager(args)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device == "cpu":
        logging.warning("CUDA requested but not available – falling back to CPU")

    app = LatentSelf(
        config=config,
        camera_index=args.camera,
        resolution=args.resolution,
        device=device,
        weights_dir=args.weights,
        ui=args.ui,
        kiosk=args.kiosk,
        demo=args.demo,
        low_power=args.low_power,
    )
    config.app = app
    app.run()


if __name__ == "__main__":
    main()
