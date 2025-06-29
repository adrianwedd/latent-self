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
    python latent_self.py --ui qt --kiosk

Interactive Controls:
    q       : Quit gracefully
    a       : Morph along the 'Age' axis
    g       : Morph along the 'Gender' axis
    e       : Morph along the 'Ethnicity' axis
    b       : Morph along a blend of all axes (default)
    F12     : (Qt only) Open admin panel
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from threading import Event, Thread
from time import time
from typing import Any, Dict
from dataclasses import dataclass
import appdirs
import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml
import socket

@dataclass
class DirectionUI:
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





# -------------------------------------------------------------------------------------------------
# Configuration management ----------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def asset_path(relative_path: str) -> Path:
    """Get the absolute path to an asset, handling PyInstaller bundling.

    Args:
        relative_path: The relative path to the asset.

    Returns:
        The absolute path to the asset.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a bundle
        base_path = Path(sys._MEIPASS)
    else:
        # Running in a normal Python environment
        base_path = Path(__file__).parent
    return base_path / relative_path


class Config:
    """Manages loading, saving, and reloading of the YAML configuration file.

    Attributes:
        config_dir: The directory where the configuration file is stored.
        config_path: The path to the configuration file.
        directions_path: The path to the directions file.
        data: The configuration data.
        directions_data: The directions data.
        app: The main application instance.
    """

    def __init__(self, cli_args: argparse.Namespace, app: 'LatentSelf' | None = None):
        """Initializes the Config class.

        Args:
            cli_args: The command-line arguments.
            app: The main application instance.
        """
        self.config_dir = Path(appdirs.user_config_dir("LatentSelf"))
        self.config_path = self.config_dir / "config.yaml"
        self.directions_path = self.config_dir / "directions.yaml"
        self._ensure_config_exists()
        self._ensure_directions_exists()
        self.data: Dict[str, Any] = {}
        self.directions_data: Dict[str, Any] = {}
        self.app = app  # Store reference to the main app instance
        self.reload()

        # CLI arguments override config file values
        self._override_with_cli(cli_args)

    def _ensure_config_exists(self) -> None:
        """If the user config file doesn't exist, copy the default one."""
        if not self.config_path.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            default_config = asset_path("data/config.yaml")
            shutil.copy(default_config, self.config_path)
            logging.info(f"Created default config at {self.config_path}")

    def _ensure_directions_exists(self) -> None:
        """If the user directions file doesn't exist, copy the default one."""
        if not self.directions_path.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            default_directions = asset_path("data/directions.yaml")
            shutil.copy(default_directions, self.directions_path)
            logging.info(f"Created default directions at {self.directions_path}")

    def reload(self) -> None:
        """Re-read the configuration file from disk."""
        with self.config_path.open("r") as f:
            self.data = yaml.safe_load(f)
        with self.directions_path.open("r") as f:
            self.directions_data = yaml.safe_load(f)
        logging.info("Configuration reloaded.")
        if self.app:
            self.app._apply_config()

    def _override_with_cli(self, args: argparse.Namespace) -> None:
        """Update config with any values explicitly set on the command line."""
        # For each CLI arg, if it's not the default value, it overrides the config.
        if args.cycle_duration is not None:
            self.data['cycle_duration'] = args.cycle_duration
        if args.blend_age is not None:
            self.data['blend_weights']['age'] = args.blend_age
        if args.blend_gender is not None:
            self.data['blend_weights']['gender'] = args.blend_gender
        if args.blend_smile is not None:
            self.data['blend_weights']['smile'] = args.blend_smile
        if args.blend_species is not None:
            self.data['blend_weights']['species'] = args.blend_species
        if args.fps is not None:
            self.data['fps'] = args.fps


# -------------------------------------------------------------------------------------------------
# Lazy model loader helpers -----------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def _lazy_once(fn):
    """Decorator that turns a function into a lazy singleton property."""
    cache = {}
    def wrapper(*args, **kwargs):
        if fn not in cache:
            cache[fn] = fn(*args, **kwargs)
        return cache[fn]
    return wrapper


@_lazy_once
def get_stylegan_generator(weights_dir: Path):
    """Load StyleGAN2-ADA FFHQ generator (Torch .pkl).

    Args:
        weights_dir: The directory where the model weights are stored.

    Returns:
        The StyleGAN generator.
    """
    import pickle  # noqa: WPS433 – model loading OK

    gen_path = weights_dir / "ffhq-1024-stylegan2.pkl"
    if not gen_path.exists():
        logging.error("StyleGAN generator not found at %s", gen_path)
        sys.exit(1)

    logging.info("Loading StyleGAN2 generator from %s", gen_path)
    with gen_path.open("rb") as fp:
        # We only need the generator component ('G_ema')
        generator = pickle.load(fp)["G_ema"].eval()
    return generator


@_lazy_once
def get_e4e_encoder(weights_dir: Path):
    """Load e4e encoder (Torch state_dict).

    Args:
        weights_dir: The directory where the model weights are stored.

    Returns:
        The e4e encoder.
    """
    try:
        from models.encoders import pSp  # Imported late; pSp repo must be on PYTHONPATH
    except ModuleNotFoundError:
        logging.error(
            "Module 'pSp' not found. Ensure the e4e repository is on your PYTHONPATH."
        )
        sys.exit(1)

    encoder_path = weights_dir / "e4e_ffhq_encode.pt"
    if not encoder_path.exists():
        logging.error("e4e encoder not found at %s", encoder_path)
        sys.exit(1)

    logging.info("Loading e4e encoder from %s", encoder_path)
    opts = torch.load(encoder_path, map_location="cpu")["opts"]
    opts["checkpoint_path"] = str(encoder_path)
    opts = argparse.Namespace(**opts)
    encoder = pSp(opts).eval()
    return encoder


@_lazy_once
def get_latent_directions(weights_dir: Path) -> Dict[str, np.ndarray]:
    """Return dict with all available unit directions from the .npz file.

    Args:
        weights_dir: The directory where the model weights are stored.

    Returns:
        A dictionary of latent directions.
    """
    path = weights_dir / "latent_directions.npz"
    if not path.exists():
        logging.error("Latent directions file not found at %s", path)
        sys.exit(1)

    logging.info("Loading latent directions from %s", path)
    with np.load(path) as data:
        return {k.upper(): data[k] for k in data.keys()}


# -------------------------------------------------------------------------------------------------
# Utility functions -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def _to_tensor(img: np.ndarray) -> torch.Tensor:  # (H W C) uint8 BGR → float32 NCHW RGB [-1,1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor.unsqueeze(0)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:  # NCHW RGB [-1,1] → (H W C) uint8 BGR
    tensor = (tensor.squeeze(0).clamp(-1, 1) + 1.0) * 127.5
    img_rgb = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def _numpy_to_qimage(img: np.ndarray) -> QImage: # (H, W, C) BGR -> QImage
    h, w, ch = img.shape
    bytes_per_line = ch * w
    return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)


# -------------------------------------------------------------------------------------------------
# Core processing class ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

################################################################################
# Face‑tracking helper                                                         #
################################################################################

class _EyeTracker:
    """Lightweight eye‑landmark tracker using Mediapipe FaceMesh with EMA smoothing."""

    def __init__(self, alpha: float = 0.4) -> None:
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self.alpha = alpha
        self.left_eye: tuple[int, int] | None = None
        self.right_eye: tuple[int, int] | None = None

    def get_eyes(self, frame_bgr: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Return (left_ey, right_eye) pixel coords or `None` if no face."""
        try:
            res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:  # Mediapipe can throw when inputs are unexpected
            logging.error("Mediapipe processing failed: %s", e)
            return None
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        le = (int(lm[33].x * w), int(lm[33].y * h))
        re = (int(lm[263].x * w), int(lm[263].y * h))

        # Exponential moving average for stability
        if self.left_eye is None or self.right_eye is None:
            self.left_eye, self.right_eye = le, re
        else:
            self.left_eye = (
                int(self.alpha * le[0] + (1 - self.alpha) * self.left_eye[0]),
                int(self.alpha * le[1] + (1 - self.alpha) * self.left_eye[1]),
            )
            self.right_eye = (
                int(self.alpha * re[0] + (1 - self.alpha) * self.right_eye[0]),
                int(self.alpha * re[1] + (1 - self.alpha) * self.right_eye[1]),
            )
        return self.left_eye, self.right_eye

class LatentSelf:
    """Main application class.

    Attributes:
        active_direction: The currently active morphing direction.
        config: The application configuration.
        camera_index: The index of the camera to use.
        resolution: The resolution of the camera feed.
        cap: The OpenCV VideoCapture object.
        camera_available: Whether the camera is available.
        device: The device to use for processing (CPU or CUDA).
        weights_dir: The directory where the model weights are stored.
        ui: The UI backend to use (cv2 or qt).
        REENCODE_INTERVAL_S: The interval at which to re-encode the face.
        G: The StyleGAN generator.
        E: The e4e encoder.
        latent_dirs: The latent directions.
        stop_event: An event to stop the processing thread.
        _processing_thread: The processing thread.
        _last_affine: The last affine transformation matrix.
        tracker: The eye tracker.
        _canonical: The canonical eye points.
        mqtt_client: The MQTT client.
        mqtt_topic: The MQTT topic to publish to.
        mqtt_heartbeat_interval: The interval at which to send MQTT heartbeats.
        _last_mqtt_heartbeat: The time of the last MQTT heartbeat.
    """
    active_direction: str

    def __init__(
        self,
        config: Config,
        camera_index: int,
        resolution: int,
        device: str,
        weights_dir: Path,
        ui: str,
    ) -> None:
        """Initializes the LatentSelf class.

        Args:
            config: The application configuration.
            camera_index: The index of the camera to use.
            resolution: The resolution of the camera feed.
            device: The device to use for processing (CPU or CUDA).
            weights_dir: The directory where the model weights are stored.
            ui: The UI backend to use (cv2 or qt).
        """
        self.config = config
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera at index {camera_index}")
            self.camera_available = False
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution)
            self.camera_available = True
        self.device = torch.device(device)
        self.weights_dir = weights_dir
        self.ui = ui
        self.REENCODE_INTERVAL_S = 10.0 # Add this line
        self.active_direction = "BLEND"


        # Pre-load models (lazy wrapper ensures each loads once)
        self.G = get_stylegan_generator(weights_dir).to(self.device)
        self.E = get_e4e_encoder(weights_dir).to(self.device)
        self.latent_dirs = get_latent_directions(weights_dir)
        self.check_orthogonality()

        self.stop_event = Event()
        self._processing_thread: Thread | VideoWorker | None = None
        self._last_affine: np.ndarray | None = None
        # Face tracker for stabilised alignment
        self.tracker = _EyeTracker(alpha=self.config.data['tracker_alpha'])
        # Canonical eye points for 256×256 aligned crop
        self._canonical = np.array([[80.0, 100.0], [176.0, 100.0]], dtype=np.float32)  # left, right eye in aligned space

        self._apply_config()
        self._setup_mqtt()

    def _setup_mqtt(self) -> None:
        if not MQTT_AVAILABLE or not self.config.data.get('mqtt', {}).get('enabled'):
            self.mqtt_client = None
            return

        broker = self.config.data['mqtt']['broker']
        port = self.config.data['mqtt']['port']
        topic_namespace = self.config.data['mqtt']['topic_namespace']
        device_id = self.config.data['mqtt'].get('device_id') or socket.gethostname()

        self.mqtt_client = mqtt.Client(client_id=device_id)
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

        try:
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()  # Start a non-blocking loop
            logging.info(f"MQTT: Connected to {broker}:{port}")
        except Exception as e:
            logging.error(f"MQTT: Could not connect to broker {broker}:{port} - {e}")
            self.mqtt_client = None

        self.mqtt_topic = f"{topic_namespace}/{device_id}/heartbeat"
        self.mqtt_heartbeat_interval = self.config.data['mqtt']['heartbeat_interval']
        self._last_mqtt_heartbeat = 0.0

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("MQTT: Successfully connected to broker.")
        else:
            logging.error(f"MQTT: Connection failed with code {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        logging.warning(f"MQTT: Disconnected with code {rc}")

    def _send_mqtt_heartbeat(self) -> None:
        if self.mqtt_client and (time() - self._last_mqtt_heartbeat) > self.mqtt_heartbeat_interval:
            try:
                payload = {"timestamp": time(), "status": "alive"}
                self.mqtt_client.publish(self.mqtt_topic, str(payload))
                self._last_mqtt_heartbeat = time()
                logging.debug(f"MQTT: Sent heartbeat to {self.mqtt_topic}")
            except Exception as e:
                logging.error(f"MQTT: Failed to send heartbeat - {e}")

    def _apply_config(self) -> None:
        """Apply configuration settings to the application's attributes."""
        self.cycle_seconds = self.config.data['cycle_duration']
        self.blend_weights = {
            k.upper(): v for k, v in self.config.data['blend_weights'].items()
        }
        if 'SPECIES' not in self.blend_weights:  # Ensure species is present even if not in config
            self.blend_weights['SPECIES'] = 0.0
        self.tracker.alpha = self.config.data['tracker_alpha']
        self.max_magnitudes = {
            k.upper(): v['max_magnitude'] for k, v in self.config.directions_data.items()
        }

    # ------------------------------------------------------------------
    # Latent utilities
    # ------------------------------------------------------------------

    def check_orthogonality(self):
        """Checks the orthogonality of the latent directions."""
        offs = np.stack([self.latent_dirs['AGE'], self.latent_dirs['GENDER'], self.latent_dirs['SMILE']])
        orth = np.dot(offs, offs.T)
        logging.info("Orthogonality check:")
        logging.info(np.round(orth, 2))

    def encode_face(self, frame: np.ndarray) -> torch.Tensor:
        """Detect eyes, align, crop to 256×256, encode to W+ latent.

        Args:
            frame: The input frame.

        Returns:
            The encoded latent vector.
        """
        eyes = self.tracker.get_eyes(frame)
        if eyes is None:
            # Fallback: centre‑crop resize if no face
            crop = cv2.resize(frame, (256, 256))
            M = None
        else:
            le, re = eyes
            M = cv2.getAffineTransform(np.array([le, re], dtype=np.float32), self._canonical)
            crop = cv2.warpAffine(frame, M, (256, 256))
        latent, _ = self.E(_to_tensor(crop).to(self.device), return_latents=True)
        # store last transform for inverse warp
        self._last_affine = M
        return latent

    def decode_latent(self, latent_w_plus: torch.Tensor, target_shape: tuple[int, int]) -> np.ndarray:
        """Decode latent and inverse‑warp into the original frame size.

        Args:
            latent_w_plus: The latent vector to decode.
            target_shape: The shape of the target frame.

        Returns:
            The decoded frame.
        """
        with torch.no_grad():
            img_out, _, _ = self.G.synthesis(latent_w_plus, noise_mode='const')
        out = _to_numpy(img_out)
        if self._last_affine is not None:
            H, W = target_shape
            inv = cv2.invertAffineTransform(self._last_affine)
            out_full = cv2.warpAffine(out, inv, (W, H), flags=cv2.WARP_INVERSE_MAP)
            return out_full
        return cv2.resize(out, target_shape[::-1])

    # ------------------------------------------------------------------
    # Morph schedule
    # ------------------------------------------------------------------

    def latent_offset(self, t: float) -> tuple[np.ndarray, float]:
        """Return a composite latent offset for time *t* and the current magnitude.

        Args:
            t: The current time.

        Returns:
            A tuple containing the latent offset and the current magnitude.
        """
        phase = (t % self.cycle_seconds) / self.cycle_seconds
        raw_amt = 1.0 - abs(phase * 2.0 - 1.0)  # Triangle wave: 0 → 1 → 0

        if self.active_direction == "BLEND":
            # Filter blend weights to only include directions that actually exist
            valid_weights = {k: v for k, v in self.blend_weights.items() if k in self.latent_dirs}
            total_weight = sum(valid_weights.values()) or 1.0
            
            direction = sum(
                (w / total_weight) * self.latent_dirs[k]
                for k, w in valid_weights.items()
            )
            # For blend mode, we don't have a single max_magnitude, so we use a default or average
            max_mag = 3.0 # Default max magnitude for blend mode
        else: # A specific direction is active
            if self.active_direction not in self.latent_dirs:
                logging.warning(f"Direction '{self.active_direction}' not found in latent directions. Using zero offset.")
                return np.zeros(512), 0.0
            direction = self.latent_dirs[self.active_direction]
            max_mag = self.max_magnitudes.get(self.active_direction, 3.0) # Default to 3.0 if not found

        current_magnitude = raw_amt * max_mag
        return current_magnitude * direction, current_magnitude

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def _process_stream(self, frame_emitter: pyqtSignal | None = None) -> None:
        """Runs in a background thread, processing and displaying video frames."""
        baseline_latent: torch.Tensor | None = None
        last_encode = 0.0
        idle_frames = 0
        idle_threshold = int(self.config.data.get('idle_seconds', 3) * self.config.data['fps'])
        fade_frames = int(self.config.data.get('idle_fade_frames', self.config.data['fps']))

        while not self.stop_event.is_set():
            if not self.camera_available:
                logging.error("Camera not available. Displaying error screen and retrying...")
                # Display a black screen with an error message
                h, w = self.resolution, self.resolution # Assuming square resolution
                error_frame = np.zeros((h, w, 3), dtype=np.uint8)
                error_text = "Camera Not Available"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(error_text, font, 1, 2)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(error_frame, error_text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if self.ui == 'qt' and frame_emitter:
                    frame_emitter.emit(_numpy_to_qimage(error_frame))
                else:
                    cv2.imshow("Latent Self", error_frame)
                    cv2.waitKey(1) # Needed to update the window

                self.stop_event.wait(5) # Wait for 5 seconds before retrying
                # Attempt to re-initialize camera
                self.cap = cv2.VideoCapture(self.camera_index) # Assuming camera_index is stored
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution)
                    self.camera_available = True
                    logging.info("Camera re-initialized successfully.")
                continue

            ret, frame = self.cap.read()
            if not ret:
                logging.error("Camera read failed – aborting")
                self.camera_available = False # Mark camera as unavailable
                continue

            now = time()
            if baseline_latent is None or (now - last_encode) > self.REENCODE_INTERVAL_S:
                baseline_latent = self.encode_face(frame)
                last_encode = now
                logging.info("Encoded new baseline latent.")

            offset, current_magnitude = self.latent_offset(now)
            latent_mod = baseline_latent + torch.from_numpy(offset).to(self.device)
            out_frame = self.decode_latent(latent_mod, (frame.shape[0], frame.shape[1]))

            # Draw tracked eyes for debug
            eyes = self.tracker.left_eye, self.tracker.right_eye
            if eyes[0] is not None and eyes[1] is not None:
                idle_frames = 0
                cv2.circle(out_frame, eyes[0], 3, (0,255,0), -1)
                cv2.circle(out_frame, eyes[1], 3, (0,255,0), -1)
            else:
                idle_frames += 1

            cv2.putText(
                out_frame,
                f"Mode: {self.active_direction.capitalize()} ({current_magnitude:+.1f})",
                (10, out_frame.shape[0] - 20), # Position at bottom-left
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

            if idle_frames > idle_threshold:
                alpha = min(1.0, (idle_frames - idle_threshold) / fade_frames)
                overlay = out_frame.copy()
                overlay[:] = (0, 0, 0)
                cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0, out_frame)

            if self.ui == 'qt' and frame_emitter:
                # Update HUD if it exists
                if hasattr(self, 'window') and hasattr(self.window, 'direction_uis'):
                    for name, ui_elements in self.window.direction_uis.items():
                        if name.upper() == self.active_direction:
                            value = current_magnitude
                        else:
                            value = 0.0
                        ui_elements.label.setText(f"{name}: {value:+.1f}")

                frame_emitter.emit(_numpy_to_qimage(out_frame))
                QThread.msleep(int(1000 / self.config.data['fps']))
            else:
                cv2.imshow("Latent Self", out_frame)
                key = cv2.waitKey(int(1000 / self.config.data['fps'])) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("y"):
                    self.active_direction = "AGE"
                elif key == ord("g"):
                    self.active_direction = "GENDER"
                elif key == ord("h"):
                    self.active_direction = "SMILE"
                elif key == ord("s"):
                    self.active_direction = "SPECIES"
                elif key == ord("b"):
                    self.active_direction = "BLEND"

            self._send_mqtt_heartbeat()

        self.cap.release()
        if self.ui == 'cv2':
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the application."""
        logging.info("Starting Latent Self…")
        if self.ui == 'qt':
            if not QT_AVAILABLE:
                logging.error("Qt UI requested, but PyQt6 is not installed. Please run: pip install PyQt6")
                sys.exit(1)
            self._run_qt()
        else:
            self._run_cv2()
        logging.info("Application shut down gracefully.")

    def _run_cv2(self) -> None:
        logging.info("Using cv2 UI. Controls: [q]uit | [y]age | [g]ender | [h]smile | [s]pecies | [b]lend")
        self._processing_thread = Thread(target=self._process_stream, daemon=True)
        self._processing_thread.start()
        self._processing_thread.join() # Wait for thread to finish (e.g. on 'q' press)

    def _run_qt(self) -> None:
        from PyQt6.QtWidgets import QApplication

        logging.info("Using Qt UI. Controls: [Esc] or [Q] to quit.")
        app = QApplication(sys.argv)
        self.window = MirrorWindow(self)

        

        self._processing_thread = VideoWorker(self)
        self._processing_thread.new_frame.connect(self.window.update_frame)
        self._processing_thread.start()

        self.window.show_fullscreen()
        app.exec()

        self.stop_event.set()
        self._processing_thread.wait()


if QT_AVAILABLE:
    class VideoWorker(QThread):
        """QThread worker for video processing."""
        new_frame = pyqtSignal(QImage)

        def __init__(self, app: "LatentSelf"):
            super().__init__()
            self.app = app

        def run(self):
            self.app._process_stream(self.new_frame)


# -------------------------------------------------------------------------------------------------
# CLI entry-point ---------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Latent Self interactive mirror",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--resolution", type=int, default=512, help="Square frame size (px)")
    parser.add_argument("--fps", type=int, default=None, help="Target frames per second (overrides config)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--weights", type=Path, default=asset_path("models"), help="Directory for model weights")
    parser.add_argument("--ui", type=str, default="cv2", choices=["cv2", "qt"], help="UI backend to use")

    g = parser.add_argument_group("Morphing Controls (overrides config)")
    g.add_argument("--cycle-duration", type=float, default=None, help="Duration of one morph cycle (seconds)")
    g.add_argument("--blend-age", type=float, default=None, help="Weight for age in blended mode")
    g.add_argument("--blend-gender", type=float, default=None, help="Weight for gender in blended mode")
    g.add_argument("--blend-smile", type=float, default=None, help="Weight for smile in blended mode")
    g.add_argument("--blend-species", type=float, default=None, help="Weight for species in blended mode")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = Config(args)

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
    )
    config.app = app
    app.run()


if __name__ == "__main__":
    main()
