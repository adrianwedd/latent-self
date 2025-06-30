"""Supporting service classes used by the main application logic."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread, Lock
from queue import SimpleQueue
from typing import Callable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtGui import QImage

from directions import Direction
from logging_setup import log_timing
from time import time
from typing import Any, Dict

from pydantic import ValidationError
from config_schema import AppConfig, CLIOverrides, DirectionsConfig

import appdirs
import cv2
import numpy as np
import torch
import yaml

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - runtime fail hard
    raise RuntimeError(
        "mediapipe is required for face tracking. Install it with 'pip install mediapipe'."
    ) from exc

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional feature
    MQTT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def asset_path(relative_path: str) -> Path:
    """Get the absolute path to an asset, handling PyInstaller bundling."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return base_path / relative_path


# ---------------------------------------------------------------------------
# Configuration manager
# ---------------------------------------------------------------------------

class ConfigManager:
    """Load and manage configuration YAML files."""

    def __init__(self, cli_args: argparse.Namespace, app: Any | None = None) -> None:
        """Initialize the manager and load configuration files.

        Args:
            cli_args: Parsed command line arguments used for overrides.
            app: Optional application instance to notify on reloads.
        """
        self.config_dir = Path(appdirs.user_config_dir("LatentSelf"))
        self.config_path = self.config_dir / "config.yaml"
        self.directions_path = self.config_dir / "directions.yaml"
        self._ensure_config_exists()
        self._ensure_directions_exists()
        self.data: Dict[str, Any] = {}
        self.directions_data: Dict[str, Any] = {}
        self.app = app
        self.reload()
        self._override_with_cli(cli_args)

    def _ensure_config_exists(self) -> None:
        """Create a default ``config.yaml`` if one is missing."""
        if not self.config_path.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            default_config = asset_path("data/config.yaml")
            self.config_path.write_bytes(default_config.read_bytes())
            logging.info(f"Created default config at {self.config_path}")

    def _ensure_directions_exists(self) -> None:
        """Create ``directions.yaml`` if it does not exist."""
        if not self.directions_path.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            default_directions = asset_path("data/directions.yaml")
            self.directions_path.write_bytes(default_directions.read_bytes())
            logging.info(f"Created default directions at {self.directions_path}")

    def _override_with_cli(self, args: argparse.Namespace) -> None:
        """Merge CLI overrides using the CLIOverrides model."""
        overrides = CLIOverrides(**vars(args))

        if overrides.cycle_duration is not None:
            self.data["cycle_duration"] = overrides.cycle_duration
        if overrides.blend_age is not None:
            self.data.setdefault("blend_weights", {})["age"] = overrides.blend_age
        if overrides.blend_gender is not None:
            self.data.setdefault("blend_weights", {})["gender"] = overrides.blend_gender
        if overrides.blend_smile is not None:
            self.data.setdefault("blend_weights", {})["smile"] = overrides.blend_smile
        if overrides.blend_species is not None:
            self.data.setdefault("blend_weights", {})["species"] = overrides.blend_species
        if overrides.fps is not None:
            self.data["fps"] = overrides.fps

        if overrides.max_cpu_mem_mb is not None:
            self.data["max_cpu_mem_mb"] = overrides.max_cpu_mem_mb
        if overrides.max_gpu_mem_gb is not None:
            self.data["max_gpu_mem_gb"] = overrides.max_gpu_mem_gb
        try:
            self.data = AppConfig(**self.data).dict()
        except ValidationError as e:
            raise RuntimeError(f"Invalid configuration after CLI overrides: {e}")

    def reload(self) -> None:
        """Reload configuration files with validation."""
        try:
            with self.config_path.open("r") as f:
                raw_cfg = yaml.safe_load(f) or {}
            cfg = AppConfig(**raw_cfg)
            self.data = cfg.dict()
        except (yaml.YAMLError, ValidationError) as e:
            raise RuntimeError(f"Invalid config.yaml: {e}")

        try:
            with self.directions_path.open("r") as f:
                raw_dir = yaml.safe_load(f) or {}
            dirs = DirectionsConfig(__root__=raw_dir)
            self.directions_data = dirs.to_dict()
        except (yaml.YAMLError, ValidationError) as e:
            raise RuntimeError(f"Invalid directions.yaml: {e}")

        logging.info("Configuration reloaded.")
        if self.app:
            self.app._apply_config()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


T = TypeVar("T")


def _lazy_once(fn: Callable[..., T]) -> Callable[..., T]:
    """Cache function results and return the same value on subsequent calls."""

    cache: Dict[Callable[..., T], T] = {}

    def wrapper(*args: Any, **kwargs: Any) -> T:
        if fn not in cache:
            cache[fn] = fn(*args, **kwargs)
        return cache[fn]

    return wrapper


@_lazy_once
def get_stylegan_generator(weights_dir: Path):
    """Load the StyleGAN generator weights once and cache the instance."""
    import pickle  # noqa: WPS433 - model loading

    gen_path = weights_dir / "ffhq-1024-stylegan2.pkl"
    if not gen_path.exists():
        logging.error("StyleGAN generator not found at %s", gen_path)
        sys.exit(1)
    logging.info("Loading StyleGAN2 generator from %s", gen_path)
    with gen_path.open("rb") as fp:
        generator = pickle.load(fp)["G_ema"].eval()
    return generator


@_lazy_once
def get_e4e_encoder(weights_dir: Path):
    """Load the e4e encoder weights once and cache the instance."""
    try:
        from models.encoders import pSp  # imported late
    except ModuleNotFoundError:
        logging.error("Module 'pSp' not found. Ensure the e4e repo is on PYTHONPATH.")
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
    """Load latent direction vectors from disk."""
    path = weights_dir / "latent_directions.npz"
    if not path.exists():
        logging.error("Latent directions file not found at %s", path)
        sys.exit(1)
    logging.info("Loading latent directions from %s", path)
    with np.load(path) as data:
        dirs = {k.upper(): data[k] for k in data.keys()}

    beauty_path = weights_dir / "beauty.npy"
    if beauty_path.exists():
        logging.info("Loading Beauty direction from %s", beauty_path)
        vec = np.load(beauty_path)
        norm = np.linalg.norm(vec)
        if norm:
            dirs[Direction.BEAUTY.value] = vec / norm

    return dirs


class ModelManager:
    """Manage ML models used by the application."""

    def __init__(self, weights_dir: Path, device: torch.device) -> None:
        """Load models and latent directions.

        Args:
            weights_dir: Directory containing the model weights.
            device: Torch device to map the models onto.
        """
        self.weights_dir = weights_dir
        self.device = device
        self.G = get_stylegan_generator(weights_dir).to(device)
        self.E = get_e4e_encoder(weights_dir).to(device)
        self.latent_dirs = get_latent_directions(weights_dir)

    def check_orthogonality(self) -> None:
        """Log dot products of key latent directions for debugging."""

        offs = np.stack([
            self.latent_dirs[Direction.AGE.value],
            self.latent_dirs[Direction.GENDER.value],
            self.latent_dirs.get(
                Direction.SMILE.value,
                np.zeros_like(self.latent_dirs[Direction.AGE.value]),
            ),
            self.latent_dirs.get(
                Direction.BEAUTY.value,
                np.zeros_like(self.latent_dirs[Direction.AGE.value]),
            ),
        ])
        orth = np.dot(offs, offs.T)
        logging.info("Orthogonality check:\n%s", np.round(orth, 2))


    def unload(self) -> None:
        """Unload models from memory."""
        self.G = None
        self.E = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

class _EyeTracker:
    """Lightweight eye-landmark tracker using Mediapipe."""

    def __init__(self, alpha: float = 0.4) -> None:
        """Create a new tracker instance.

        Args:
            alpha: Smoothing factor between 0 and 1 for landmark movement.
        """
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self.alpha = alpha
        self.left_eye: tuple[int, int] | None = None
        self.right_eye: tuple[int, int] | None = None

    def get_eyes(self, frame_bgr: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Return smoothed eye coordinates from a BGR frame.

        Args:
            frame_bgr: Current video frame in BGR format.

        Returns:
            Tuple of ``(left_eye, right_eye)`` pixel coordinates if a face is
            detected, otherwise ``None``.
        """
        try:
            res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:  # mediapipe can throw
            logging.error("Mediapipe processing failed: %s", e)
            return None
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        le = (int(lm[33].x * w), int(lm[33].y * h))
        re = (int(lm[263].x * w), int(lm[263].y * h))

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


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert a BGR image ``ndarray`` to a normalized PyTorch tensor."""

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor.unsqueeze(0)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to a BGR image."""

    tensor = (tensor.squeeze(0).clamp(-1, 1) + 1.0) * 127.5
    img_rgb = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _numpy_to_qimage(img: np.ndarray) -> "QImage":
    """Convert a ``numpy`` image to a ``QImage`` for Qt display."""

    from PyQt6.QtGui import QImage

    h, w, ch = img.shape
    bytes_per_line = ch * w
    return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)


class VideoProcessor:
    """Encapsulate the video capture and morphing pipeline."""

    def __init__(
        self,
        model_manager: ModelManager,
        config: ConfigManager,
        device: torch.device,
        camera_index: int,
        resolution: int,
        ui: str,
        telemetry: "TelemetryClient | None" = None,
    ) -> None:
        """Create a new processor instance.

        Args:
            model_manager: Provider of GAN and encoder models.
            config: Configuration manager with runtime settings.
            device: Torch device for heavy computation.
            camera_index: Index of the webcam to open.
            resolution: Square frame size in pixels.
            ui: ``"cv2"`` or ``"qt"`` user interface backend.
            telemetry: Optional telemetry client for heartbeats.
        """
        self.model_manager = model_manager
        self.config = config
        self.device = device
        self.camera_index = camera_index
        self.resolution = resolution
        self.ui = ui
        self.telemetry = telemetry

        self.cap: cv2.VideoCapture | None = None
        self.camera_available = False

        self.tracker = _EyeTracker(alpha=self.config.data.get("tracker_alpha", 0.4))
        self.tracker_lock = Lock()
        self._canonical = np.array([[80.0, 100.0], [176.0, 100.0]], dtype=np.float32)
        self.stop_event = Event()
        self._processing_thread: Thread | None = None
        self.REENCODE_INTERVAL_S = 10.0
        self._direction_lock = Lock()
        self.command_queue: SimpleQueue[Direction] = SimpleQueue()
        self.active_direction = Direction.BLEND
        self._last_affine: np.ndarray | None = None

        self._apply_config()

    # Configuration from config manager
    def _apply_config(self) -> None:
        """Apply the latest configuration values to runtime state."""
        self.cycle_seconds = self.config.data["cycle_duration"]
        self.blend_weights = {Direction.from_str(k).value: v for k, v in self.config.data["blend_weights"].items()}
        self.max_magnitudes = {Direction.from_str(k).value: v["max_magnitude"] for k, v in self.config.directions_data.items()}
        self.direction_labels = {Direction.from_str(k).value: v.get("label", k.capitalize()) for k, v in self.config.directions_data.items()}
        self._hud_values = {k: None for k in self.direction_labels}
        with self.tracker_lock:
            self.tracker.alpha = self.config.data.get("tracker_alpha", 0.4)

    # ------------------------------------------------------------------
    # Core latent helpers
    # ------------------------------------------------------------------
    def encode_face(self, frame: np.ndarray) -> torch.Tensor:
        """Encode the current frame into latent ``w+`` space."""
        with log_timing("encode_face"):
            with self.tracker_lock:
                eyes = self.tracker.get_eyes(frame)
            if eyes is None:
                crop = cv2.resize(frame, (256, 256))
                M = None
            else:
                le, re = eyes
                M = cv2.getAffineTransform(np.array([le, re], dtype=np.float32), self._canonical)
                crop = cv2.warpAffine(frame, M, (256, 256))
            latent, _ = self.model_manager.E(_to_tensor(crop).to(self.device), return_latents=True)
            self._last_affine = M
            return latent

    def decode_latent(self, latent_w_plus: torch.Tensor, target_shape: tuple[int, int]) -> np.ndarray:
        """Decode a latent tensor back into an image matching ``target_shape``."""

        with log_timing("decode_latent"):
            with torch.no_grad():
                img_out, _, _ = self.model_manager.G.synthesis(latent_w_plus, noise_mode="const")
            out = _to_numpy(img_out)
            if self._last_affine is not None:
                H, W = target_shape
                inv = cv2.invertAffineTransform(self._last_affine)
                out_full = cv2.warpAffine(out, inv, (W, H), flags=cv2.WARP_INVERSE_MAP)
                return out_full
            return cv2.resize(out, target_shape[::-1])

    def latent_offset(self, t: float) -> tuple[np.ndarray, float]:
        """Compute the latent direction offset at time ``t``."""

        phase = (t % self.cycle_seconds) / self.cycle_seconds
        raw_amt = 1.0 - abs(phase * 2.0 - 1.0)

        with self._direction_lock:
            active = self.active_direction

        if active is Direction.BLEND:
            valid_weights = {k: v for k, v in self.blend_weights.items() if k in self.model_manager.latent_dirs}
            total_weight = sum(valid_weights.values()) or 1.0
            direction = sum((w / total_weight) * self.model_manager.latent_dirs[k] for k, w in valid_weights.items())
            max_mag = 3.0
        else:
            if active.value not in self.model_manager.latent_dirs:
                logging.warning("Direction '%s' not found in latent directions", active.value)
                return np.zeros(512), 0.0
            direction = self.model_manager.latent_dirs[active.value]
            max_mag = self.max_magnitudes.get(active.value, 3.0)

        current_magnitude = raw_amt * max_mag
        return current_magnitude * direction, current_magnitude

    def _send_mqtt_heartbeat(self) -> None:
        if self.telemetry:
            self.telemetry.send_heartbeat()

    def _process_stream(self, frame_emitter: "pyqtSignal" | None = None) -> None:
        """Main loop reading camera frames and emitting processed output."""

        from PyQt6.QtCore import QThread

        self.cap = cv2.VideoCapture(self.camera_index)
        self.camera_available = self.cap.isOpened()
        if self.camera_available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution)
        else:
            logging.error("Failed to open camera at index %s", self.camera_index)

        baseline_latent: torch.Tensor | None = None
        last_encode = 0.0
        idle_frames = 0
        idle_threshold = int(self.config.data.get("idle_seconds", 3) * self.config.data["fps"])
        fade_frames = int(self.config.data.get("idle_fade_frames", self.config.data["fps"]))
        retry_delay = 1.0
        max_delay = 30.0

        try:
            while not self.stop_event.is_set():
                self._drain_direction_queue()
                if not self.camera_available:
                    logging.error("Camera not available. Displaying error screen and retrying...")
                    h, w = self.resolution, self.resolution
                    error_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    error_text = "Camera Not Available"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(error_text, font, 1, 2)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(error_frame, error_text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    if self.ui == "qt" and frame_emitter:
                        frame_emitter.emit(_numpy_to_qimage(error_frame))
                    else:
                        cv2.imshow("Latent Self", error_frame)
                        cv2.waitKey(1)

                    self.stop_event.wait(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
                    self.cap = cv2.VideoCapture(self.camera_index)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution)
                        self.camera_available = True
                        retry_delay = 1.0
                        logging.info("Camera re-initialized successfully.")
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Camera read failed – retrying")
                    self.camera_available = False
                    continue

                now = time()
                if baseline_latent is None or (now - last_encode) > self.REENCODE_INTERVAL_S:
                    baseline_latent = self.encode_face(frame)
                    last_encode = now
                    logging.info("Encoded new baseline latent.")

                offset, current_magnitude = self.latent_offset(now)
                latent_mod = baseline_latent + torch.from_numpy(offset).to(self.device)
                out_frame = self.decode_latent(latent_mod, (frame.shape[0], frame.shape[1]))

                eyes = self.tracker.left_eye, self.tracker.right_eye
                if eyes[0] is not None and eyes[1] is not None:
                    idle_frames = 0
                    cv2.circle(out_frame, eyes[0], 3, (0, 255, 0), -1)
                    cv2.circle(out_frame, eyes[1], 3, (0, 255, 0), -1)
                else:
                    idle_frames += 1

                with self._direction_lock:
                    mode = self.active_direction
                mode_label = self.direction_labels.get(mode.value, mode.value.capitalize())
                cv2.putText(
                    out_frame,
                    f"Mode: {mode_label.capitalize()} ({current_magnitude:+.1f})",
                    (10, out_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if idle_frames > idle_threshold:
                    alpha = min(1.0, (idle_frames - idle_threshold) / fade_frames)
                    overlay = out_frame.copy()
                    overlay[:] = (0, 0, 0)
                    cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0, out_frame)

                if self.ui == "qt" and frame_emitter:
                    frame_emitter.emit(_numpy_to_qimage(out_frame))
                    QThread.msleep(int(1000 / self.config.data["fps"]))
                else:
                    cv2.imshow("Latent Self", out_frame)
                    key = cv2.waitKey(int(1000 / self.config.data["fps"])) & 0xFF
                    if key == ord("q"):
                        self.stop_event.set()
                        break
                    else:
                        ch = chr(key).lower()
                        direction = Direction.from_key(ch)
                        if direction:
                            with self._direction_lock:
                                self.active_direction = direction

                    self._send_mqtt_heartbeat()
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.ui == "cv2":
                cv2.destroyAllWindows()

    # Public control -----------------------------------------------------
    def start(self, frame_emitter: "pyqtSignal" | None = None) -> None:
        """Begin processing the camera stream in a background thread."""
        self.stop_event.clear()
        self._processing_thread = Thread(target=self._process_stream, args=(frame_emitter,), daemon=True)
        self._processing_thread.start()

    def join(self) -> None:
        """Block until the processing thread exits."""
        if self._processing_thread is not None:
            self._processing_thread.join()

    def stop(self) -> None:
        """Signal the processing thread to stop and wait for shutdown."""
        self.stop_event.set()
        self.join()

    # Direction control -------------------------------------------------
    def enqueue_direction(self, direction: Direction | str) -> None:
        """Request a change of active direction from another thread."""
        if isinstance(direction, str):
            direction = Direction.from_str(direction)
        self.command_queue.put(direction)

    def _drain_direction_queue(self) -> None:
        """Apply any queued direction changes."""
        while not self.command_queue.empty():
            new_dir = self.command_queue.get_nowait()
            with self._direction_lock:
                self.active_direction = new_dir

    def get_active_direction(self) -> Direction:
        """Return the currently active morphing direction."""
        with self._direction_lock:
            return self.active_direction


# ---------------------------------------------------------------------------
# Telemetry / MQTT
# ---------------------------------------------------------------------------

class TelemetryClient:
    """Lightweight MQTT heartbeat publisher."""

    def __init__(self, config: ConfigManager) -> None:
        """Connect to the broker if telemetry is enabled."""
        if not MQTT_AVAILABLE or not config.data.get("mqtt", {}).get("enabled"):
            self.client = None
            return

        broker = config.data["mqtt"]["broker"]
        port = config.data["mqtt"]["port"]
        topic_namespace = config.data["mqtt"]["topic_namespace"]
        device_id = config.data["mqtt"].get("device_id") or platform.node()

        self.client = mqtt.Client(client_id=device_id)
        self.client.on_connect = lambda c, u, f, rc: logging.info("MQTT connected" if rc == 0 else f"MQTT connect failed {rc}")
        self.client.on_disconnect = lambda c, u, rc: logging.warning(f"MQTT disconnected {rc}")
        try:
            self.client.connect(broker, port, 60)
            self.client.loop_start()
            logging.info("MQTT: Connected to %s:%s", broker, port)
        except Exception as e:  # pragma: no cover - network
            logging.warning("MQTT: Could not connect to broker %s:%s - %s", broker, port, e)
            self.client = None

        self.topic = f"{topic_namespace}/{device_id}/heartbeat"
        self.interval = config.data["mqtt"]["heartbeat_interval"]
        self._last_ts = 0.0

    def send_heartbeat(self) -> None:
        """Publish a heartbeat message if the interval has elapsed."""

        if self.client and (time() - self._last_ts) > self.interval:
            try:
                payload = {"timestamp": time(), "status": "alive"}
                self.client.publish(self.topic, str(payload))
                self._last_ts = time()
                logging.debug("MQTT: Sent heartbeat")
            except Exception as e:
                logging.warning("MQTT: Failed to send heartbeat - %s", e)

    def shutdown(self) -> None:
        """Disconnect cleanly from the MQTT broker."""

        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                logging.warning("MQTT cleanup failed: %s", e)

class MemoryMonitor:
    """Background memory usage monitor."""

    def __init__(self, config: ConfigManager) -> None:
        """Create a memory monitor from configuration values."""
        self.interval = config.data.get("memory_check_interval", 10)
        self.max_cpu_mb = config.data.get("max_cpu_mem_mb")
        self.max_gpu_gb = config.data.get("max_gpu_mem_gb")
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """Start background monitoring if enabled."""

        if self.interval <= 0:
            return
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Periodically log memory usage statistics."""

        import psutil
        proc = psutil.Process()
        while not self._stop.is_set():
            cpu_mb = proc.memory_info().rss / (1024 * 1024)
            gpu_gb = 0.0
            if torch.cuda.is_available():
                gpu_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            if self.max_cpu_mb and cpu_mb > self.max_cpu_mb:
                logging.warning("CPU memory usage %.1f MB exceeds limit %s", cpu_mb, self.max_cpu_mb)
            if self.max_gpu_gb and gpu_gb > self.max_gpu_gb:
                logging.warning("GPU memory usage %.2f GB exceeds limit %s", gpu_gb, self.max_gpu_gb)
            logging.debug("Memory usage: CPU %.1f MB | GPU %.2f GB", cpu_mb, gpu_gb)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        """Stop the monitoring thread."""

        if not self._stop.is_set():
            self._stop.set()
            if self._thread:
                self._thread.join()

