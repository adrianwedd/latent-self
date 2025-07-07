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
from typing import Callable, TypeVar, TYPE_CHECKING, Sequence
from urllib.parse import urlparse
import os

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtGui import QImage

from directions import Direction
from logging_setup import FrameTimer, log_timing
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
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    trt = None
    cuda = None

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
    mqtt = None
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


def select_torch_device(preference: str = "auto") -> torch.device:
    """Return a torch.device respecting GPU availability.

    Args:
        preference: ``"auto"``, ``"cpu"`` or ``"cuda"``.

    Returns:
        Resolved :class:`torch.device` instance.
    """

    pref = preference.lower()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logging.warning("CUDA requested but not available – falling back to CPU")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    logging.warning("CUDA not available – using CPU")
    return torch.device("cpu")


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
        self._lock = Lock()
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
        with self._lock:
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

            if overrides.gaze_mode is not None:
                self.data["gaze_mode"] = overrides.gaze_mode

            if overrides.max_cpu_mem_mb is not None:
                self.data["max_cpu_mem_mb"] = overrides.max_cpu_mem_mb
            if overrides.max_gpu_mem_gb is not None:
                self.data["max_gpu_mem_gb"] = overrides.max_gpu_mem_gb
            if overrides.emotion is not None:
                self.data["active_emotion"] = overrides.emotion.value
            if overrides.device is not None:
                self.data["device"] = overrides.device
            try:
                self.data = AppConfig(**self.data).model_dump()
            except ValidationError as e:
                raise RuntimeError(f"Invalid configuration after CLI overrides: {e}")

    def reload(self) -> None:
        """Reload configuration files with validation."""
        with self._lock:
            try:
                with self.config_path.open("r") as f:
                    raw_cfg = yaml.safe_load(f) or {}
                cfg = AppConfig(**raw_cfg)
                self.data = cfg.model_dump()
            except (yaml.YAMLError, ValidationError) as e:
                raise RuntimeError(f"Invalid config.yaml: {e}")

            try:
                with self.directions_path.open("r") as f:
                    raw_dir = yaml.safe_load(f) or {}
                dirs = DirectionsConfig(root=raw_dir)
                self.directions_data = dirs.model_dump()
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


class OnnxGenerator:
    """Wrapper for StyleGAN ONNX generator."""

    def __init__(self, path: Path, device: torch.device) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def synthesis(self, latent_w_plus: torch.Tensor, noise_mode: str = "const"):
        w = latent_w_plus.detach().cpu().numpy()
        out = self.session.run(None, {self.input_name: w})[0]
        img = torch.from_numpy(out)
        return img, None, None


class OnnxEncoder:
    """Wrapper for e4e ONNX encoder."""

    def __init__(self, path: Path, device: torch.device) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image: torch.Tensor, return_latents: bool = False):
        arr = image.detach().cpu().numpy()
        out = self.session.run(None, {self.input_name: arr})[0]
        tensor = torch.from_numpy(out)
        if return_latents:
            return tensor, None
        return tensor


class TRTModule:
    """Minimal TensorRT engine wrapper."""

    def __init__(self, path: Path) -> None:
        if trt is None or cuda is None:
            raise RuntimeError("tensorrt not installed")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with path.open("rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append((host_mem, device_mem))
            else:
                self.outputs.append((host_mem, device_mem))
        self.output_shape = self.engine.get_binding_shape(self.engine.num_bindings - 1)

    def run(self, array: np.ndarray) -> np.ndarray:
        inp_host, inp_dev = self.inputs[0]
        out_host, out_dev = self.outputs[0]
        np.copyto(inp_host, array.ravel())
        cuda.memcpy_htod_async(inp_dev, inp_host, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(out_host, out_dev, self.stream)
        self.stream.synchronize()
        return out_host.reshape(self.output_shape)


class TRTGenerator(TRTModule):
    """TensorRT StyleGAN generator."""

    def synthesis(self, latent_w_plus: torch.Tensor, noise_mode: str = "const"):
        out = self.run(latent_w_plus.detach().cpu().numpy())
        img = torch.from_numpy(out)
        return img, None, None


class TRTEncoder(TRTModule):
    """TensorRT e4e encoder."""

    def __call__(self, image: torch.Tensor, return_latents: bool = False):
        out = self.run(image.detach().cpu().numpy())
        tensor = torch.from_numpy(out)
        if return_latents:
            return tensor, None
        return tensor


def load_stylegan(weights_dir: Path, device: torch.device):
    engine_path = weights_dir / "stylegan2.engine"
    onnx_path = weights_dir / "stylegan2.onnx"
    if engine_path.exists() and trt is not None:
        logging.info("Loading TensorRT generator from %s", engine_path)
        return TRTGenerator(engine_path)
    if onnx_path.exists() and ort is not None:
        logging.info("Loading ONNX generator from %s", onnx_path)
        return OnnxGenerator(onnx_path, device)
    return get_stylegan_generator(weights_dir).to(device)


def load_e4e(weights_dir: Path, device: torch.device):
    engine_path = weights_dir / "e4e.engine"
    onnx_path = weights_dir / "e4e.onnx"
    if engine_path.exists() and trt is not None:
        logging.info("Loading TensorRT encoder from %s", engine_path)
        return TRTEncoder(engine_path)
    if onnx_path.exists() and ort is not None:
        logging.info("Loading ONNX encoder from %s", onnx_path)
        return OnnxEncoder(onnx_path, device)
    return get_e4e_encoder(weights_dir).to(device)


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
        self.model_load_failed = False
        self.error_message = ""
        try:
            self.G = load_stylegan(weights_dir, device)
        except Exception as e:  # noqa: BLE001 - runtime
            logging.exception("Failed to load StyleGAN: %s", e)
            self.G = None
            self.model_load_failed = True
            self.error_message += f"StyleGAN: {e}"
        try:
            self.E = load_e4e(weights_dir, device)
        except Exception as e:  # noqa: BLE001 - runtime
            logging.exception("Failed to load e4e encoder: %s", e)
            self.E = None
            self.model_load_failed = True
            if self.error_message:
                self.error_message += "; "
            self.error_message += f"e4e: {e}"
        try:
            self.latent_dirs = get_latent_directions(weights_dir)
        except Exception as e:  # noqa: BLE001 - runtime
            logging.exception("Failed to load latent directions: %s", e)
            self.latent_dirs = {}
            self.model_load_failed = True
            if self.error_message:
                self.error_message += "; "
            self.error_message += f"directions: {e}"

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

    def __init__(self, alpha: float = 0.4, canonical: Sequence[Sequence[float]] | None = None) -> None:
        """Create a new tracker instance.

        Args:
            alpha: Smoothing factor between 0 and 1 for landmark movement.
            canonical: Target eye coordinates for face alignment.
        """
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self.alpha = alpha
        self.canonical = np.array(canonical or [[80.0, 100.0], [176.0, 100.0]], dtype=np.float32)
        self.left_eye: tuple[int, int] | None = None
        self.right_eye: tuple[int, int] | None = None
        self.gaze_norm: tuple[float, float] | None = None

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
        self.gaze_norm = (
            ((self.left_eye[0] + self.right_eye[0]) / 2) / w,
            ((self.left_eye[1] + self.right_eye[1]) / 2) / h,
        )
        return self.left_eye, self.right_eye

    def get_gaze(self) -> tuple[float, float] | None:
        """Return last normalized gaze coordinates if available."""
        return self.gaze_norm


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
        demo: bool = False,
        low_power: bool = False,
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
            low_power: Enable adaptive resolution and frame skipping.
        """
        self.model_manager = model_manager
        self.config = config
        self.device = device
        self.camera_index = camera_index
        self.resolution = resolution
        self.ui = ui
        self.telemetry = telemetry
        self.demo = demo
        self.low_power = low_power
        self.demo_frames: list[Path] = []
        self._demo_index = 0
        if self.demo:
            demo_video = asset_path("data/demo.mp4")
            demo_dir = asset_path("data/demo")
            if demo_video.exists():
                self.camera_index = str(demo_video)
            elif demo_dir.exists():
                self.demo_frames = sorted(demo_dir.glob("*.jpg")) + sorted(demo_dir.glob("*.png"))
            else:
                logging.warning(
                    "Demo mode enabled but no demo media found in data/"
                )

        self.cap: cv2.VideoCapture | None = None
        self.camera_available = False

        eye_cfg = self.config.data.get("eye_tracker", {})
        canonical = [
            eye_cfg.get("left_eye", [80.0, 100.0]),
            eye_cfg.get("right_eye", [176.0, 100.0]),
        ]
        self.tracker = _EyeTracker(
            alpha=self.config.data.get("tracker_alpha", 0.4),
            canonical=canonical,
        )
        self.tracker_lock = Lock()
        self.stop_event = Event()
        self._processing_thread: Thread | None = None
        self.REENCODE_INTERVAL_S = 10.0
        self._direction_lock = Lock()
        self.command_queue: SimpleQueue[Direction] = SimpleQueue()
        self.active_direction = Direction.BLEND
        self.gaze_mode = self.config.data.get("gaze_mode", False)
        self._gaze_last: Direction | None = None
        self._gaze_map = {
            (0, 0): Direction.AGE,
            (1, 0): Direction.GENDER,
            (0, 1): Direction.ETHNICITY,
            (1, 1): Direction.SPECIES,
        }
        self._last_affine: np.ndarray | None = None
        self._encode_durations: list[float] = []
        self.encode_fps = 0.0
        self._skip_next = False
        self.target_fps = self.config.data.get("fps", 15)
        self.frame_timer = FrameTimer(self.config.data.get("metrics_interval", 10))

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
            eye_cfg = self.config.data.get("eye_tracker", {})
            canonical = [
                eye_cfg.get("left_eye", [80.0, 100.0]),
                eye_cfg.get("right_eye", [176.0, 100.0]),
            ]
            self.tracker.canonical = np.array(canonical, dtype=np.float32)
        self.gaze_mode = self.config.data.get("gaze_mode", False)
        self.target_fps = self.config.data.get("fps", 15)
        emotion = self.config.data.get("active_emotion")
        if emotion:
            try:
                with self._direction_lock:
                    self.active_direction = Direction.from_str(emotion)
            except ValueError:
                logging.warning("Unknown emotion '%s' in config", emotion)

    # ------------------------------------------------------------------
    # Core latent helpers
    # ------------------------------------------------------------------
    def encode_face(
        self, frame: np.ndarray, eyes: tuple[tuple[int, int], tuple[int, int]] | None = None
    ) -> torch.Tensor:
        """Encode the current frame into latent ``w+`` space."""
        start = time()
        with log_timing("encode_face"):
            with self.tracker_lock:
                if eyes is None:
                    eyes = self.tracker.get_eyes(frame)
            if eyes is None:
                crop = cv2.resize(frame, (256, 256))
                M = None
            else:
                le, re = eyes
                M = cv2.getAffineTransform(
                    np.array([le, re], dtype=np.float32),
                    self.tracker.canonical,
                )
                crop = cv2.warpAffine(frame, M, (256, 256))
            latent, _ = self.model_manager.E(_to_tensor(crop).to(self.device), return_latents=True)
            self._last_affine = M
        dur = time() - start
        self._encode_durations.append(dur)
        if len(self._encode_durations) > 10:
            self._encode_durations.pop(0)
        total = sum(self._encode_durations)
        if total > 0:
            self.encode_fps = len(self._encode_durations) / total
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
        """Compute the current morph offset vector and its magnitude.

        The timestamp ``t`` is converted into a phase of the morph cycle by
        taking ``t % cycle_seconds`` and normalising it to ``[0, 1)``. This phase
        is then mapped to a triangle wave using ``1 - abs(phase * 2 - 1)`` so the
        value smoothly ramps from ``0`` to ``1`` and back within each cycle.
        Multiplying this waveform by ``max_magnitudes`` yields the final
        ``current_magnitude`` used for scaling.

        If :attr:`active_direction` is :class:`~Direction.BLEND`, ``blend_weights``
        are normalised and combined so that each latent direction contributes
        proportionally. Otherwise, the single active direction is used. The
        resulting unit vector is multiplied by ``current_magnitude`` to produce
        the offset added to the baseline latent.

        Args:
            t: Absolute or relative timestamp, typically from :func:`time.time`.

        Returns:
            Tuple[np.ndarray, float]: ``offset`` is the scaled latent direction
            vector, and ``current_magnitude`` is the scalar amount used for the
            scaling.

        Example:
            >>> vp = VideoProcessor(...)
            >>> offset, mag = vp.latent_offset(time.time())
            >>> latent = baseline_latent + torch.from_numpy(offset).to(vp.device)
        """

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

    def _handle_camera_error(
        self,
        frame_emitter: "pyqtSignal" | None,
        retry_delay: float,
        max_delay: float,
    ) -> float:
        """Display an error screen and retry camera initialization."""

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
        self._init_camera()
        if self.camera_available:
            retry_delay = 1.0
            logging.info("Camera re-initialized successfully.")
        return retry_delay

    def _process_frame(
        self,
        frame: np.ndarray,
        baseline_latent: torch.Tensor | None,
        last_encode: float,
        idle_frames: int,
    ) -> tuple[np.ndarray, torch.Tensor, float, int, float]:
        """Process a single frame and return augmented output."""

        now = time()
        with self.tracker_lock:
            eyes = self.tracker.get_eyes(frame)
            gaze = self.tracker.get_gaze()

        if self.model_manager.model_load_failed:
            out_frame = frame.copy()
            current_magnitude = 0.0
        else:
            if self.gaze_mode and gaze is not None:
                self._update_direction_from_gaze(gaze)

            if baseline_latent is None or (now - last_encode) > self.REENCODE_INTERVAL_S:
                baseline_latent = self.encode_face(frame, eyes)
                last_encode = now
                logging.info("Encoded new baseline latent.")

            offset, current_magnitude = self.latent_offset(now)
            latent_mod = baseline_latent + torch.from_numpy(offset).to(self.device)
            out_frame = self.decode_latent(latent_mod, (frame.shape[0], frame.shape[1]))

        if eyes is not None:
            idle_frames = 0
            cv2.circle(out_frame, eyes[0], 3, (0, 255, 0), -1)
            cv2.circle(out_frame, eyes[1], 3, (0, 255, 0), -1)
        else:
            idle_frames += 1

        return out_frame, baseline_latent, last_encode, idle_frames, current_magnitude

    def _display_frame(
        self,
        out_frame: np.ndarray,
        frame_emitter: "pyqtSignal" | None,
        current_magnitude: float,
        idle_frames: int,
        idle_threshold: int,
        fade_frames: int,
    ) -> bool:
        """Render the frame via Qt or OpenCV."""

        self._draw_hud(out_frame, current_magnitude)
        if self.model_manager.model_load_failed:
            self._draw_model_error(out_frame)

        self._apply_idle_overlay(out_frame, idle_frames, idle_threshold, fade_frames)

        if self.ui == "qt" and frame_emitter:
            from PyQt6.QtCore import QThread

            frame_emitter.emit(_numpy_to_qimage(out_frame))
            QThread.msleep(int(1000 / self.config.data["fps"]))
            return True

        cv2.imshow("Latent Self", out_frame)
        key = cv2.waitKey(int(1000 / self.config.data["fps"])) & 0xFF
        return self._handle_frame_input(key)

    def _draw_hud(self, frame: np.ndarray, magnitude: float) -> None:
        """Overlay status text on the output frame."""

        with self._direction_lock:
            mode = self.active_direction
        mode_label = self.direction_labels.get(mode.value, mode.value.capitalize())
        cv2.putText(
            frame,
            f"Mode: {mode_label.capitalize()} ({magnitude:+.1f})",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_model_error(self, frame: np.ndarray) -> None:
        """Display a warning when models failed to load."""

        text = "Models failed to load"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = (frame.shape[1] - w) // 2
        y = h + 10
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _apply_idle_overlay(
        self,
        frame: np.ndarray,
        idle_frames: int,
        idle_threshold: int,
        fade_frames: int,
    ) -> None:
        """Darken the frame when the user is idle."""

        if idle_frames <= idle_threshold:
            return

        alpha = min(1.0, (idle_frames - idle_threshold) / fade_frames)
        overlay = frame.copy()
        overlay[:] = (0, 0, 0)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _handle_frame_input(self, key: int) -> bool:
        """Process keyboard input and update state."""

        if key == ord("q"):
            self.stop_event.set()
            return False

        ch = chr(key).lower()
        direction = Direction.from_key(ch)
        if direction:
            with self._direction_lock:
                self.active_direction = direction

        self._send_mqtt_heartbeat()
        return True

    def _maybe_adjust_performance(self) -> None:
        """Adapt resolution or drop frames when encode FPS is too low."""
        if not self.low_power or not self._encode_durations:
            return
        if self.encode_fps >= self.target_fps:
            return
        if self.resolution > 128:
            self.resolution = max(128, self.resolution // 2)
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution)
            logging.warning("Low-power mode: reduced resolution to %spx", self.resolution)
        else:
            self._skip_next = True

    def _get_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame from camera or demo media."""

        if self.demo_frames:
            frame = cv2.imread(str(self.demo_frames[self._demo_index]))
            self._demo_index = (self._demo_index + 1) % len(self.demo_frames)
            return frame is not None, frame

        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if self.demo and not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def _init_camera(self) -> None:
        """Initialize the camera capture device."""

        self.cap = cv2.VideoCapture(self.camera_index)
        self.camera_available = self.cap.isOpened()
        if self.camera_available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution)
        else:
            logging.error("Failed to open camera at index %s", self.camera_index)

    def _process_stream(self, frame_emitter: "pyqtSignal" | None = None) -> None:
        """Main loop orchestrating capture, processing and display.

        This loop delegates camera retries, frame processing and UI drawing to
        helper methods to keep the flow readable.
        """

        if not self.demo_frames:
            self._init_camera()
        else:
            self.cap = None
            self.camera_available = True

        baseline_latent: torch.Tensor | None = None
        last_encode = 0.0
        idle_frames = 0
        idle_threshold = int(self.config.data.get("idle_seconds", 3) * self.config.data["fps"])
        fade_frames = int(self.config.data.get("idle_fade_frames", self.config.data["fps"]))
        retry_delay = 1.0
        max_delay = 30.0
        last_out_frame: np.ndarray | None = None
        last_current_magnitude = 0.0

        try:
            while not self.stop_event.is_set():
                self._drain_direction_queue()

                if self.low_power and self._skip_next:
                    self._skip_next = False
                    if last_out_frame is not None:
                        if not self._display_frame(
                            last_out_frame,
                            frame_emitter,
                            last_current_magnitude,
                            idle_frames,
                            idle_threshold,
                            fade_frames,
                        ):
                            break
                    continue

                if not self.camera_available:
                    retry_delay = self._handle_camera_error(frame_emitter, retry_delay, max_delay)
                    continue

                ret, frame = self._get_frame()
                if not ret:
                    logging.error("Camera read failed – retrying")
                    self.camera_available = False
                    continue

                with self.frame_timer.track():
                    (
                        out_frame,
                        baseline_latent,
                        last_encode,
                        idle_frames,
                        current_magnitude,
                    ) = self._process_frame(
                        frame, baseline_latent, last_encode, idle_frames
                    )
                
                if not self._display_frame(
                    out_frame,
                    frame_emitter,
                    current_magnitude,
                    idle_frames,
                    idle_threshold,
                    fade_frames,
                ):
                    break
                last_out_frame = out_frame
                last_current_magnitude = current_magnitude
                self._maybe_adjust_performance()
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

    def _update_direction_from_gaze(self, gaze: tuple[float, float]) -> None:
        """Set direction based on normalized gaze coordinates."""
        region = (
            1 if gaze[0] >= 0.5 else 0,
            1 if gaze[1] >= 0.5 else 0,
        )
        direction = self._gaze_map.get(region)
        if direction and direction != self._gaze_last:
            with self._direction_lock:
                self.active_direction = direction
            self._gaze_last = direction

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

        mqtt_cfg = config.data["mqtt"]
        broker = mqtt_cfg["broker"]
        port = mqtt_cfg["port"]
        use_tls = mqtt_cfg.get("tls", False)

        # Support mqtt:// and mqtts:// URLs
        if broker.startswith("mqtt://") or broker.startswith("mqtts://"):
            parsed = urlparse(broker)
            broker = parsed.hostname or "localhost"
            if parsed.port:
                port = parsed.port
            use_tls = parsed.scheme == "mqtts"

        topic_namespace = mqtt_cfg["topic_namespace"]
        device_id = mqtt_cfg.get("device_id") or platform.node()

        self.client = mqtt.Client(client_id=device_id)
        self.client.on_connect = lambda c, u, f, rc: logging.info("MQTT connected" if rc == 0 else f"MQTT connect failed {rc}")
        self.client.on_disconnect = lambda c, u, rc: logging.warning(f"MQTT disconnected {rc}")

        username = mqtt_cfg.get("username")
        password = mqtt_cfg.get("password")
        if password and isinstance(password, str) and password.startswith("$"):
            password = os.getenv(password[1:], "")

        if username:
            self.client.username_pw_set(username, password or None)

        if use_tls:
            self.client.tls_set(
                ca_certs=mqtt_cfg.get("ca_cert"),
                certfile=mqtt_cfg.get("client_cert"),
                keyfile=mqtt_cfg.get("client_key"),
            )

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
    """Background memory usage monitor.

    When ``live_memory_stats`` is enabled and PyQt6 is available, the
    :attr:`memory_update` signal periodically emits the current CPU and GPU
    usage in megabytes and gigabytes respectively.
    """

    def __init__(self, config: ConfigManager) -> None:
        """Create a memory monitor from configuration values."""
        self.interval = config.data.get("memory_check_interval", 10)
        self.max_cpu_mb = config.data.get("max_cpu_mem_mb")
        self.max_gpu_gb = config.data.get("max_gpu_mem_gb")
        self.emit_signals = config.data.get("live_memory_stats", False)
        self.emitter = None
        self.memory_update = None
        if self.emit_signals:
            try:
                from PyQt6.QtCore import QObject, pyqtSignal

                class _Emitter(QObject):
                    memory_update = pyqtSignal(float, float)

                self.emitter = _Emitter()
                self.memory_update = self.emitter.memory_update
            except Exception as e:  # pragma: no cover - optional feature
                logging.warning("Live memory stats disabled: %s", e)
                self.emit_signals = False
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
            if self.emit_signals and self.memory_update:
                self.memory_update.emit(cpu_mb, gpu_gb)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        """Stop the monitoring thread."""

        if not self._stop.is_set():
            self._stop.set()
            if self._thread:
                self._thread.join()

