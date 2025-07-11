"""Configuration schema models used for validation and CLI overrides."""

from __future__ import annotations

from pydantic import BaseModel, Field, RootModel
from pydantic_settings import BaseSettings
from typing import Dict, List

from directions import Direction

class BlendWeights(BaseModel):
    """Weights for each morphing direction used when blending."""

    age: float = 0.4
    gender: float = 0.3
    ethnicity: float = 0.5
    species: float = 0.2
    smile: float | None = None

class MQTTConfig(BaseModel):
    """Settings for optional MQTT heartbeat publishing."""

    enabled: bool = False
    broker: str = "localhost"
    port: int = 1883
    topic_namespace: str = "mirror"
    device_id: str = ""
    heartbeat_interval: int = 5
    username: str | None = None
    password: str | None = None
    tls: bool = False
    ca_cert: str | None = None
    client_cert: str | None = None
    client_key: str | None = None

class OSCConfig(BaseModel):
    """Settings for optional OSC control."""

    enabled: bool = False
    port: int = 9000

class EyeTrackerConfig(BaseModel):
    """Configuration for eye tracking alignment."""

    left_eye: list[float] = Field(default_factory=lambda: [80.0, 100.0])
    right_eye: list[float] = Field(default_factory=lambda: [176.0, 100.0])


class ScheduleEntry(BaseModel):
    """Single scheduled action entry."""

    time: str
    preset: str | None = None
    model: str | None = None


class AppConfig(BaseModel):
    """Primary application configuration model."""

    cycle_duration: float = 12.0
    blend_weights: BlendWeights = Field(default_factory=BlendWeights)
    fps: int = 15
    tracker_alpha: float = 0.4
    eye_tracker: EyeTrackerConfig = Field(default_factory=EyeTrackerConfig)
    gaze_mode: bool = False
    device: str = "auto"
    admin_password_hash: str = ""
    admin_api_token: str = ""
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    osc: OSCConfig = Field(default_factory=OSCConfig)
    idle_seconds: int = 3
    max_cpu_mem_mb: int | None = None
    max_gpu_mem_gb: float | None = None
    emotion: Direction | None = None
    memory_check_interval: int = 10
    live_memory_stats: bool = False
    idle_fade_frames: int | None = None
    active_emotion: Direction = Direction.HAPPY
    schedule: List["ScheduleEntry"] = Field(default_factory=list)

class DirectionEntry(BaseModel):
    """Metadata for a single latent direction."""

    label: str
    max_magnitude: float = 3.0

class DirectionsConfig(RootModel[Dict[str, DirectionEntry]]):
    """Container for multiple DirectionEntry objects."""

    root: Dict[str, DirectionEntry]

    def to_dict(self) -> Dict[str, Dict[str, float | str]]:
        """Return plain dictionary representation."""
        return {k: v.model_dump() for k, v in self.root.items()}


class CLIOverrides(BaseSettings):
    """Command-line arguments mapped to config values."""

    cycle_duration: float | None = None
    blend_age: float | None = None
    blend_gender: float | None = None
    blend_smile: float | None = None
    blend_species: float | None = None
    fps: int | None = None
    max_cpu_mem_mb: int | None = None
    max_gpu_mem_gb: float | None = None
    gaze_mode: bool | None = None
    emotion: Direction | None = None
    device: str | None = None
