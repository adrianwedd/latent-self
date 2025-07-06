"""Configuration schema models used for validation and CLI overrides."""

from __future__ import annotations

from pydantic import BaseModel, Field, RootModel
from pydantic_settings import BaseSettings
from typing import Dict

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

class AppConfig(BaseModel):
    """Primary application configuration model."""

    cycle_duration: float = 12.0
    blend_weights: BlendWeights = Field(default_factory=BlendWeights)
    fps: int = 15
    tracker_alpha: float = 0.4
    canonical_eyes: list[list[float]] = Field(
        default_factory=lambda: [[80.0, 100.0], [176.0, 100.0]]
    )
    admin_password_hash: str = ""
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    idle_seconds: int = 3
    max_cpu_mem_mb: int | None = None
    max_gpu_mem_gb: float | None = None
    emotion: Direction | None = None
    memory_check_interval: int = 10
    idle_fade_frames: int | None = None
    active_emotion: Direction = Direction.HAPPY

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
    emotion: Direction | None = None
