from __future__ import annotations

from pydantic import BaseModel, BaseSettings, Field
from typing import Dict

class BlendWeights(BaseModel):
    age: float = 0.4
    gender: float = 0.3
    ethnicity: float = 0.5
    species: float = 0.2
    smile: float | None = None

class MQTTConfig(BaseModel):
    enabled: bool = False
    broker: str = "localhost"
    port: int = 1883
    topic_namespace: str = "mirror"
    device_id: str = ""
    heartbeat_interval: int = 5

class AppConfig(BaseModel):
    cycle_duration: float = 12.0
    blend_weights: BlendWeights = Field(default_factory=BlendWeights)
    fps: int = 15
    tracker_alpha: float = 0.4
    admin_password_hash: str = ""
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    idle_seconds: int = 3
    idle_fade_frames: int | None = None

class DirectionEntry(BaseModel):
    label: str
    max_magnitude: float = 3.0

class DirectionsConfig(BaseModel):
    __root__: Dict[str, DirectionEntry]

    def to_dict(self) -> Dict[str, Dict[str, float | str]]:
        return {k: v.dict() for k, v in self.__root__.items()}

class CLIOverrides(BaseSettings):
    cycle_duration: float | None = None
    blend_age: float | None = None
    blend_gender: float | None = None
    blend_smile: float | None = None
    blend_species: float | None = None
    fps: int | None = None
