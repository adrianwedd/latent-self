from __future__ import annotations

from enum import Enum

class Direction(str, Enum):
    """Known latent directions."""

    AGE = "AGE"
    GENDER = "GENDER"
    SMILE = "SMILE"
    ETHNICITY = "ETHNICITY"
    SPECIES = "SPECIES"
    BLEND = "BLEND"

    _HOTKEYS = {
        "y": AGE,
        "g": GENDER,
        "h": SMILE,
        "e": ETHNICITY,
        "s": SPECIES,
        "b": BLEND,
    }

    @classmethod
    def from_str(cls, name: str) -> "Direction":
        try:
            return cls[name.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown direction: {name}") from exc

    @classmethod
    def from_key(cls, key: str) -> "Direction | None":
        return cls._HOTKEYS.get(key.lower())

    @classmethod
    def key(cls, direction: "Direction") -> str:
        for k, v in cls._HOTKEYS.items():
            if v is direction:
                return k
        raise ValueError(f"No hotkey for {direction}")
