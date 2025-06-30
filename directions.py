"""Registry of supported latent directions and hotkeys."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Optional

class Direction(str, Enum):
    """Known latent directions."""

    AGE = "AGE"
    GENDER = "GENDER"
    SMILE = "SMILE"
    ETHNICITY = "ETHNICITY"
    SPECIES = "SPECIES"
    BLEND = "BLEND"

    @classmethod
    def from_str(cls, name: str) -> "Direction":
        """Look up a direction by name (case-insensitive)."""

        try:
            return cls[name.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown direction: {name}") from exc

    @classmethod
    def from_key(cls, key: str) -> Optional["Direction"]:
        """Return the direction bound to a keyboard shortcut."""

        return HOTKEYS.get(key.lower())

    @classmethod
    def key(cls, direction: "Direction") -> str:
        """Return the hotkey associated with ``direction``."""

        for k, v in HOTKEYS.items():
            if v is direction:
                return k
        raise ValueError(f"No hotkey for {direction}")


HOTKEYS: dict[str, Direction] = {
    "y": Direction.AGE,
    "g": Direction.GENDER,
    "h": Direction.SMILE,
    "e": Direction.ETHNICITY,
    "s": Direction.SPECIES,
    "b": Direction.BLEND,
}
