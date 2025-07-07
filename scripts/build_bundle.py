#!/usr/bin/env python3
"""Build PyInstaller bundle for the current platform."""
from __future__ import annotations
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    system = platform.system()
    if system == "Windows":
        # pyinstaller on Windows requires pywin32 for proper hooks
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
    elif system == "Darwin":
        # no special dependencies yet, placeholder for future macOS tweaks
        pass

    spec = ROOT / "latent_self.spec"
    subprocess.check_call(["pyinstaller", str(spec)])


if __name__ == "__main__":
    main()
