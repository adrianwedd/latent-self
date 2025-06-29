#!/usr/bin/env python3
"""
Latent‑Self “one‑button” bootstrap

• Creates/activates local .venv (if not inside one).
• Installs *all* required Python deps, including niche repos:
  - stylegan2‑ada‑pytorch   (for torch_utils + dnnlib)
  - encoder4editing (e4e / pSp)
  - rich, mediapipe, PyQt6 (optional), etc.
• Downloads generator, encoder, and latent_directions.
• Verifies SHA‑256s.
• Copies default config & directions to user dir.
• Prints colourful success banner.

Run from repo root:

    python scripts/setup.py          # interactive
    python scripts/setup.py --yes    # non‑interactive / CI
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, venv
from pathlib import Path
from urllib.request import urlopen

# ────────────────────────────────────────────────────────────────────────────────
# Rich console (self‑install)
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    )
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()
ROOT   = Path(__file__).resolve().parents[1]
VENV   = ROOT / ".venv"
MODELS = ROOT / "models"
DATA   = ROOT / "data"
USERC  = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "LatentSelf"
PY_MIN = (3, 10)

CORE_PKGS = [
    # Wheel‑available libs
    "opencv-python-headless",
    "mediapipe>=0.10",
    "numpy", "pillow", "PyYAML", "appdirs",
    # encoder4editing (pSp) – pip‑installable
]

STYLEGAN_REPO = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
STYLEGAN_DIR  = ROOT / "libs" / "stylegan2-ada"

ENCODER_REPO = "https://github.com/omertov/encoder4editing.git"
ENCODER_DIR  = ROOT / "libs" / "encoder4editing"

OPTIONAL = {
    "qt": ["PyQt6", "PyQt6-Qt6"],
}

MODELS_META = {
    # filename : (url, sha256 or None)
    "ffhq-1024-stylegan2.pkl":
        (
            "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            (
                "a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c",
                "22151b43e01d6b96d2f772671c5f3cf73b63fed6d2b2661d72f6cf6ca9b39194",
            ),
        ),
    "e4e_ffhq_encode.pt":
        ("https://huggingface.co/camenduru/PTI/resolve/main/e4e_ffhq_encode.pt",
         "748f4cb01604d2db53f141fa10542d91c44f7b98b713c2b49c375ddfac3f4efd"),
    "latent_directions.npz":
        ("https://raw.githubusercontent.com/genforce/interfacegan/master/boundaries/latent_directions_ffhq.npz", None),
}

# ────────────────────────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(8192), b""):
            h.update(blk)
    return h.hexdigest()

def ensure_python():
    if sys.version_info < PY_MIN:
        console.print(f"[bold red]Python {PY_MIN[0]}.{PY_MIN[1]}+ required.[/]")
        sys.exit(1)

def ensure_venv():
    if "VIRTUAL_ENV" in os.environ:
        return
    if not VENV.exists():
        console.print(":snake: [cyan]Creating .venv …[/]")
        venv.create(VENV, with_pip=True)
    act = VENV / ("Scripts/activate" if platform.system()=="Windows" else "bin/activate")
    console.print(f"[yellow]Activate with:[/]  `source {act}` then re‑run this script.")
    sys.exit(0)

def pip_install(pkgs: list[str], retry_ok: bool = False):
    if not pkgs:
        return True
    try:
        console.print(f":package: Installing [green]{', '.join(pkgs)}[/]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
        return True
    except subprocess.CalledProcessError as e:
        if retry_ok:
            return False
        raise

def ensure_stylegan_repo():
    """
    NVLabs repo has no setup.py; install fails.
    Clone sources under libs/ and inject a .pth so torch_utils import works.
    """
    if (STYLEGAN_DIR / "torch_utils").exists():
        console.print("[grey58]stylegan2‑ada source already present.[/]")
        return
    console.print(":arrow_down:  Cloning NVLabs stylegan2‑ada‑pytorch …")
    STYLEGAN_DIR.parent.mkdir(exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", STYLEGAN_REPO, str(STYLEGAN_DIR)])
    # Create a .pth inside site‑packages pointing to this dir
    import site
    pth = Path(site.getsitepackages()[0]) / "stylegan2_ada_local.pth"
    pth.write_text(str(STYLEGAN_DIR.resolve()))
    console.print("[green]✓  torch_utils import path registered.[/]")

def ensure_encoder_repo():
    """
    Clone encoder4editing into libs/ and register a .pth so
    `models.encoders.pSp` is importable.
    """
    if (ENCODER_DIR / "models").exists():
        console.print("[grey58]encoder4editing source already present.[/]")
        return
    console.print(":arrow_down:  Cloning encoder4editing …")
    ENCODER_DIR.parent.mkdir(exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", ENCODER_REPO, str(ENCODER_DIR)])

    import site
    pth = Path(site.getsitepackages()[0]) / "encoder4editing_local.pth"
    pth.write_text(str(ENCODER_DIR.resolve()))
    console.print("[green]✓  encoder4editing import path registered.[/]")

def download_models():
    MODELS.mkdir(exist_ok=True)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeRemainingColumn(), console=console) as p:
        for fname,(url,checksums) in MODELS_META.items():
            dest = MODELS / fname
            if checksums is None:
                allowed = None
            elif isinstance(checksums, (list, tuple, set)):
                allowed = set(checksums)
            else:
                allowed = {checksums}

            if dest.exists() and (allowed is None or sha256(dest) in allowed):
                console.print(f"[grey58]{fname} already present.")
                continue
            t = p.add_task(f"Downloading {fname}", total=None)
            with urlopen(url) as r, open(dest,"wb") as f:
                while chunk := r.read(8192):
                    f.write(chunk)
            p.remove_task(t)
            if allowed and sha256(dest) not in allowed:
                dest.unlink()
                console.print(f"[bold red]Checksum mismatch for {fname} – abort.[/]")
                sys.exit(1)
            console.print(f"[green]✓[/] {fname}")

def copy_default_configs():
    USERC.mkdir(parents=True, exist_ok=True)
    for yml in ("config.yaml","directions.yaml"):
        dst = USERC / yml
        if not dst.exists():
            shutil.copy(DATA / yml, dst)
            console.print(f"Installed default {yml} → {dst}")

# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Latent‑Self Bootstrap")
    parser.add_argument("--yes","-y",action="store_true",help="Automatic yes to prompts")
    args = parser.parse_args()

    console.print("[bold magenta]\nLatent‑Self Automated Setup[/]\n")
    ensure_python()
    ensure_venv()
    pip_install(CORE_PKGS)

    # Attempt to `import torch_utils`; if that fails, clone local repo fallback
    try:
        import importlib; importlib.import_module("torch_utils")
    except ModuleNotFoundError:
        console.print("[yellow]torch_utils not found after pip install – using source fallback.[/]")
        ensure_stylegan_repo()

    try:
        import importlib; importlib.import_module("models.encoders")
    except ModuleNotFoundError:
        console.print("[yellow]encoder4editing not importable – using source fallback.[/]")
        ensure_encoder_repo()

    # Optional extras
    if args.yes or console.input("Install Qt GUI support? ([y]/n) ").strip().lower() in ("","y"):
        pip_install(OPTIONAL["qt"])

    download_models()
    copy_default_configs()

    console.print(
        "\n[bold green]✓  Setup complete![/]  Activate your venv and run:\n"
        "   [cyan]python latent_self.py --ui qt[/]  (or cv2)\n"
    )

if __name__ == "__main__":
    main()