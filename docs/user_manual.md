# User Manual

This guide walks you through installing Latent Self, adjusting its configuration and resolving common issues.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/latent-self.git
   cd latent-self
   ```
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required model weights (StyleGAN generator, e4e encoder and `latent_directions.npz`) and place them inside the `models/` directory.
4. (Optional) Build a standalone executable with PyInstaller:
   ```bash
   pyinstaller latent_self.spec
   ```

## Configuration

When run for the first time, `data/config.yaml` is copied to a user specific directory such as `~/.latent_self/` on Linux/macOS or `%APPDATA%\LatentSelf\` on Windows. The admin panel writes back to this file whenever you adjust settings.

Key options include:

- `cycle_duration` – seconds for a full morph cycle.
- `blend_weights` – relative strength of each latent direction.
- `fps` – target frames per second.
- `admin_password_hash` – hashed password generated via `scripts/generate_password_hash.py`.
- `mqtt` – optional heartbeat settings.

You can also edit the YAML file directly or override values via CLI arguments.

## Common Issues

**Camera not detected**
: Ensure no other application is using the webcam and that the correct `camera_index` is set.

**Missing model weights**
: Double‑check that all `.pkl`, `.pt` and `.npz` files are present in `models/`.

**Qt UI fails to start**
: Confirm that PyQt6 is installed. Use `pip install PyQt6` if necessary.

**Poor performance**
: Running on CPU can be slow. Install CUDA drivers and use `--cuda` to enable GPU acceleration.

For additional help see the [project documentation](DOCS.md).
