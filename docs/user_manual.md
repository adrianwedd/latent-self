# User Manual

This guide walks you through installing Latent Self, adjusting its configuration and resolving common issues.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/latent-self.git
   cd latent-self
   ```
2. Ensure you have **Python 3.11** installed and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the required model weights (StyleGAN generator, e4e encoder and `latent_directions.npz`) and place them inside the `models/` directory.
5. (Optional) Build a standalone executable with PyInstaller:
   ```bash
   pyinstaller latent_self.spec
   ```

## Configuration

When run for the first time, `data/config.yaml` is copied to a user specific directory such as `~/.latent_self/` on Linux/macOS or `%APPDATA%\LatentSelf\` on Windows. The admin panel writes back to this file whenever you adjust settings.

Key options include:

- `cycle_duration` – seconds for a full morph cycle.
- `blend_weights` – relative strength of each latent direction including `beauty`.
- `active_emotion` – starting emotion (happy, angry, sad, fear, disgust, surprise).
- `fps` – target frames per second.
- `gaze_mode` – switch directions based on where you look.
- `admin_password_hash` – hashed password generated via `scripts/generate_password_hash.py`.
- `admin_api_token` – token that must match the `X-Admin-Token` header for remote API requests.
- `mqtt` – optional heartbeat settings.
- `live_memory_stats` – show real-time CPU/GPU usage in the admin panel.

You can also edit the YAML file directly or override values via CLI arguments.

## UI Controls

### OpenCV Window

The default OpenCV UI accepts single-key shortcuts:

```
q - quit
y - age
g - gender
h - smile
e - ethnicity
s - species
u - beauty
1 - happy
2 - angry
3 - sad
4 - fear
5 - disgust
6 - surprise
b - blended morph
```

### Qt Mirror

When running with `--ui qt`, press **F12** to open the admin panel. Use **Q** or
**Esc** to quit. The same direction keys as above apply. When *Gaze Mode* is enabled,
looking at different screen quadrants automatically changes the morphing direction.
Enable *Live Memory Stats* to show CPU and GPU usage bars inside the dialog.

## Command-Line Options

Important flags (run `python latent_self.py -h` for the full list):

| Option | Description |
|--------|-------------|
| `--camera N` | Select webcam index |
| `--resolution PX` | Frame size (square pixels) |
| `--fps N` | Target frames per second |
| `--device {auto,cpu,cuda}` | Select processing device |
| `--low-power` | Adaptive frame dropping |
| `--demo` | Use prerecorded media from `data/` |
| `--ui {cv2,qt}` | UI backend |
| `--kiosk` | Hide cursor and launch fullscreen (Qt only) |
| `--gaze-mode` | Enable gaze-driven direction switching |
| `--cycle-duration SECS` | Duration of a morph cycle |
| `--blend-age WEIGHT` | Age blend weight |
| `--blend-gender WEIGHT` | Gender blend weight |
| `--blend-smile WEIGHT` | Smile blend weight |
| `--blend-species WEIGHT` | Species blend weight |
| `--emotion NAME` | Starting emotion |

## Common Issues

**Camera not detected**
: Ensure no other application is using the webcam and that the correct `camera_index` is set.

**Missing model weights**
: Double‑check that all `.pkl`, `.pt` and `.npz` files are present in `models/`.

**Qt UI fails to start**
: Confirm that PyQt6 is installed. Use `pip install PyQt6` if necessary.

**Poor performance**
: Running on CPU can be slow. Install CUDA drivers and run with `--device cuda` to enable GPU acceleration.

![Admin Controls](images/admin_controls.png)
(Run `python ../../scripts/capture_screenshots.py` to regenerate.)

## Demo

You can preview the experience without a webcam by running in demo mode:

```bash
python latent_self.py --demo
```

![Demo GIF](images/demo.gif)
(Run `python ../../scripts/capture_screenshots.py` to regenerate.)

For additional help see the [project documentation](DOCS.md).
