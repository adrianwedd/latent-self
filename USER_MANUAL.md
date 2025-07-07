# User Manual

This document explains how to set up and run **Latent Self**. It also covers the available user interfaces, the admin panel and the most common command line options.

## Setup

1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/your-username/latent-self.git
   cd latent-self
   ```
2. Create a Python 3.11 virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download the model weights (`ffhq-1024-stylegan2.pkl`, `e4e_ffhq_encode.pt` and `latent_directions.npz`) and place them in the `models/` folder.

Optional: build a standalone executable using PyInstaller with `pyinstaller latent_self.spec`.

## Running the Application

Execute the main script to start with the default OpenCV window:
```bash
python latent_self.py
```

To use the Qt based mirror UI run:
```bash
python latent_self.py --ui qt
```
Add `--kiosk` to launch fullscreen and hide the mouse cursor. Use `--demo` to replay prerecorded media from `data/` when a webcam is not available.

## Admin Panel

When running with the Qt interface, press **F12** to open the admin controls. Changes are written back to `config.yaml` and take effect immediately. Use **Q** or **Esc** to exit the mirror.

## Command Line Options

Common flags (see `-h` for the full list):

| Option | Description |
|-------|-------------|
| `--camera N` | Select webcam index |
| `--resolution PX` | Frame size (square) |
| `--fps N` | Target frames per second |
| `--device {auto,cpu,cuda}` | Select processing device |
| `--low-power` | Adaptive frame dropping |
| `--demo` | Use prerecorded media |
| `--ui {cv2,qt}` | UI backend |
| `--kiosk` | Fullscreen mode (Qt only) |
| `--cycle-duration SECS` | Morph cycle duration |
| `--blend-age WEIGHT` | Age blend weight |
| `--blend-gender WEIGHT` | Gender blend weight |
| `--blend-smile WEIGHT` | Smile blend weight |
| `--blend-species WEIGHT` | Species blend weight |
| `--emotion NAME` | Starting emotion |

Consult `docs/user_manual.md` or the troubleshooting guide for more details.
