# Latent Self

An interactive face-morphing mirror installation.

## Overview

Latent Self is an interactive art installation that uses a webcam to capture a user's face and then applies a series of transformations to it in real-time. The transformed image is then displayed on a screen, creating a "latent self" of the user.

## Features

*   Real-time face morphing
*   Multiple transformation axes (age, gender, smile, species, beauty)
*   Emotion bank with six presets (happy, angry, sad, fear, disgust, surprise)
*   Adjustable blend weights for each axis
*   Fullscreen kiosk mode (`--kiosk`)
*   Admin panel for on-site configuration
*   MQTT heartbeat for remote monitoring
*   Typed configuration via **pydantic-settings**
*   Demo mode with prerecorded media (`--demo`)
*   Optional live memory usage readout in the admin panel
*   Periodic logging of average FPS and latency metrics

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/latent-self.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the model weights (see below).

## Model Weights

This project requires the following model weights:

* [`ffhq-1024-stylegan2.pkl`](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) – StyleGAN‑ADA generator  
* [`e4e_ffhq_encode.pt`](https://huggingface.co/camenduru/PTI/resolve/main/e4e_ffhq_encode.pt) – e4e encoder  
* [`latent_directions.npz`](https://raw.githubusercontent.com/genforce/interfacegan/master/boundaries/latent_directions_ffhq.npz) – W⁺ latent directions (age, gender, smile)

Place these files in a `models` directory in the project root.

To speed up inference you can convert the generator and encoder to ONNX or TensorRT:
```bash
python scripts/convert_models.py --weights models --out models --tensorrt
```
ModelManager will automatically load `*.onnx` or `*.engine` files if present.

## Admin Password

Generate a hashed password for the admin panel:

```bash
python scripts/generate_password_hash.py mysecret
```

Copy the printed hash into your `config.yaml` under the
`admin_password_hash` field.

## Usage

```bash
python latent_self.py
python latent_self.py --ui qt --kiosk  # Qt fullscreen
python latent_self.py --demo           # Use prerecorded media
```

### Controls

Keyboard shortcuts when running with the default OpenCV UI:

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

## Demo

![Demo GIF](docs/images/demo.gif)
(If the image fails to load, run `python scripts/capture_screenshots.py` to generate it.)

To try the application without a webcam, place a `demo.mp4` file or a folder of
images inside the `data/` directory and run with `--demo`.

For additional options run:

```bash
python latent_self.py -h
```

See the [User Manual](USER_MANUAL.md) for detailed setup and the
[Troubleshooting Guide](docs/troubleshooting.md) for common issues.

## Metrics Output

The application logs average FPS and frame latency every few seconds. Adjust
`metrics_interval` in `config.yaml` to control how often these statistics are
emitted.

![Admin Controls](docs/images/admin_controls.png)
(Generate with `python scripts/capture_screenshots.py` if missing.)

