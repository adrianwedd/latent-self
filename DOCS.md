# Latent Self Documentation

This document provides detailed information about the Latent Self application, its architecture, and how to extend it.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Extending Functionality](#extending-functionality)
  - [Adding New Latent Directions](#adding-new-latent-directions)
  - [Customizing the UI](#customizing-the-ui)
- [Troubleshooting](#troubleshooting)

- [User Manual](docs/user_manual.md)
## Project Overview

Latent Self is an interactive art installation that uses real-time face morphing. It captures a user's face via webcam and applies transformations based on pre-trained StyleGAN and e4e models. The transformed image is displayed, creating a dynamic "altered reflection."

## Architecture

The application is primarily written in Python and leverages the following key libraries:

- **OpenCV**: For webcam capture and basic image manipulation.
- **PyQt6**: For building the graphical user interface, including the fullscreen mirror display and the admin panel.
- **PyTorch**: The underlying framework for the StyleGAN and e4e models.
- **Mediapipe**: Used for robust face and eye tracking to ensure accurate alignment for morphing.
- **PyInstaller**: For packaging the application into a standalone executable.
- **paho-mqtt**: (Optional) For sending heartbeat signals to an MQTT broker for remote monitoring.
- **pydantic-settings**: For typed configuration management and CLI overrides.

### Core Components

- `latent_self.py`: Contains the main application logic, including video processing, model inference, and UI integration.
- `config.py` (Implicit within `latent_self.py`): Handles loading, saving, and managing application settings from `data/config.yaml`.
- `ui/fullscreen.py`: Implements the main mirror display window.
- `ui/admin.py`: Provides an administrative interface for adjusting application parameters.

### Diagram

The following PlantUML diagram illustrates the high-level architecture:

```plantuml
!include docs/architecture.puml
```

## Configuration

Default configuration settings are located in `data/config.yaml`. On the first run, this file is copied to a user-specific configuration directory (e.g., `~/.latent_self/config.yaml` on Linux/macOS, or `C:\Users\<User>\AppData\Local\LatentSelf\LatentSelf\config.yaml` on Windows).

Changes made via the admin panel are saved to this user-specific `config.yaml`.

Key configuration parameters include:

- `cycle_duration`: The time (in seconds) for one complete morphing cycle.
- `blend_weights`: Dictionary controlling the influence of different latent directions (age, gender, ethnicity, species) when in blended morphing mode.
- `fps`: Target frames per second for the display.
- `tracker_alpha`: Smoothing factor for the eye-tracking algorithm.
- `canonical_eyes`: Reference eye coordinates used for face alignment.
- `admin_password_hash`: Hashed password for accessing the admin panel.
- To generate a new hash run `python scripts/generate_password_hash.py` and
  paste the output into `admin_password_hash` in your config file.
- `mqtt`: MQTT broker settings for optional remote monitoring.

## Extending Functionality

### Adding New Latent Directions

To add a new morphing direction (e.g., "happiness"), follow these steps:

1.  **Train a new latent direction**: This involves using a StyleGAN latent space exploration technique (e.g., GANSpace, InterFaceGAN) to identify a meaningful direction vector in the latent space. The output should be a NumPy array representing this direction.

2.  **Update `latent_directions.npz`**: Add your new direction vector to the `latent_directions.npz` file. This file is a NumPy archive containing all the direction vectors. Ensure the key for your new direction matches the desired name (e.g., `happiness`).

3.  **Modify `latent_self.py`**:
    - In the `get_latent_directions` function, add your new direction's key to the tuple of loaded directions (e.g., `("age", "gender", "ethnicity", "species", "happiness")`).
    - In the `_apply_config` method, ensure your new direction is initialized in `self.blend_weights` if it's not already present in the config file.
    - In the `_process_stream` method, add a new `elif` clause to handle a key press for your new direction (e.g., `elif key == ord("h"): self.active_direction = "HAPPINESS"`).

4.  **Modify `ui/admin.py`**: Add a new `QSlider` to the `AdminDialog._setup_ui` method for your new blend weight. Remember to multiply/divide by 100 for slider value conversion. Also, update the `save_and_reload` method to save the new blend weight to the config.

5.  **Update `data/config.yaml`**: Add a default blend weight for your new direction under the `blend_weights` section.

### Customizing the UI

- **Fullscreen Mirror (`ui/fullscreen.py`)**: This file handles the main display. You can modify its `paintEvent` method to draw custom overlays, information, or visual effects.
- **Admin Panel (`ui/admin.py`)**: The `_setup_ui` method defines the layout and widgets of the admin panel. You can add new controls (sliders, spinboxes, checkboxes) and connect them to configuration parameters.

## Troubleshooting

- **Camera Not Detected**: Ensure your webcam is properly connected and not in use by another application. Check the `camera_index` in `latent_self.py` or `config.yaml`.
- **Model Loading Errors**: Verify that all required model weights (`.pkl`, `.pt`, `.npz`) are present in the `models/` directory and are not corrupted.
- **PyQt6 Issues**: If the Qt UI fails to launch, ensure PyQt6 is correctly installed (`pip install PyQt6`).
- **Performance Issues**: Real-time performance is heavily dependent on GPU availability. Ensure CUDA is properly configured if you intend to use it (`--cuda` flag).

For step-by-step setup see the [User Manual](docs/user_manual.md).
