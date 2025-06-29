# Latent Self

An interactive face-morphing mirror installation.

## Overview

Latent Self is an interactive art installation that uses a webcam to capture a user's face and then applies a series of transformations to it in real-time. The transformed image is then displayed on a screen, creating a "latent self" of the user.

## Features

*   Real-time face morphing
*   Multiple transformation axes (age, gender, smile, species)
*   Adjustable blend weights for each axis
*   Fullscreen kiosk mode
*   Admin panel for on-site configuration
*   MQTT heartbeat for remote monitoring

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

## Usage

```bash
python latent_self.py
```

## Demo

![Demo GIF](https://via.placeholder.com/600x400.gif?text=Demo+GIF+Placeholder)