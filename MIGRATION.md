# Migration Guide

This document explains the changes made to the latent-self project.

## Latent Directions

The latent directions have been updated to use the InterFaceGAN vectors for age, gender, and smile. The new vectors are stored in `latent_directions.npz`.

The `directions.yaml` file contains metadata for each direction, including a
human‑friendly `label` and the default `max_magnitude`.

## Hot-keys

The following hot-keys have been added:

*   `y`: Morph along the 'Age' axis
*   `g`: Morph along the 'Gender' axis
*   `h`: Morph along the 'Smile' axis

## HUD

A HUD has been added to the UI to display the current vector and magnitude.

## Orthogonality Check

An orthogonality check has been added to the application to ensure that the latent directions are orthogonal. The check is performed on startup and the results are logged to the console.
