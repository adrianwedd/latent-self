# Troubleshooting

This guide lists common problems when running Latent Self and how to resolve them.

## Camera not detected
- Ensure the webcam is properly connected and not in use by another application.
- Specify the correct device index with `--camera N` if you have multiple cameras.
- On Linux, verify that your user has permission to access `/dev/video*`.

## Missing model weights
- Download `ffhq-1024-stylegan2.pkl`, `e4e_ffhq_encode.pt` and `latent_directions.npz`.
- Place these files inside the `models/` directory at the project root.
- Check file paths and permissions if the application still cannot load them.

## Qt installation problems
- The Qt UI requires **PyQt6**. Install it via:
  ```bash
  pip install PyQt6
  ```
- If you encounter platform-specific Qt errors, try reinstalling PyQt6 or using the OpenCV UI with `--ui cv2`.

For further help see the [User Manual](user_manual.md).
