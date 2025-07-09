# Solucion de Problemas

Esta guia enumera problemas comunes al ejecutar Latent Self y como resolverlos.

## Camara no detectada
- Ensure the webcam is properly connected and not in use by another application.
- Specify the correct device index with `--camera N` if you have multiple cameras.
- On Linux, verify that your user has permission to access `/dev/video*`.

## Pesos del modelo faltantes
- Download `ffhq-1024-stylegan2.pkl`, `e4e_ffhq_encode.pt` and `latent_directions.npz`.
- Place these files inside the `models/` directory at the project root.
- Check file paths and permissions if the application still cannot load them.

## Problemas de instalacion de Qt
- The Qt UI requires **PyQt6**. Install it via:
  ```bash
  pip install PyQt6
  ```
- If you encounter platform-specific Qt errors, try reinstalling PyQt6 or using the OpenCV UI with `--ui cv2`.

Para mas ayuda consulte el [Manual de Usuario](user_manual.md).
