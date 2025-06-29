# Latent Self

This project is an interactive art installation that uses a webcam to capture a user's face and then applies a series of transformations to it in real-time. The transformed image is then displayed on a screen, creating a "latent self" of the user.

## Key Technologies

*   **Python:** The core application logic is written in Python.
*   **OpenCV:** Used for capturing video from the webcam and for basic image processing.
*   **PyQt6:** The graphical user interface is built using PyQt6, which allows for a fullscreen, kiosk-style deployment.
*   **PyInstaller:** Used to package the application into a single executable file for easy distribution.
*   **StyleGAN & e4e:** The face transformation is powered by a StyleGAN model, with the e4e encoder used to project the user's face into the latent space of the model.
*   **Mediapipe:** Used for face and eye tracking.
*   **MQTT:** The application can optionally send a heartbeat to an MQTT broker for remote monitoring.

## My Role

My role in this project is to take the initial prototype and productize it, making it suitable for a gallery setting. This includes:

*   **Improving the user interface:** Replacing the basic OpenCV window with a fullscreen PyQt6 application.
*   **Adding a configuration system:** Allowing for the easy adjustment of the application's parameters.
*   **Creating an admin panel:** Giving curators the ability to fine-tune the experience on-site.
*   **Packaging the application:** Creating a single executable that can be easily deployed.
*   **Adding robustness and polish:** Implementing features like structured logging, error handling, and an idle state.
*   **Extending the application's functionality:** Adding new transformation options, such as the "species" morph.
