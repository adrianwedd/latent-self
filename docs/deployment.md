# Deployment

This project can run as a kiosk service using **systemd**.

1. Build the application with PyInstaller. A helper script installs any
   OS-specific dependencies and runs the build:
   ```bash
   python scripts/build_bundle.py
   ```
2. Copy the service file and executable using the provided install script:
   ```bash
   sudo deploy/install_kiosk.sh
   ```
   This script creates a dedicated `kiosk` user and places the executable in
   `/opt/latent-self/`. It also installs `deploy/latent_self.service` into
   `/etc/systemd/system/`.
3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable latent-self.service
   sudo systemctl start latent-self.service
   ```

The service is configured with `Restart=on-failure` and `WatchdogSec=30s`
to automatically restart the application if it crashes.

PyInstaller bundles can be produced on Linux, Windows and macOS. The GitHub
Actions workflow uploads the artifacts from all three platforms under the
`dist/` directory.
