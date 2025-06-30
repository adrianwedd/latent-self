# Deployment

This project can run as a kiosk service using **systemd**.

1. Build the application with PyInstaller:
   ```bash
   pyinstaller latent_self.spec
   ```
2. Copy the service file and executable using the provided install script:
   ```bash
   sudo deploy/install_kiosk.sh
   ```
   This places the executable in `/opt/latent-self/` and installs
   `deploy/latent_self.service` into `/etc/systemd/system/`.
3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable latent-self.service
   sudo systemctl start latent-self.service
   ```

The service is configured with `Restart=on-failure` and `WatchdogSec=30s`
to automatically restart the application if it crashes.
