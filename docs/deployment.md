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
   The script creates a dedicated `kiosk` user by default and places the
   executable in `/opt/latent-self/`. You can override the account by setting
   the `KIOSK_USER` environment variable:
   ```bash
   sudo KIOSK_USER=myuser deploy/install_kiosk.sh
   ```
   The service file is installed to `/etc/systemd/system/`.

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

## macOS

1. Build the macOS bundle:
   ```bash
   python scripts/build_bundle.py
   ```
   The resulting `LatentSelf.app` appears inside `dist/`.
2. Move the app to `/Applications` or another desired path.
3. Create a LaunchAgent so the kiosk starts on login. Save the following
   as `~/Library/LaunchAgents/com.latentself.kiosk.plist`:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
     <dict>
       <key>Label</key><string>com.latentself.kiosk</string>
       <key>ProgramArguments</key>
       <array>
         <string>/Applications/LatentSelf.app/Contents/MacOS/LatentSelf</string>
       </array>
       <key>RunAtLoad</key><true/>
     </dict>
   </plist>
   ```
   Load it with:
   ```bash
   launchctl load -w ~/Library/LaunchAgents/com.latentself.kiosk.plist
   ```
4. Alternatively, add the app to your **Login Items** in System Settings.

### macOS Troubleshooting
- If the app is blocked as "unidentified developer", right-click and choose
  **Open** or remove the quarantine attribute:
  `xattr -d com.apple.quarantine /Applications/LatentSelf.app`.
- Grant webcam permission under **System Settings › Privacy & Security › Camera** if video capture fails.

## Windows

1. Build the Windows executable:
   ```bash
   python scripts/build_bundle.py
   ```
   The output `LatentSelf.exe` lives in `dist\LatentSelf\`.
2. To launch at startup create a shortcut in the `shell:startup` folder or
   install a service with **nssm**:
   ```powershell
   nssm install LatentSelf "C:\\path\\to\\LatentSelf.exe"
   nssm start LatentSelf
   ```

### Windows Troubleshooting
- A missing `MSVCP*.dll` message means you need the Microsoft Visual C++
  Redistributable installed.
- If the Qt platform plugin "windows" cannot be loaded, ensure all files from
  the PyInstaller `dist` directory remain together.
