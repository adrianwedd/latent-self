#!/bin/bash
# install_kiosk.sh - Installs the Latent Self kiosk application.

set -e

APP_NAME="latent-self"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Paths are resolved relative to this script so it can be run from any directory.
INSTALL_DIR="/opt/$APP_NAME"
SERVICE_FILE="$SCRIPT_DIR/latent_self.service"
# PyInstaller build output should live in ../dist/latent-self relative to the repo root
EXECUTABLE_FILE="$SCRIPT_DIR/../dist/$APP_NAME"
SERVICE_PATH="/etc/systemd/system/$(basename "$SERVICE_FILE")"
KIOSK_USER="${KIOSK_USER:-kiosk}"
KIOSK_GROUP="${KIOSK_GROUP:-$KIOSK_USER}"

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root." >&2
    exit 1
fi

if [ ! -f "$EXECUTABLE_FILE" ]; then
    echo "Error: Application executable not found at $EXECUTABLE_FILE. Please build the application first." >&2
    exit 1
fi

if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Systemd service file not found at $SERVICE_FILE." >&2
    exit 1
fi

echo "Installing Latent Self kiosk..."

# 1. Create user and directories
if ! id "$KIOSK_USER" &>/dev/null; then
    useradd -r -s /bin/false "$KIOSK_USER"
fi
install -d -o "$KIOSK_USER" -g "$KIOSK_GROUP" "$INSTALL_DIR"

# 2. Copy application files
install -o "$KIOSK_USER" -g "$KIOSK_GROUP" -m 755 "$EXECUTABLE_FILE" "$INSTALL_DIR/$APP_NAME"
install -o root -g root -m 644 "$SERVICE_FILE" "$SERVICE_PATH"

# 3. Set permissions (redundant when using install but kept for clarity)
chmod 755 "$INSTALL_DIR/$APP_NAME"

# 4. Reload systemd and enable the service
if ! command -v systemctl &> /dev/null
then
    echo "Error: systemctl command not found. This script requires systemd." >&2
    exit 1
fi

systemctl daemon-reload
systemctl enable "$(basename "$SERVICE_FILE")"

# 5. (Optional) Hide TTY switching for a more secure kiosk
# This prevents users from switching to a virtual console.
# MASK_PATH="/etc/systemd/system/getty@.service.d/override.conf"
# mkdir -p "$(dirname "$MASK_PATH")"
# cat > "$MASK_PATH" <<EOF
# [Unit]
# Mask=yes
# EOF

echo "Installation complete."
echo "You can start the service with: systemctl start $(basename "$SERVICE_FILE")"
