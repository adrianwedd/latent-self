#!/bin/bash
# install_kiosk.sh - Installs the Latent Self kiosk application.

set -e

APP_NAME="latent-self"
INSTALL_DIR="/opt/$APP_NAME"
SERVICE_FILE="$APP_NAME.service"
EXECUTABLE_FILE="../dist/$APP_NAME"
SERVICE_PATH="/etc/systemd/system/$SERVICE_FILE"
KIOSK_USER="kiosk"
KIOSK_GROUP="$KIOSK_USER"

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
useradd -r -s /bin/false "$KIOSK_USER" || \
    echo "User '$KIOSK_USER' already exists."
mkdir -p "$INSTALL_DIR"

# 2. Copy application files
cp "$EXECUTABLE_FILE" "$INSTALL_DIR/"
cp "$SERVICE_FILE" "$SERVICE_PATH"

# 3. Set permissions
chown -R "$KIOSK_USER":"$KIOSK_GROUP" "$INSTALL_DIR"
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
