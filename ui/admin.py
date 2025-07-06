"""ui/admin.py - Admin panel for adjusting settings."""

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)
from werkzeug.security import check_password_hash

class AdminDialog(QDialog):
    """A dialog for adjusting application settings."""

    def __init__(self, parent=None):
        """Prompt for password and prepare the form."""
        super().__init__(parent)
        self.app = parent.app
        self.setWindowTitle("Admin Panel")
        self.setModal(True)

        if not self._check_password():
            self.reject()
            return

        self._setup_ui()

    def _check_password(self) -> bool:
        """Prompt for password and check against hash in config."""
        password, ok = QInputDialog.getText(self, "Password Required", "Enter admin password:", QInputDialog.EchoMode.Password)
        if not ok:
            return False

        password_hash = self.app.config.data.get("admin_password_hash")
        if not password_hash or not check_password_hash(password_hash, password):
            QMessageBox.warning(self, "Authentication Failed", "Incorrect password.")
            return False
        return True

    def _setup_ui(self) -> None:
        """Create the settings form."""
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # -- Cycle Duration --
        self.cycle_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.cycle_duration_slider.setRange(1, 30)
        self.cycle_duration_slider.setValue(int(self.app.config.data['cycle_duration']))
        form_layout.addRow("Cycle Duration (s):", self.cycle_duration_slider)

        # -- Blend Weights --
        self.age_slider = QSlider(Qt.Orientation.Horizontal)
        self.age_slider.setRange(0, 100)
        self.age_slider.setValue(int(self.app.config.data['blend_weights']['age'] * 100))
        form_layout.addRow("Blend Age:", self.age_slider)

        self.gender_slider = QSlider(Qt.Orientation.Horizontal)
        self.gender_slider.setRange(0, 100)
        self.gender_slider.setValue(int(self.app.config.data['blend_weights']['gender'] * 100))
        form_layout.addRow("Blend Gender:", self.gender_slider)

        self.ethnicity_slider = QSlider(Qt.Orientation.Horizontal)
        self.ethnicity_slider.setRange(0, 100)
        self.ethnicity_slider.setValue(int(self.app.config.data['blend_weights']['ethnicity'] * 100))
        form_layout.addRow("Blend Ethnicity:", self.ethnicity_slider)

        self.species_slider = QSlider(Qt.Orientation.Horizontal)
        self.species_slider.setRange(0, 100)
        self.species_slider.setValue(int(self.app.config.data['blend_weights']['species'] * 100))
        form_layout.addRow("Blend Species:", self.species_slider)

        # -- Emotion Selection --
        self.emotion_groupbox = QGroupBox()
        emotion_layout = QHBoxLayout()
        self.emotion_buttons = {}
        emotions = ["Happy", "Angry", "Sad", "Fear", "Disgust", "Surprise"]
        current = self.app.config.data.get('active_emotion', 'HAPPY').capitalize()
        for name in emotions:
            btn = QRadioButton(name)
            if name == current:
                btn.setChecked(True)
            emotion_layout.addWidget(btn)
            self.emotion_buttons[name.upper()] = btn
        self.emotion_groupbox.setLayout(emotion_layout)
        form_layout.addRow("Emotion:", self.emotion_groupbox)

        # -- FPS Target --
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(self.app.config.data['fps'])
        form_layout.addRow("FPS Target:", self.fps_spinbox)

        # -- Tracker Alpha --
        self.tracker_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.tracker_alpha_slider.setRange(0, 100)
        self.tracker_alpha_slider.setValue(int(self.app.config.data['tracker_alpha'] * 100))
        form_layout.addRow("Tracker Alpha:", self.tracker_alpha_slider)

        layout.addLayout(form_layout)

        # -- Buttons --
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.save_and_reload)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def save_and_reload(self) -> None:
        """Save the new settings and reload the application config."""
        # Update config data
        self.app.config.data['cycle_duration'] = self.cycle_duration_slider.value()
        self.app.config.data['blend_weights']['age'] = self.age_slider.value() / 100.0
        self.app.config.data['blend_weights']['gender'] = self.gender_slider.value() / 100.0
        self.app.config.data['blend_weights']['ethnicity'] = self.ethnicity_slider.value() / 100.0
        self.app.config.data['blend_weights']['species'] = self.species_slider.value() / 100.0
        self.app.config.data['fps'] = self.fps_spinbox.value()
        self.app.config.data['tracker_alpha'] = self.tracker_alpha_slider.value() / 100.0

        # emotion selection
        for name, btn in self.emotion_buttons.items():
            if btn.isChecked():
                self.app.config.data['active_emotion'] = name
                # Apply immediately
                self.app.video.enqueue_direction(name)
                break

        # Save to file
        with self.app.config.config_path.open("w") as f:
            yaml.dump(self.app.config.data, f, default_flow_style=False)

        # Reload config in the main app
        self.app.config.reload()

        self.accept()
