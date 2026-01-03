"""ui/admin.py - Admin panel for adjusting settings."""

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QCheckBox,
    QMessageBox,
    QRadioButton,
    QSlider,
    QSpinBox,
    QProgressBar,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
)
from .xycontrol import XYControl
from werkzeug.security import check_password_hash
from password_utils import hash_password

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
        err = getattr(self.app.model_manager, "error_message", "")
        if err:
            QMessageBox.critical(self, "Model Load Error", err)
        if getattr(self.app.memory, "memory_update", None):
            self.app.memory.memory_update.connect(self._update_memory_bars)
        if getattr(self.app, "worker", None) and hasattr(self.app.worker, "preview_frame"):
            self.app.worker.preview_frame.connect(self._update_preview)

    def _check_password(self) -> bool:
        """Prompt for password or set a new one if none is configured."""
        password_hash = self.app.config.data.get("admin_password_hash")
        if not password_hash:
            return self._set_new_password()

        password, ok = QInputDialog.getText(
            self,
            "Password Required",
            "Enter admin password:",
            QInputDialog.EchoMode.Password,
        )
        if not ok:
            return False

        if not check_password_hash(password_hash, password):
            QMessageBox.warning(self, "Authentication Failed", "Incorrect password.")
            return False
        return True

    def _set_new_password(self) -> bool:
        """Prompt the user to choose an admin password and store the hash."""
        pwd, ok = QInputDialog.getText(
            self,
            "Set Admin Password",
            "Create a new admin password:",
            QInputDialog.EchoMode.Password,
        )
        if not ok or not pwd:
            QMessageBox.warning(self, "Password Required", "A password is required to access the admin panel.")
            return False

        confirm, ok = QInputDialog.getText(
            self,
            "Confirm Password",
            "Re-enter the password:",
            QInputDialog.EchoMode.Password,
        )
        if not ok or pwd != confirm:
            QMessageBox.warning(self, "Mismatch", "Passwords did not match.")
            return False

        self.app.config.data["admin_password_hash"] = hash_password(pwd)
        with self.app.config.config_path.open("w") as f:
            yaml.dump(self.app.config.data, f, default_flow_style=False)
        return True

    def _setup_ui(self) -> None:
        """Create the settings form."""
        layout = QVBoxLayout(self)
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(160, 160)
        layout.addWidget(self.preview_label)

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

        self.gaze_checkbox = QCheckBox()
        self.gaze_checkbox.setChecked(self.app.config.data.get('gaze_mode', False))
        form_layout.addRow("Gaze Mode:", self.gaze_checkbox)

        self.memory_checkbox = QCheckBox()
        self.memory_checkbox.setChecked(self.app.config.data.get('live_memory_stats', False))
        form_layout.addRow("Live Memory Stats:", self.memory_checkbox)

        # -- XY Control --
        self.xy_widget = XYControl()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        for key, label in self.app.video.direction_labels.items():
            self.x_combo.addItem(label, key)
            self.y_combo.addItem(label, key)
        xy_select = QHBoxLayout()
        xy_select.addWidget(self.x_combo)
        xy_select.addWidget(self.y_combo)
        xy_box = QVBoxLayout()
        xy_box.addLayout(xy_select)
        xy_box.addWidget(self.xy_widget)
        xy_group = QGroupBox()
        xy_group.setLayout(xy_box)
        form_layout.addRow("Latent XY:", xy_group)

        self._xy = (0.0, 0.0)
        self.xy_widget.moved.connect(self._on_xy_move)
        self.x_combo.currentIndexChanged.connect(lambda _: self._apply_xy())
        self.y_combo.currentIndexChanged.connect(lambda _: self._apply_xy())

        cpu_max = self.app.config.data.get('max_cpu_mem_mb') or 4096
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, cpu_max)
        form_layout.addRow("CPU MB:", self.cpu_bar)

        gpu_max = int((self.app.config.data.get('max_gpu_mem_gb') or 8) * 1024)
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setRange(0, gpu_max)
        form_layout.addRow("GPU MB:", self.gpu_bar)

        # -- Presets --
        self.preset_combo = QComboBox()
        for name in self.app.config.list_presets():
            self.preset_combo.addItem(name)
        self.load_preset_btn = QPushButton("Load")
        self.save_preset_btn = QPushButton("Save Preset")
        preset_row = QHBoxLayout()
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(self.load_preset_btn)
        preset_row.addWidget(self.save_preset_btn)
        form_layout.addRow("Presets:", preset_row)

        layout.addLayout(form_layout)

        # -- Buttons --
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.save_and_reload)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.load_preset_btn.clicked.connect(self._load_selected_preset)
        self.save_preset_btn.clicked.connect(self._save_preset)

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
        self.app.config.data['gaze_mode'] = self.gaze_checkbox.isChecked()
        self.app.config.data['live_memory_stats'] = self.memory_checkbox.isChecked()

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
        self.app.memory.emit_signals = self.app.config.data.get('live_memory_stats', False)

        self.accept()

    def _update_memory_bars(self, cpu_mb: float, gpu_gb: float) -> None:
        """Update progress bars with current memory usage."""
        self.cpu_bar.setValue(int(cpu_mb))
        self.gpu_bar.setValue(int(gpu_gb * 1024))

    def _update_preview(self, img: QImage) -> None:
        """Display a downscaled video preview."""
        self.preview_label.setPixmap(QPixmap.fromImage(img))

    # XY control helpers
    def _on_xy_move(self, x: float, y: float) -> None:
        self._xy = (x, y)
        self._apply_xy()

    def _apply_xy(self) -> None:
        dir_x = self.x_combo.currentData()
        dir_y = self.y_combo.currentData()
        if dir_x and dir_y:
            self.app.video.update_xy_control(
                self._xy[0],
                self._xy[1],
                dir_x,
                dir_y,
            )

    # Preset helpers
    def _load_selected_preset(self) -> None:
        name = self.preset_combo.currentText()
        if name:
            self.app.config.load_preset(name)
            self.app.memory.emit_signals = self.app.config.data.get("live_memory_stats", False)
            self.accept()

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name:
            self.app.config.save_preset(name)
            self.preset_combo.clear()
            for p in self.app.config.list_presets():
                self.preset_combo.addItem(p)


