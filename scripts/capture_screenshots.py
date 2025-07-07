import sys
from pathlib import Path
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QTimer
from ui.fullscreen import MirrorWindow
from ui.admin import AdminDialog
from ui.xycontrol import XYControl
# Patch XYControl paintEvent for headless screenshot
from PyQt6.QtWidgets import QWidget

def _safe_paint(self, event):
    painter = QPainter(self)
    painter.fillRect(self.rect(), Qt.GlobalColor.lightGray)
    painter.setPen(QPen(Qt.GlobalColor.black))
    painter.drawLine(int(self.width()/2), 0, int(self.width()/2), self.height())
    painter.drawLine(0, int(self.height()/2), self.width(), int(self.height()/2))
    painter.setBrush(QColor(200,0,0))
    painter.drawEllipse(self._pos, 5, 5)
    painter.end()
XYControl.paintEvent = _safe_paint

class DummyVideo:
    def __init__(self):
        self.direction_labels = {}
    def enqueue_direction(self, *args, **kwargs):
        pass

class DummyMemory:
    def __init__(self):
        self.emit_signals = False

class DummyModelManager:
    def __init__(self):
        self.error_message = ""

class DummyConfig:
    def __init__(self):
        self.data = {
            'cycle_duration': 12,
            'blend_weights': {
                'age': 0.4,
                'gender': 0.3,
                'ethnicity': 0.5,
                'species': 0.2,
            },
            'active_emotion': 'HAPPY',
            'fps': 15,
            'tracker_alpha': 0.4,
            'gaze_mode': False,
            'live_memory_stats': False,
            'max_cpu_mem_mb': 4096,
            'max_gpu_mem_gb': 8,
            'admin_password_hash': ''
        }
        self.config_path = Path('docs/dummy_config.yaml')
    def reload(self):
        pass
    def list_presets(self):
        return []

class DummyApp:
    def __init__(self):
        self.config = DummyConfig()
        self.video = DummyVideo()
        self.memory = DummyMemory()
        self.model_manager = DummyModelManager()


def main():
    qt_app = QApplication(sys.argv)
    dummy = DummyApp()
    win = MirrorWindow(dummy)
    img = QImage(640, 480, QImage.Format.Format_RGB32)
    img.fill(0xFF6666)
    win.update_frame(img)
    win.resize(640, 480)
    win.show()

    def _save_with_base64(path, pix):
        pix.save(path)
        b64_path = path + '.b64'
        with open(path, 'rb') as f_in, open(b64_path, 'wb') as f_out:
            import base64
            f_out.write(base64.b64encode(f_in.read()))

    def capture_mirror():
        screen = qt_app.primaryScreen()
        pix = screen.grabWindow(win.winId())
        _save_with_base64('docs/images/demo_window.png', pix)

    def show_admin():
        AdminDialog._check_password = lambda self: True
        dialog = AdminDialog(win)
        dialog.show()
        qt_app.processEvents()
        screen = qt_app.primaryScreen()
        pix = screen.grabWindow(dialog.winId())
        _save_with_base64('docs/images/admin_controls.png', pix)
        QTimer.singleShot(500, qt_app.quit)

    QTimer.singleShot(500, capture_mirror)
    QTimer.singleShot(1000, show_admin)

    qt_app.exec()

if __name__ == '__main__':
    main()
