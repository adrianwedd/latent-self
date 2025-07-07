import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import yaml
from types import SimpleNamespace
from PyQt6.QtCore import QObject, pyqtSignal, Qt
import pytest

from ui.admin import AdminDialog
from ui.fullscreen import MirrorWindow
from directions import Direction

class DummyMemory(QObject):
    memory_update = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()

class DummyConfig:
    def __init__(self, path):
        self.data = {
            'admin_password_hash': 'x',
            'cycle_duration': 5,
            'blend_weights': {'age': 0.1, 'gender': 0.2, 'ethnicity': 0.3, 'species': 0.4},
            'fps': 30,
            'tracker_alpha': 0.5,
            'gaze_mode': False,
            'live_memory_stats': True,
            'max_cpu_mem_mb': 512,
            'max_gpu_mem_gb': 1,
        }
        self.config_path = path / 'config.yaml'
        self.config_path.write_text(yaml.dump(self.data))
        self.reload_called = False

    def reload(self):
        self.reload_called = True
        self.data = yaml.safe_load(self.config_path.read_text())

class DummyVideo:
    def __init__(self):
        self.enqueued = []
    def enqueue_direction(self, d):
        self.enqueued.append(d)

@pytest.fixture()
def dummy_app(tmp_path):
    cfg = DummyConfig(tmp_path)
    mem = DummyMemory()
    video = DummyVideo()
    return SimpleNamespace(config=cfg, memory=mem, video=video)

def test_admin_memory_signal_and_save(qtbot, dummy_app, monkeypatch):
    parent = MirrorWindow(dummy_app)
    qtbot.addWidget(parent)
    monkeypatch.setattr(AdminDialog, '_check_password', lambda self: True)
    dialog = AdminDialog(parent)
    qtbot.addWidget(dialog)

    # emit memory update
    dummy_app.memory.memory_update.emit(123.0, 1.5)
    qtbot.waitUntil(lambda: dialog.cpu_bar.value() == 123)
    if int(1.5 * 1024) > dialog.gpu_bar.maximum():
        expected_gpu = -1
    else:
        expected_gpu = int(1.5 * 1024)
    assert dialog.gpu_bar.value() == expected_gpu

    dialog.cycle_duration_slider.setValue(10)
    save_btn = dialog.button_box.button(dialog.button_box.StandardButton.Save)
    qtbot.mouseClick(save_btn, Qt.MouseButton.LeftButton)

    assert dummy_app.config.reload_called
    assert dummy_app.config.data['cycle_duration'] == 10


def test_mirror_window_keypresses(qtbot, dummy_app, monkeypatch):
    win = MirrorWindow(dummy_app)
    qtbot.addWidget(win)

    # patch AdminDialog
    class DummyDialog:
        def __init__(self, parent=None):
            self.exec_called = False
        def exec(self):
            self.exec_called = True
    dummy = DummyDialog()
    monkeypatch.setattr('ui.fullscreen.AdminDialog', lambda parent: dummy)

    qtbot.keyPress(win, Qt.Key.Key_F12)
    assert dummy.exec_called

    qtbot.keyPress(win, Qt.Key.Key_Y)
    assert dummy_app.video.enqueued[-1] == Direction.AGE

    closed = False
    def fake_close(*_):
        nonlocal closed
        closed = True
    monkeypatch.setattr(MirrorWindow, 'close', fake_close)
    qtbot.keyPress(win, Qt.Key.Key_Q)
    assert closed
