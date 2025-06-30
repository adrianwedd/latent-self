import types, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.modules.setdefault('cv2', types.ModuleType('cv2'))
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('mediapipe', types.ModuleType('mediapipe'))
import numpy as np
from threading import Lock

from directions import Direction
from services import VideoProcessor


class DummyModelManager:
    def __init__(self, latent_dirs):
        self.latent_dirs = latent_dirs


class DummyProcessor:
    def __init__(self):
        self.cycle_seconds = 4.0
        self.blend_weights = {
            Direction.AGE.value: 0.6,
            Direction.GENDER.value: 0.4,
        }
        self.max_magnitudes = {
            Direction.AGE.value: 3.0,
            Direction.GENDER.value: 3.0,
        }
        self.active_direction = Direction.AGE
        self.model_manager = DummyModelManager({
            Direction.AGE.value: np.ones(2),
            Direction.GENDER.value: np.full(2, 2.0),
        })
        self._direction_lock = Lock()

def test_magnitude_range():
    proc = DummyProcessor()
    # t=0 -> magnitude 0
    offset, mag = VideoProcessor.latent_offset(proc, 0.0)
    assert mag == 0
    assert np.allclose(offset, np.zeros(2))
    # t=cycle/2 -> max magnitude
    t_half = proc.cycle_seconds / 2
    offset, mag = VideoProcessor.latent_offset(proc, t_half)
    assert mag == 3.0
    assert np.allclose(offset, np.ones(2) * 3.0)


def test_single_direction():
    proc = DummyProcessor()
    t_half = proc.cycle_seconds / 2
    offset, mag = VideoProcessor.latent_offset(proc, t_half)
    assert np.allclose(offset, np.ones(2) * 3.0)
    assert mag == 3.0


def test_blend_mode():
    proc = DummyProcessor()
    proc.active_direction = Direction.BLEND
    t_half = proc.cycle_seconds / 2
    expected_dir = 0.6 * np.ones(2) + 0.4 * np.full(2, 2.0)
    offset, mag = VideoProcessor.latent_offset(proc, t_half)
    assert mag == 3.0
    assert np.allclose(offset, expected_dir * 3.0)

