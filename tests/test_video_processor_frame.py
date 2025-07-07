import sys, types, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# stub heavy modules
sys.modules.setdefault('cv2', types.ModuleType('cv2'))
cv2 = sys.modules['cv2']
cv2.circle = lambda *a, **k: None
cv2.COLOR_BGR2RGB = 42
cv2.FONT_HERSHEY_SIMPLEX = 0
cv2.LINE_AA = 0
cv2.cvtColor = lambda img, code: img
cv2.addWeighted = lambda s1,a,s2,b,g,d: d.__setitem__(slice(None), s1)
cv2.getTextSize = lambda text, font, scale, thickness: ((0,0), None)

sys.modules.setdefault('torch', types.ModuleType('torch'))
torch = sys.modules["torch"]
class DummyTorch:
    def __init__(self, arr):
        self.arr = np.array(arr)
    def to(self, device):
        return self
    def __radd__(self, other):
        return other + self.arr
    def __add__(self, other):
        if isinstance(other, DummyTorch):
            return self.arr + other.arr
        return self.arr + other

torch.from_numpy = lambda arr: DummyTorch(arr)

sys.modules.setdefault('mediapipe', types.ModuleType('mediapipe'))

from directions import Direction
import services

class DummyEyeTracker:
    def __init__(self, *a, **kw):
        self.alpha = kw.get('alpha', 0.4)
        self.canonical = np.array(kw.get('canonical', [[0,0],[0,0]]), dtype=np.float32)
    def get_eyes(self, frame):
        return (1,1), (2,2)
    def get_gaze(self):
        return None

services._EyeTracker = DummyEyeTracker

class DummyModelManager:
    def __init__(self):
        self.latent_dirs = {Direction.AGE.value: np.ones(2)}
        self.model_load_failed = False
        self.E = lambda tensor, return_latents=True: (0, None)
        self.G = types.SimpleNamespace(synthesis=lambda latent, noise_mode='const': (latent, None, None))

class DummyAudio:
    def __init__(self, vol=0.0):
        self.volume = vol

class DummyConfig:
    def __init__(self):
        self.data = {
            'cycle_duration': 2.0,
            'blend_weights': {'age': 1.0},
            'fps': 30,
            'tracker_alpha': 0.4,
            'eye_tracker': {'left_eye': [80,100], 'right_eye': [176,100]},
            'gaze_mode': False,
            'idle_seconds': 3,
            'idle_fade_frames': 30,
        }
        self.directions_data = {'age': {'label': 'Age', 'max_magnitude': 3.0}}


def make_processor():
    cfg = DummyConfig()
    mm = DummyModelManager()
    return services.VideoProcessor(mm, cfg, device='cpu', camera_index=0, resolution=64, ui='cv2', audio=DummyAudio())


def test_process_frame_encodes(monkeypatch):
    vp = make_processor()
    frame = np.zeros((2,2,3), dtype=np.uint8)
    monkeypatch.setattr(vp, 'encode_face', lambda f, e: 0)
    monkeypatch.setattr(vp, 'decode_latent', lambda latent, shape: frame)
    monkeypatch.setattr(vp, 'latent_offset', lambda t: (np.zeros(2), 0.0))
    out, base, last, idle, mag = services.VideoProcessor._process_frame(vp, frame, None, 0.0, 0)
    assert base == 0
    assert idle == 0
    assert mag == 0.0
    assert out is frame


def test_process_frame_idle_and_failure(monkeypatch):
    vp = make_processor()
    frame = np.zeros((2,2,3), dtype=np.uint8)
    vp.model_manager.model_load_failed = True
    vp.tracker.get_eyes = lambda f: None
    out, base, last, idle, mag = services.VideoProcessor._process_frame(vp, frame, 0, 0.0, 1)
    assert idle == 2
    assert mag == 0.0
    assert np.array_equal(out, frame)
