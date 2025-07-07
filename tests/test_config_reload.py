import sys, types, argparse, pathlib
from pathlib import Path
import yaml
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

sys.modules.setdefault('cv2', types.ModuleType('cv2'))
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('mediapipe', types.ModuleType('mediapipe'))

import services
from services import ConfigManager

class DummyApp:
    def __init__(self, config):
        self.config = config
        self.cycle_seconds = None
        self.blend_weights = None
        self.tracker_alpha = None

    def _apply_config(self):
        self.cycle_seconds = self.config.data['cycle_duration']
        self.blend_weights = self.config.data['blend_weights']
        self.tracker_alpha = self.config.data['tracker_alpha']

def make_args(tmpdir: Path):
    return argparse.Namespace(
        cycle_duration=None,
        blend_age=None,
        blend_gender=None,
        blend_smile=None,
        blend_species=None,
        fps=None,
        max_cpu_mem_mb=None,
        max_gpu_mem_gb=None,
    )

def test_reload_applies_changes(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        yaml.dump({
            "cycle_duration": 4.0,
            "blend_weights": {"age": 0.5, "gender": 0.5, "ethnicity": 0.5, "species": 0.5},
            "tracker_alpha": 0.4,
        })
    )
    (cfg_dir / "directions.yaml").write_text(
        yaml.dump({"age": {"label": "Age", "max_magnitude": 3.0}})
    )

    monkeypatch.setattr('appdirs.user_config_dir', lambda *_: str(cfg_dir))
    args = make_args(tmp_path)

    config = ConfigManager(args, app=None)
    app = DummyApp(config)
    config.app = app
    app._apply_config()

    assert app.cycle_seconds == 4.0

    # modify file
    (cfg_dir / "config.yaml").write_text(
        yaml.dump({
            "cycle_duration": 6.0,
            "blend_weights": {"age": 0.1, "gender": 0.2, "ethnicity": 0.3, "species": 0.4},
            "tracker_alpha": 0.7,
        })
    )

    config.reload()
    assert app.cycle_seconds == 6.0
    assert app.blend_weights['age'] == 0.1
    assert app.tracker_alpha == 0.7


def test_invalid_config_schema(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.dump({"cycle_duration": "fast"}))
    (cfg_dir / "directions.yaml").write_text(yaml.dump({}))
    monkeypatch.setattr('appdirs.user_config_dir', lambda *_: str(cfg_dir))
    args = make_args(tmp_path)
    with pytest.raises(RuntimeError):
        ConfigManager(args, app=None)

