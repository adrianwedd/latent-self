import sys, types, argparse, pathlib
from pathlib import Path
import yaml
import threading
import urllib.request
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

sys.modules.setdefault('cv2', types.ModuleType('cv2'))
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('mediapipe', types.ModuleType('mediapipe'))

from services import ConfigManager
import config_api


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


def test_reload_endpoint(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.dump({"cycle_duration": 4.0}))
    (cfg_dir / "directions.yaml").write_text(yaml.dump({}))

    monkeypatch.setattr('appdirs.user_config_dir', lambda *_: str(cfg_dir))
    args = make_args(tmp_path)
    config = ConfigManager(args, app=None)
    api = config_api.ConfigAPIServer(config, port=8765)
    api.start()
    try:
        (cfg_dir / "config.yaml").write_text(yaml.dump({"cycle_duration": 8.0}))
        req = urllib.request.Request("http://127.0.0.1:8765/reload", method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        time.sleep(0.1)
        assert config.data["cycle_duration"] == 8.0
    finally:
        api.shutdown()


def test_thread_safe_reload(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.dump({"cycle_duration": 1.0}))
    (cfg_dir / "directions.yaml").write_text(yaml.dump({}))

    monkeypatch.setattr('appdirs.user_config_dir', lambda *_: str(cfg_dir))
    args = make_args(tmp_path)
    config = ConfigManager(args, app=None)

    (cfg_dir / "config.yaml").write_text(yaml.dump({"cycle_duration": 5.0}))

    errors = []

    def worker():
        try:
            config.reload()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert config.data["cycle_duration"] == 5.0
