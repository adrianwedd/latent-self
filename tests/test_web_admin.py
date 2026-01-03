import sys, types, argparse, pathlib
from pathlib import Path
import base64
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

sys.modules.setdefault('cv2', types.ModuleType('cv2'))
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('mediapipe', types.ModuleType('mediapipe'))

from services import ConfigManager
from web_admin import WebAdmin
from password_utils import hash_password


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        cycle_duration=None,
        blend_age=None,
        blend_gender=None,
        blend_smile=None,
        blend_species=None,
        fps=None,
        max_cpu_mem_mb=None,
        max_gpu_mem_gb=None,
        gaze_mode=None,
        emotion=None,
        device=None,
    )


def setup_config(tmp_path: Path, monkeypatch, password: str | None = None) -> ConfigManager:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    data = {"cycle_duration": 4.0}
    if password:
        data["admin_password_hash"] = hash_password(password)
    (cfg_dir / "config.yaml").write_text(yaml.dump(data))
    (cfg_dir / "directions.yaml").write_text(yaml.dump({}))
    monkeypatch.setattr("appdirs.user_config_dir", lambda *_: str(cfg_dir))
    return ConfigManager(make_args(), app=None)


def auth_header(pwd: str) -> dict[str, str]:
    token = base64.b64encode(f"user:{pwd}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_basic_auth_required(tmp_path, monkeypatch):
    config = setup_config(tmp_path, monkeypatch, password="secret")
    admin = WebAdmin(config, host="127.0.0.1", port=0)
    client = admin.app.test_client()

    assert client.get("/config").status_code == 401
    assert client.post("/reload").status_code == 401

    resp = client.get("/config", headers=auth_header("secret"))
    assert resp.status_code == 200


def test_config_endpoints(tmp_path, monkeypatch):
    config = setup_config(tmp_path, monkeypatch, password="secret")
    admin = WebAdmin(config, host="127.0.0.1", port=0)
    client = admin.app.test_client()
    headers = auth_header("secret")

    resp = client.get("/config", headers=headers)
    assert resp.status_code == 200
    assert resp.get_json()["cycle_duration"] == 4.0

    reload_called = []

    def wrapped_reload():
        reload_called.append(True)

    monkeypatch.setattr(config, "reload", wrapped_reload)

    resp = client.post("/config", json={"cycle_duration": 6.0}, headers=headers)
    assert resp.status_code == 200
    assert reload_called
    assert "cycle_duration: 6.0" in config.config_path.read_text()

    reload_called.clear()
    resp = client.post("/reload", headers=headers)
    assert resp.status_code == 200
    assert reload_called
