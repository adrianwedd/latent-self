import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import yaml
import pytest
from pydantic import ValidationError
from config_schema import AppConfig, DirectionsConfig
import json
import jsonschema


def test_app_config_valid():
    with open('data/config.yaml') as f:
        data = yaml.safe_load(f)
    config = AppConfig(**data)
    assert config.cycle_duration > 0
    assert 'age' in config.blend_weights.model_dump()
    assert len(config.eye_tracker.left_eye) == 2
    assert len(config.eye_tracker.right_eye) == 2

def test_json_schema_validation():
    with open('data/config_schema.json') as sf:
        schema = json.load(sf)
    with open('data/config.yaml') as cf:
        data = yaml.safe_load(cf)
    jsonschema.validate(data, schema)


def test_app_config_invalid_cycle_duration():
    data = {'cycle_duration': -1}
    cfg = AppConfig(**data)
    assert cfg.cycle_duration == -1


def test_directions_valid():
    with open('data/directions.yaml') as f:
        data = yaml.safe_load(f)
    dirs = DirectionsConfig(root=data)
    assert 'age' in dirs.to_dict()


def test_directions_invalid():
    data = {'age': {'label': 'Age', 'max_magnitude': 'high'}}
    with pytest.raises(ValidationError):
        DirectionsConfig(root=data)
