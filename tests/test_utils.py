import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
import types; sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("mediapipe", types.ModuleType("mediapipe"))
import io
import json
import logging

import pytest

from directions import Direction
from logging_setup import JsonFormatter, log_timing
from latent_self import _validate_args


class DummyParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def test_validate_args_valid(tmp_path):
    parser = DummyParser()
    tmp_path.mkdir(exist_ok=True)
    args = argparse.Namespace(
        camera=0,
        resolution=512,
        fps=30,
        cycle_duration=5.0,
        blend_age=0.5,
        blend_gender=None,
        blend_smile=None,
        blend_species=None,
        emotion=None,
        max_cpu_mem=None,
        max_gpu_mem=None,
        weights=tmp_path,
    )
    _validate_args(args, parser)
    assert args.weights.exists()


def test_validate_args_invalid_camera(tmp_path):
    parser = DummyParser()
    tmp_path.mkdir(exist_ok=True)
    args = argparse.Namespace(
        camera=-1,
        resolution=512,
        fps=None,
        cycle_duration=None,
        blend_age=None,
        blend_gender=None,
        blend_smile=None,
        blend_species=None,
        emotion=None,
        max_cpu_mem=None,
        max_gpu_mem=None,
        weights=tmp_path,
    )
    with pytest.raises(ValueError):
        _validate_args(args, parser)


def test_direction_lookup():
    assert Direction.from_str('age') is Direction.AGE
    assert Direction.from_key('y') is Direction.AGE
    with pytest.raises(ValueError):
        Direction.from_str('unknown')


def test_json_formatter_and_timing():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger('test_json')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    with log_timing('step'):
        logger.info('hello')

    handler.flush()
    out = stream.getvalue().strip()
    record = json.loads(out)
    assert record['message'] == 'hello'
    assert record['level'] == 'info'
