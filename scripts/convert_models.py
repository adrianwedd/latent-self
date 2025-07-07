#!/usr/bin/env python3
"""Convert StyleGAN2 and e4e weights to ONNX or TensorRT engines."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from services import get_stylegan_generator, get_e4e_encoder


def export_onnx(weights_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Export generator and encoder to ONNX files."""
    G = get_stylegan_generator(weights_dir)
    E = get_e4e_encoder(weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g_out = out_dir / "stylegan2.onnx"
    z = torch.randn(1, G.z_dim)
    c = torch.zeros(1, G.c_dim)
    torch.onnx.export(G.cpu(), (z, c), g_out, opset_version=16, input_names=["z", "c"], output_names=["img"])
    print(f"\u2713 wrote {g_out}")

    e_out = out_dir / "e4e.onnx"
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(E.cpu(), dummy, e_out, opset_version=16, input_names=["img"], output_names=["latent"])
    print(f"\u2713 wrote {e_out}")
    return g_out, e_out


def export_tensorrt(onnx_path: Path, engine_path: Path) -> None:
    """Convert an ONNX file to a TensorRT engine."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        parser.parse(f.read())
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, config)
    with engine_path.open("wb") as f:
        f.write(engine.serialize())
    print(f"\u2713 wrote {engine_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to ONNX/TensorRT")
    parser.add_argument("--weights", default="models", help="Directory with original weights")
    parser.add_argument("--out", default="models", help="Output directory for converted models")
    parser.add_argument("--tensorrt", action="store_true", help="Also produce TensorRT engines")
    args = parser.parse_args()

    weights_dir = Path(args.weights)
    out_dir = Path(args.out)

    g_onnx, e_onnx = export_onnx(weights_dir, out_dir)

    if args.tensorrt:
        export_tensorrt(g_onnx, out_dir / "stylegan2.engine")
        export_tensorrt(e_onnx, out_dir / "e4e.engine")


if __name__ == "__main__":
    main()
