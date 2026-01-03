import base64
from pathlib import Path


def decode_images(image_dir: Path) -> None:
    for b64_path in image_dir.glob('*.b64'):
        out_path = b64_path.with_suffix('')
        data = base64.b64decode(b64_path.read_bytes())
        out_path.write_bytes(data)
        print(f"Decoded {b64_path} -> {out_path}")


def main() -> None:
    decode_images(Path('docs/images'))


if __name__ == '__main__':
    main()
