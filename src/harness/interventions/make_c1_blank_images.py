"""C1: gera uma cópia do arquivo .arrow de teste com as imagens substituídas
por uma imagem neutra (cor constante), preservando todas as outras colunas.

Uso:
    python -m harness.interventions.make_c1_blank_images \
        --input data/finetune_arrows/vqa_slack_test.arrow \
        --output data/finetune_arrows/vqa_slack_test_c1.arrow
"""
from __future__ import annotations

import argparse
import io

from PIL import Image

from harness.interventions.arrow_io import read_arrow, write_arrow

DEFAULT_SIZE = 384
DEFAULT_FILL_COLOR = (128, 128, 128)


def make_blank_jpeg_bytes(size: int = DEFAULT_SIZE, fill_color=DEFAULT_FILL_COLOR) -> bytes:
    """Gera os bytes JPEG de uma imagem quadrada de cor constante."""
    img = Image.new("RGB", (size, size), fill_color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def blank_out_images(input_path: str, output_path: str, size: int = DEFAULT_SIZE,
                      fill_color=DEFAULT_FILL_COLOR) -> None:
    df = read_arrow(input_path)
    blank_bytes = make_blank_jpeg_bytes(size, fill_color)
    df = df.copy()
    df["image"] = [blank_bytes] * len(df)
    write_arrow(df, output_path)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    blank_out_images(args.input, args.output, size=args.size)
    print(f"C1 escrito em {args.output}")
