"""Helpers shared by the harness tests: build synthetic .arrow files in the
same schema as make_arrow_vqa of LaPA/M3AE."""
import io

import pandas as pd
from PIL import Image

from harness.interventions.arrow_io import write_arrow


def sample_rows(n: int = 2) -> dict:
    return {
        "image": [f"IMG{i}".encode() for i in range(n)],
        "questions": [[f"Q{i}a", f"Q{i}b"] for i in range(n)],
        "answers": [[[f"ans{i}a"], [f"ans{i}b"]] for i in range(n)],
        "answer_labels": [[[i * 2], [i * 2 + 1]] for i in range(n)],
        "answer_scores": [[[1.0], [1.0]] for i in range(n)],
        "image_id": [f"imgs/xmlab{i}/source.jpg" for i in range(n)],
        "question_id": [[i * 2, i * 2 + 1] for i in range(n)],
        "answer_type": [[0, 0] for i in range(n)],
        "split": ["test"] * n,
    }


def write_sample_arrow(path: str, n: int = 2) -> None:
    write_arrow(pd.DataFrame(sample_rows(n)), path)


def sample_rows_with_real_images(n: int = 2, size: int = 64) -> dict:
    """Like sample_rows(), but the 'image' column holds real JPEG bytes
    (needed by tests that must OPEN the image, not only overwrite it)."""
    rows = sample_rows(n)
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    images = []
    for i in range(n):
        img = Image.new("RGB", (size, size), colors[i % len(colors)])
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        images.append(buf.getvalue())
    rows["image"] = images
    return rows


def write_sample_arrow_with_real_images(path: str, n: int = 2, size: int = 64) -> None:
    write_arrow(pd.DataFrame(sample_rows_with_real_images(n, size=size)), path)
