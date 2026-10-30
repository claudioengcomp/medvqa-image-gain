"""DICOM of VinDr-CXR -> 8-bit PNG, grey scale, longer side = --long-side (default 1024),
lossless, with manifest (id, native size, PNG size, scale, PNG sha256, window used).
Window: VOI LUT / WindowCenter+Width of the DICOM when they exist (pydicom.pixels.apply_voi_lut), otherwise
min-max; MONOCHROME1 is inverted to MONOCHROME2. Processes in parallel (CPU), skips PNGs already
existing with sha256 in the manifest. No DICOM is copied; only the derived PNG leaves the Mac.

    python -m scripts_credentialed.heal.dicom_to_png --dicom-dir <base>/train --ids data/isbi_occlusion/heal/ids_heal_train.txt \
        --out-dir <png_dir> --manifest data/isbi_occlusion/heal/png_manifest_train.json --long-side 1024 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def dicom_to_gray8(path: str) -> tuple[np.ndarray, dict]:
    import pydicom
    from pydicom.pixels import apply_voi_lut
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    info = {"rows": int(ds.Rows), "cols": int(ds.Columns), "photometric": str(ds.get("PhotometricInterpretation", "")),
            "bits_stored": int(ds.get("BitsStored", 0) or 0), "transfer_syntax": str(ds.file_meta.TransferSyntaxUID.name)}
    try:
        v = apply_voi_lut(arr, ds); info["window"] = "voi_lut"
    except Exception:
        v = arr; info["window"] = "none"
    v = v.astype(np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        g = np.zeros_like(v, dtype=np.uint8)
    else:
        g = np.round((v - lo) / (hi - lo) * 255.0).astype(np.uint8)
    if info["photometric"] == "MONOCHROME1":
        g = 255 - g; info["inverted"] = True
    else:
        info["inverted"] = False
    return g, info


def convert_one(args: tuple[str, str, int]) -> dict:
    src, dst, long_side = args
    from PIL import Image
    g, info = dicom_to_gray8(src)
    h, w = g.shape
    scale = long_side / max(h, w) if max(h, w) > long_side else 1.0
    out_w, out_h = (int(round(w * scale)), int(round(h * scale))) if scale < 1.0 else (w, h)
    im = Image.fromarray(g)
    if scale < 1.0:
        im = im.resize((out_w, out_h), Image.LANCZOS)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="PNG", optimize=True)
    data = Path(dst).read_bytes()
    return {"id": Path(src).stem, "png": os.path.basename(dst), "native_h": h, "native_w": w, "png_h": out_h, "png_w": out_w,
            "scale": scale, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **info}


def run(dicom_dir: str, ids: list[str], out_dir: str, manifest: str, long_side: int, workers: int) -> dict:
    man = json.loads(Path(manifest).read_text()) if Path(manifest).is_file() else {"long_side": long_side, "items": {}}
    if man.get("long_side") != long_side:
        raise SystemExit(f"manifest with long_side={man.get('long_side')} != {long_side}")
    todo = []
    for i in ids:
        dst = os.path.join(out_dir, f"{i}.png")
        if i in man["items"] and Path(dst).is_file() and Path(dst).stat().st_size == man["items"][i]["bytes"]:
            continue
        todo.append((os.path.join(dicom_dir, f"{i}.dicom"), dst, long_side))
    errors = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(convert_one, t): t for t in todo}
        for n, f in enumerate(as_completed(futs), 1):
            src = futs[f][0]
            try:
                r = f.result(); man["items"][r["id"]] = r
            except Exception as e:  # noqa: BLE001
                errors[Path(src).stem] = str(e)[:200]
            if n % 200 == 0 or n == len(todo):
                Path(manifest).write_text(json.dumps(man, indent=1) + "\n"); print(f"{n}/{len(todo)}", flush=True)
    man["n_items"] = len(man["items"]); man["errors"] = errors
    Path(manifest).write_text(json.dumps(man, indent=1) + "\n")
    return {"converted_now": len(todo) - len(errors), "errors": len(errors), "n_items": man["n_items"], "skipped": len(ids) - len(todo)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-dir", required=True); ap.add_argument("--ids", required=True, help="list with train/<id>.dicom or <id>")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--long-side", type=int, default=1024); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    ids = [l.strip().split("/")[-1].replace(".dicom", "") for l in Path(a.ids).read_text().splitlines() if l.strip()]
    print(json.dumps(run(a.dicom_dir, ids, a.out_dir, a.manifest, a.long_side, a.workers)))


if __name__ == "__main__":
    main()
