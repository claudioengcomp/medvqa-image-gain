"""v7_mimic/scripts/02_png_from_jpg.py -- 2026-10-10 -- lasid-gpu1 (so CPU)

PNG derivado do MIMIC-CXR-JPG com o MESMO pipeline do heal_png: reutiliza, sem editar,
harness/isbi_occlusion/heal/dicom_to_png.py (commit 2b7e731 do repo occlusion_study) -- run() e convert_one()
(lado maior --long-side=1024, LANCZOS, PNG optimize, manifesto com sha256/tamanhos/escala).
Unica diferenca: a fonte e JPG 8 bits, nao DICOM. Por isso (a) o leitor dicom_to_gray8 e substituido em
tempo de execucao (monkeypatch neste processo; workers herdam por fork) por um leitor PIL que devolve o
cinza 8 bits tal como esta no JPG (sem VOI LUT, sem min-max -- os JPG ja sao 8 bits janelados pelos
autores do MIMIC-CXR-JPG; min/max nativos ficam registrados no manifesto), e (b) run() espera
<dicom_dir>/<id>.dicom, entao criamos uma arvore de SYMLINKS <stage>/<dicom_id>.dicom -> JPG original.
Nada e copiado; o JPG original nao e tocado.

  python 02_png_from_jpg.py --items data/items_test.parquet --split test --jpg-root <raw mimic-cxr-jpg/2.1.0> \
      --stage tmp/v7_mimic/stage/test --out-dir <derived>/mimic_png/test --manifest <derived>/mimic_png/png_manifest_test.json --workers 8
"""
from __future__ import annotations
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, json, os, sys, time, socket
from pathlib import Path
import numpy as np
import pandas as pd

OCCLUSION_STUDY = f"{_MEDVQA_ROOT}/doutorado/medvqa-occlusion-study"
sys.path.insert(0, OCCLUSION_STUDY)
import scripts_credentialed.heal.dicom_to_png as dtp  # noqa: E402  (so leitura; nao editado)


def jpg_to_gray8(path: str):
    from PIL import Image
    im = Image.open(path)
    mode = im.mode
    if mode != "L":
        im = im.convert("L")
    g = np.asarray(im, dtype=np.uint8)
    info = {"rows": int(g.shape[0]), "cols": int(g.shape[1]), "photometric": f"JPEG:{mode}", "bits_stored": 8,
            "transfer_syntax": "JPEG (MIMIC-CXR-JPG 2.1.0)", "window": "jpeg_as_is", "inverted": False,
            "native_min": int(g.min()), "native_max": int(g.max())}
    return g, info


dtp.dicom_to_gray8 = jpg_to_gray8  # monkeypatch em memoria; o arquivo do repositorio nao muda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--jpg-root", required=True); ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--long-side", type=int, default=1024); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_parquet(a.items, columns=["dicom_id", "image_path"]).drop_duplicates("dicom_id")
    Path(a.stage).mkdir(parents=True, exist_ok=True); Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    missing = []
    for d, p in zip(df.dicom_id, df.image_path):
        src = Path(a.jpg_root, p)
        if not src.is_file():
            missing.append(p); continue
        lnk = Path(a.stage, f"{d}.dicom")
        if not lnk.is_symlink():
            lnk.symlink_to(src)
    if missing:
        print(json.dumps({"missing_jpg": len(missing), "first": missing[:5]})); sys.exit(2)
    ids = sorted(df.dicom_id)
    t0 = time.time()
    r = dtp.run(a.stage, ids, a.out_dir, a.manifest, a.long_side, a.workers)
    man = json.loads(Path(a.manifest).read_text())
    man["provenance"] = {"source": "MIMIC-CXR-JPG 2.1.0 (PhysioNet) JPG -> PNG", "split": a.split, "items_parquet": a.items,
                         "script": "v7_mimic/scripts/02_png_from_jpg.py", "reused": f"{OCCLUSION_STUDY}/harness/isbi_occlusion/heal/dicom_to_png.py @2b7e731 (run, convert_one)",
                         "reader": "PIL JPEG -> L 8 bits, sem VOI LUT/min-max (jpeg_as_is)", "long_side": a.long_side,
                         "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(), "seconds": round(time.time() - t0, 1)}
    Path(a.manifest).write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
