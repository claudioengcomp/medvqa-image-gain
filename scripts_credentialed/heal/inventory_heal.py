"""Inventory of the HEAL arm (decided on 13/09/2026): only CLOSED-FORM questions from the official
VinDr test of HEAL-MedVQA ("Does [anatomy] suffer from [disease]?" / "Is there [disease] in the
[anatomy]...?"), closed_yes and closed_no separated; target = mask of the item's anatomy (HF RLE,
column-major); occlusion_study = mask of ANOTHER anatomy of the same image (any test item, open
or closed-form) with area ratio in the interval [0,7; 1,4] and overlap <= 5 % of the target, preferring the
ratio closest to 1; with no candidate, horizontal mirror of the mask (contralateral); if the mirror overlaps the target (> 5 %, midline anatomies
such as aorta and heart), the Kvasir pre-registered positioner (hflip with margin, vflip,
rot180, 50 translations with seed 20260913, no overlap). **E4 (15/09):** the fallback requires >= 80 % of the
occlusion_study inside the THORAX, defined by the population atlas (thorax_atlas.py) passed in --thorax-atlas;
with no valid position, the question is dropped and counted. Without --thorax-atlas the FOV is the whole frame (pre-E4). Open ones are left out (evaluable=False), with the reason.

Output: inventory JSON in Kvasir format (questions[], units[]) with integer qids (3_000_000 +
index) and the original HEAL question_id; masks are not written here (recomputed from the RLE when
building the arrows), only the area, the occlusion_study choice and the area ratio.

    python -m scripts_credentialed.heal.inventory_heal --annotations data/isbi_occlusion/heal/heal_annotations_test.parquet \
        --png-manifest data/isbi_occlusion/heal/png_manifest_test.json --out results/isbi_occlusion/inventory_heal_test.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from scripts_credentialed.heal.heal_rle import decode_rle

QID_BASE = 3_000_000
OCCLUSION_STUDY_SEED = 20260913   # same seed as the Kvasir occlusion study
AREA_RATIO = (0.7, 1.4)
MAX_OVERLAP = 0.05
CLOSED = {"closed_yes": "closed_yes", "closed_no": "closed_no"}
_YES = re.compile(r"\byes\b", re.I); _NO = re.compile(r"\bno\b", re.I)


def normalize_answer(ans: str, qtype: str) -> str:
    """Closed-form answer -> 'yes'/'no'. The HF copy has sentences ('... The answer is Yes'); the type confirms it."""
    a = str(ans)
    if qtype == "closed_yes" or (_YES.search(a) and not _NO.search(a.split("answer is")[-1] if "answer is" in a else a)):
        return "yes"
    return "no"


def choose_occlusion_study(target: np.ndarray, candidates: list[tuple[str, np.ndarray]], fov: np.ndarray | None = None) -> dict:
    """Another anatomy of the same image with area ratio in AREA_RATIO and overlap <= MAX_OVERLAP;
    tie: ratio closest to 1. No candidate: horizontal mirror."""
    ta = float(target.sum())
    best = None
    for name, m in candidates:
        ratio = float(m.sum()) / ta if ta else 0.0
        if not (AREA_RATIO[0] <= ratio <= AREA_RATIO[1]):
            continue
        overlap = float((m & target).sum()) / ta if ta else 1.0
        if overlap > MAX_OVERLAP:
            continue
        score = abs(np.log(ratio))
        if best is None or score < best[0]:
            best = (score, name, ratio, overlap)
    if best:
        return {"method": f"anatomy:{best[1]}", "area_ratio": best[2], "overlap": best[3]}
    mirror = target[:, ::-1]
    overlap = float((mirror & target).sum()) / ta if ta else 1.0
    if overlap <= MAX_OVERLAP:
        return {"method": "hflip", "area_ratio": 1.0, "overlap": overlap}
    # midline anatomies (aorta, heart): the mirror falls onto the target itself. Fallback = positioner
    # Kvasir pre-registered (masks.place_occlusion_study): hflip with a 5 px margin, vflip, rot180, 50 translations
    # with fixed seed, FOV = whole frame, >= 80 % inside the frame, no overlap.
    from harness.isbi_occlusion.masks import place_occlusion_study
    fov_kind = "thorax_atlas" if fov is not None else "full_frame"
    fov = fov if fov is not None else np.ones_like(target, dtype=bool)
    placed, how = place_occlusion_study(target, fov, seed=OCCLUSION_STUDY_SEED, n_random=50, margin_px=5, min_fov_frac=0.8)
    if placed is None:
        return {"method": None, "area_ratio": None, "overlap": overlap, "reason": how, "fov": fov_kind}
    return {"method": how, "area_ratio": float(placed.sum()) / ta, "overlap": 0.0, "mirror_overlap": overlap, "fov": fov_kind,
            "fov_frac": float((placed & fov).sum() / placed.sum())}


def build_inventory(annotations: str, png_manifest: str | None, thorax_atlas: str | None = None) -> dict:
    import pandas as pd
    df = pd.read_parquet(annotations)
    pm = json.loads(Path(png_manifest).read_text())["items"] if png_manifest else {}
    atlas = np.load(thorax_atlas) if thorax_atlas else None
    from scripts_credentialed.heal.thorax_atlas import atlas_for_shape
    questions, units = [], []
    by_image = collections.OrderedDict()
    for i, r in enumerate(df.itertuples(index=False)):
        by_image.setdefault(r.image_id, []).append((i, r))
    qid = QID_BASE
    n_missing_png = 0
    for image_id, rows in by_image.items():
        h, w = int(rows[0][1].mask_h), int(rows[0][1].mask_w)
        fov = atlas_for_shape(atlas, h, w) if atlas is not None else None
        masks = {}
        for _, r in rows:
            key = r.anatomy
            if key not in masks:
                masks[key] = decode_rle(r.mask_rle, h, w)
        unit_qids = []
        for _, r in rows:
            closed = r.question_type in CLOSED
            target = masks[r.anatomy]
            cands = [(a, m) for a, m in masks.items() if a != r.anatomy]
            plc = choose_occlusion_study(target, cands, fov) if closed else None
            no_place = closed and plc["method"] is None
            q = {"qid": qid, "heal_question_id": r.question_id, "image_id": image_id, "question": r.question,
                 "answer": normalize_answer(r.answer, r.question_type) if closed else str(r.answer),
                 "question_type": r.question_type, "anatomy": r.anatomy,
                 "question_class": r.question_type if closed else "open", "evaluable": bool(closed) and not no_place,
                 "exclusion_reason": None if closed and not no_place else ("no valid position for the occlusion_study" if closed else "open question (out of scope)"),
                 "target_area_frac": float(target.mean()), "occlusion_study": plc, "xray": True,
                 "n_anatomies_in_image": len(masks)}
            if pm and image_id not in pm:
                n_missing_png += 1
            questions.append(q); unit_qids.append(qid); qid += 1
        units.append({"image_id": image_id, "native_h": h, "native_w": w, "anatomies": sorted(masks),
                      "png": pm.get(image_id, {}).get("png"), "png_scale": pm.get(image_id, {}).get("scale"), "qids": unit_qids})
    ev = [q for q in questions if q["evaluable"]]
    summary = {"n_questions": len(questions), "n_evaluable": len(ev), "n_images": len(units),
               "by_class": dict(collections.Counter(q["question_class"] for q in questions)),
               "occlusion_study_methods": dict(collections.Counter(str(q["occlusion_study"]["method"]).split(":")[0] for q in ev)),
               "n_closed_without_placement": sum(1 for q in questions if q["question_class"] in CLOSED and not q["evaluable"]),
               "n_missing_png": n_missing_png, "area_ratio": AREA_RATIO, "max_overlap": MAX_OVERLAP}
    summary["thorax_atlas"] = thorax_atlas
    return {"dataset": "heal_vindr_test", "annotations": annotations, "png_manifest": png_manifest, "summary": summary,
            "questions": questions, "units": units}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True); ap.add_argument("--png-manifest", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--thorax-atlas", default=None, help="E4: .npy of the thoracic atlas (thorax_atlas.py); FOV of the fallback positioner")
    a = ap.parse_args(argv)
    inv = build_inventory(a.annotations, a.png_manifest, a.thorax_atlas)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(inv, indent=1) + "\n")
    print(json.dumps(inv["summary"], indent=1))


if __name__ == "__main__":
    main()
