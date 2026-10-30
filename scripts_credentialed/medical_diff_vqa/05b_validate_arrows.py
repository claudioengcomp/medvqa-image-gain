"""v7_mimic/scripts/05b_validate_arrows.py -- valida arrows com o loader do med_vlm em CPU (CUDA_VISIBLE_DEVICES="").
Usa harness.circle_data_bridge.load_condition_dataset (o mesmo que o ramo HEAL de med_vlm/data/loader.py chama) e
monta um batch com o coletor/normalizador do med_vlm quando importavel; registra em logs/validate_arrows_medvlm.json."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import json, os, sys, time, socket, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, f"{_MEDVQA_ROOT}/doutorado/medvqa-occlusion-study")
from pathlib import Path
from harness.circle_data_bridge import load_condition_dataset
from harness.interventions.arrow_io import read_arrow, count_questions
V7 = Path(f"{_MEDVQA_ROOT}/v7_mimic"); out = {"host": socket.gethostname(), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "splits": {}}
splits = sys.argv[1:] or ["test", "val"]
import torch
out["torch_cuda_available"] = bool(torch.cuda.is_available())
vocab = json.load(open(V7 / "data/answer_vocab.json"))
for s in splits:
    p = V7 / f"data/vqa_mimic_{s}.arrow"; man = json.load(open(V7 / f"data/manifest_mimic_{s}.json"))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    df = read_arrow(str(p)); ds = load_condition_dataset(str(p))
    ex = ds[0]; img = ex["image"]
    labels_ok = all(vocab[a] == l for ans, lab in zip(df.answers, df.answer_labels) for a, l in zip([x[0] for x in ans], [x[0] for x in lab]))
    batch = [ds[i] for i in range(min(8, len(ds)))]
    r = {"arrow": str(p), "sha256_matches_manifest": sha == man["file"]["sha256"], "rows_images": len(df), "n_questions": count_questions(df),
         "ds_len_equals_questions": len(ds) == count_questions(df), "features": str(ds.features), "example": {"id": ex["id"], "question": ex["question"], "answers": ex["answers"], "image_mode": img.mode, "image_size": img.size},
         "batch8_ids": [b["id"] for b in batch], "batch8_image_sizes": sorted({b["image"].size for b in batch}), "answer_labels_match_vocab": labels_ok,
         "answer_type_values": sorted({int(t) for ts in df.answer_type for t in ts}), "image_id_prefix_ok": all(i.startswith("mimic/") for i in df.image_id), "split_values": sorted(set(df.split))}
    # tenta o normalizador/coletor do med_vlm (sem modelo)
    try:
        from med_vlm.data.normalizer import build_normalizer
        norm = build_normalizer(image_key="image", question_key="question", answers_key="answers", id_key="id", preprocessor=None)
        n = norm(ds[0]); r["med_vlm_normalizer_keys"] = sorted(n.keys())
    except Exception as e:  # noqa: BLE001
        r["med_vlm_normalizer_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["splits"][s] = r
    print(json.dumps({k: v for k, v in r.items() if k != "features"}, ensure_ascii=False))
(V7 / "logs/validate_arrows_medvlm.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
