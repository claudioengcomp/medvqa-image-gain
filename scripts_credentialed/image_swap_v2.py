"""Image swap (condition C2, "A - A_swap" in the paper) for the two credentialed benchmarks, as functions with explicit paths.
Same logic as image_swap_heal_mimic_original.py (the script that actually ran; kept verbatim for the record), seed 42:

HEAL-MedVQA: derangement at the level of the RADIOGRAPH. image_id has the form "heal/<radiograph>#<region>"; the unique
  radiographs are sorted, perm = derangement(n_radiographs, seed), and every row (radiograph, region) of radiograph i receives
  the image (and the image_id of the first row) of radiograph perm[i]. No question receives its own radiograph.
Medical-Diff-VQA: derangement over the rows (images) with exclusion of the SAME PATIENT: perm = derangement(n, seed); for each
  i whose donor has the same subject_id, swap perm[i] with perm[j], j scanned in the order numpy.RandomState(seed).permutation(n),
  accepting the first swap that leaves both i and j without fixed point and without same-patient donor.

Usage:
  python scripts_credentialed/image_swap_v2.py heal  --src <vqa_heal_test.arrow>  --out <dir> [--seed 42]
  python scripts_credentialed/image_swap_v2.py mimic --src <vqa_mimic_test.arrow> --items <items_test.parquet> --out <dir> [--seed 42]
Writes <out>/vqa_<name>_test_c2v2.arrow, <out>/swap_map.json (row -> receiver/donor image_id and question ids) and <out>/manifest.json.
Credentialed images never leave the credentialed user's machine; this script only reads the arrows that user built."""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
from harness.interventions.make_c2_shuffled_pairs import derangement
from harness.interventions.arrow_io import read_arrow, write_arrow, count_questions
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def _finish(name, src, df, rows_map, extra, out, seed):
    os.makedirs(out, exist_ok=True); o = f"{out}/vqa_{name}_test_c2v2.arrow"; write_arrow(df, o)
    a, b = read_arrow(src), read_arrow(o); assert len(a) == len(b) and all(list(x) == list(y) for x, y in zip(a["question_id"], b["question_id"]))
    ha = [hashlib.sha256(x).hexdigest() for x in a["image"]]; hb = [hashlib.sha256(x).hexdigest() for x in b["image"]]
    nq = lambda cond: sum(len(a["question_id"][i]) for i in range(len(a)) if cond(i))
    man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "script": os.path.basename(__file__), "seed": seed, "source": os.path.basename(src), "source_sha256": sha(src), "output": os.path.basename(o), "sha256": sha(o),
           "n_rows": len(a), "n_questions": count_questions(a), "questions_same_image_content": nq(lambda i: ha[i] == hb[i]), "questions_same_image_id": nq(lambda i: a["image_id"][i] == b["image_id"][i]), **extra}
    json.dump(rows_map, open(f"{out}/swap_map.json", "w")); man["map"] = "swap_map.json"; man["map_sha256"] = sha(f"{out}/swap_map.json")
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1); return man
def swap_heal(src, out, seed=42):
    df = read_arrow(src); rad = lambda x: x.split("/")[-1].split("#")[0]
    R = sorted({rad(x) for x in df["image_id"]}); first = {}
    for i, x in enumerate(df["image_id"]): first.setdefault(rad(x), i)
    perm = derangement(len(R), seed); donor = {R[i]: R[perm[i]] for i in range(len(R))}
    img, iid = df["image"].tolist(), df["image_id"].tolist(); new = df.copy(); new["image"] = [img[first[donor[rad(x)]]] for x in iid]; new["image_id"] = [iid[first[donor[rad(x)]]] for x in iid]
    assert all(rad(x) != rad(y) for x, y in zip(iid, new["image_id"]))
    rows = [{"row": i, "image_id": iid[i], "radiograph": rad(iid[i]), "donor_radiograph": donor[rad(iid[i])], "question_ids": [int(q) for q in df["question_id"][i]]} for i in range(len(df))]
    return _finish("heal", src, new, rows, {"rule": "derangement per radiograph", "n_radiographs": len(R), "questions_same_radiograph": sum(len(df["question_id"][i]) for i in range(len(df)) if rad(iid[i]) == rad(new["image_id"][i]))}, out, seed)
def swap_mimic(src, items_parquet, out, seed=42):
    df = read_arrow(src); it = pd.read_parquet(items_parquet); D2 = {r.dicom_id: (r.subject_id, r.study_id) for r in it.itertuples()}
    dic = lambda x: os.path.splitext(x.split("/")[-1])[0]; subj = [D2[dic(x)][0] for x in df["image_id"]]; n = len(df)
    perm = derangement(n, seed).copy(); order = np.random.RandomState(seed).permutation(n); repairs = 0
    ok = lambda i, v: v != i and subj[v] != subj[i]
    for i in range(n):
        if ok(i, perm[i]): continue
        for j in order:
            if j != i and ok(i, perm[j]) and ok(j, perm[i]): perm[i], perm[j] = perm[j], perm[i]; repairs += 1; break
        else: raise SystemExit(f"no valid swap for row {i}")
    assert all(ok(i, perm[i]) for i in range(n)) and sorted(perm) == list(range(n))
    img, iid = df["image"].tolist(), df["image_id"].tolist(); new = df.copy(); new["image"] = [img[p] for p in perm]; new["image_id"] = [iid[p] for p in perm]
    rows = [{"row": i, "image_id_receiver": iid[i], "image_id_donor": iid[perm[i]], "question_ids": [int(q) for q in df["question_id"][i]]} for i in range(n)]
    return _finish("mimic", src, new, rows, {"rule": "derangement with same-patient exclusion (subject_id from the items table)", "repairs": repairs, "questions_same_patient": sum(len(df["question_id"][i]) for i in range(n) if subj[perm[i]] == subj[i])}, out, seed)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("benchmark", choices=["heal", "mimic"]); ap.add_argument("--src", required=True); ap.add_argument("--items", default=None); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
    man = swap_heal(a.src, a.out, a.seed) if a.benchmark == "heal" else swap_mimic(a.src, a.items, a.out, a.seed)
    print(json.dumps({k: v for k, v in man.items() if not k.endswith("sha256")}, indent=1))
