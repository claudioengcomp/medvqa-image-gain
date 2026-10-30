"""Self-test of scripts_credentialed/ on SYNTHETIC data (no real HEAL-MedVQA / VinDr-CXR / Medical-Diff-VQA / MIMIC-CXR content).
It fabricates tiny inputs with the schemas those scripts expect, runs the reconstruction steps and checks the output formats:
  1. HEAL-MedVQA train/val arrows (make_heal_train_arrows.build): stratified image split, atomic answers, M3AE arrow schema.
  2. Medical-Diff-VQA single-image subset (01b_make_subset.py) from synthetic raw tables -> items_{train,val,test}.parquet.
  3. Medical-Diff-VQA arrows (05_make_arrows.py) from the items tables and synthetic PNGs.
  4. Grey-image test arrow (make_c1_blank_images.blank_out_images) and the closed-only cut used for the grey baseline.
  5. Image swap (image_swap_v2.swap_heal / swap_mimic) with the paper's rules (seed 42).
Usage: python scripts_credentialed/selftest_synthetic.py [--keep <dir>]     (exit 0 = every check passed)"""
import argparse, gzip, hashlib, io, json, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(PKG, "src")); sys.path.insert(0, PKG)
import numpy as np, pandas as pd
from PIL import Image
from harness.interventions.arrow_io import read_arrow, write_arrow, count_questions, ARROW_COLUMNS
from harness.interventions.make_c1_blank_images import blank_out_images
def png(seed, size=64):
    rng = np.random.RandomState(seed); b = io.BytesIO(); Image.fromarray(rng.randint(0, 255, (size, size), dtype=np.uint8), "L").save(b, format="PNG"); return b.getvalue()
def check(cond, msg, rep):
    rep["checks"].append({"ok": bool(cond), "check": msg}); print(("ok   " if cond else "FAIL ") + msg, flush=True); return bool(cond)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keep", default=None); a = ap.parse_args()
    T = a.keep or tempfile.mkdtemp(prefix="selftest_credentialed_"); os.makedirs(T, exist_ok=True); rep = {"tmp": T, "checks": []}; env = dict(os.environ, PYTHONPATH=os.path.join(PKG, "src") + os.pathsep + PKG, MEDVQA_ROOT=T)
    # ---------- 1. HEAL train/val arrows
    from scripts_credentialed.heal.make_heal_train_arrows import build as heal_build
    H = f"{T}/heal"; os.makedirs(f"{H}/png", exist_ok=True); rng = np.random.RandomState(0); anats = ["Aorta", "Heart", "Left lower lung", "Right upper lung"]; rows = []; pm = {}
    for i in range(40):
        img = f"synth{i:03d}"; b = png(i); open(f"{H}/png/{img}.png", "wb").write(b); pm[img] = {"png": f"{img}.png", "sha256": hashlib.sha256(b).hexdigest()}
        for k in range(3):
            qt = ["closed_yes", "closed_no", "open_normal", "open_abnormal"][rng.randint(4)]; an = anats[rng.randint(len(anats))]
            ans = {"closed_yes": "Yes, it does.", "closed_no": "No, it does not.", "open_normal": "It has no abnormalities<seg>.", "open_abnormal": "It suffers from finding-A<seg>."}[qt]
            rows.append({"image_id": img, "question": f"Does the {an.lower()} suffer from finding-{k}?", "answer": ans, "question_type": qt, "anatomy": an})
    ann = pd.DataFrame(rows); ann.to_parquet(f"{H}/annotations_train.parquet", index=False); json.dump({"items": pm}, open(f"{H}/png_manifest.json", "w"))
    te = ann.iloc[:0].copy(); te.loc[0] = ["synth_test_000", "Does the heart suffer from finding-9?", "Yes, it does.", "closed_yes", "Heart"]; te.to_parquet(f"{H}/annotations_test.parquet", index=False)
    man = heal_build(f"{H}/annotations_train.parquet", f"{H}/png", f"{H}/png_manifest.json", f"{H}/out", f"{H}/manifest.json", f"{H}/annotations_test.parquet", seed=20260913, val_frac=0.10)
    tr, va = read_arrow(f"{H}/out/vqa_heal_train.arrow"), read_arrow(f"{H}/out/vqa_heal_val.arrow")
    check(list(tr.columns) == ARROW_COLUMNS and list(va.columns) == ARROW_COLUMNS, "HEAL arrows have the M3AE schema", rep)
    check(count_questions(tr) + count_questions(va) == len(ann) and not (set(tr["image_id"]) & set(va["image_id"])), "HEAL: all questions kept, train/val images disjoint", rep)
    check(set(x for row in tr["answers"] for a_ in row for x in a_) <= {"yes", "no", "none", "finding-a"}, "HEAL: answers reduced to atomic facts", rep)
    check(man["vocab_size"] >= 3 and json.load(open(f"{H}/out/heal_answer_vocab.json"))["no"] == 0, "HEAL: answer vocabulary with no=0, yes=1", rep)
    # ---------- 2. Medical-Diff-VQA subset from synthetic raw tables
    M = f"{T}/mimic"; RD, RM, V7 = f"{M}/raw_diffvqa", f"{M}/raw_cxr", f"{M}/v7"; [os.makedirs(d, exist_ok=True) for d in (RD, RM, f"{V7}/analysis", f"{V7}/data", f"{V7}/docs")]
    ents = ["pneumonia", "pleural effusion", "cardiomegaly"]; json.dump({"entities": {e: 100 for e in ents}, "no_entity": {}}, open(f"{V7}/analysis/entities.json", "w"))
    rng = np.random.RandomState(1); Q = []; A = []; META = []; SP = []; sid = 50000000; did = 0
    for subj in range(60):
        subject = 10000000 + subj; split = ["train", "train", "train", "validate", "test"][subj % 5]
        for st in range(2):
            study = sid + subj * 10 + st; dicom = f"d{did:06d}"; did += 1
            A.append({"study_id": study, "subject_id": subject, "dicom_id": dicom, "view": "PA", "split": split}); META.append({"dicom_id": dicom, "subject_id": subject, "study_id": study, "ViewPosition": "PA" if rng.rand() < 0.8 else "AP"}); SP.append({"dicom_id": dicom, "study_id": study, "subject_id": subject, "split": split})
            for e in ents:
                Q.append({"study_id": study, "subject_id": subject, "ref_id": 0, "question_type": "presence", "question": f"is there {e} in the {['left', 'right'][rng.randint(2)]} lung?", "answer": ["yes", "no"][rng.randint(2)]})
            Q.append({"study_id": study, "subject_id": subject, "ref_id": 0, "question_type": "location", "question": f"where in the image is the {ents[rng.randint(3)]} located?", "answer": ["left lung", "right lung"][rng.randint(2)]})
            Q.append({"study_id": study, "subject_id": subject, "ref_id": 0, "question_type": "view", "question": "is this pa view?", "answer": "yes"})
            Q.append({"study_id": study, "subject_id": subject, "ref_id": 1, "question_type": "difference", "question": "what has changed compared with the previous image?", "answer": "nothing"})
    pd.DataFrame(Q).to_csv(f"{RD}/mimic_pair_questions.csv", index=False); pd.DataFrame(A).to_csv(f"{RD}/mimic_all.csv", index=False)
    with gzip.open(f"{RM}/mimic-cxr-2.0.0-metadata.csv.gz", "wt") as f: pd.DataFrame(META).to_csv(f, index=False)
    with gzip.open(f"{RM}/mimic-cxr-2.0.0-split.csv.gz", "wt") as f: pd.DataFrame(SP).to_csv(f, index=False)
    r = subprocess.run([sys.executable, f"{HERE}/medical_diff_vqa/01b_make_subset.py"], env=dict(env, MEDDIFF_RAW=RD, MIMICCXR_RAW=RM, MIMIC_OUT=V7), capture_output=True, text=True)
    ok = check(r.returncode == 0 and all(os.path.exists(f"{V7}/data/items_{s}.parquet") for s in ("train", "val", "test")), "Medical-Diff-VQA: 01b_make_subset.py runs on synthetic raw tables and writes items_{train,val,test}.parquet", rep)
    if not ok: print(r.stdout[-1500:], r.stderr[-3000:]); rep["subset_stderr"] = r.stderr[-3000:]
    else:
        items = {s: pd.read_parquet(f"{V7}/data/items_{s}.parquet") for s in ("train", "val", "test")}
        need = {"qid", "subject_id", "study_id", "dicom_id", "question_type", "template_id", "question", "answer", "answer_type", "split"}
        check(all(need <= set(d.columns) for d in items.values()), "Medical-Diff-VQA: items tables carry the expected columns", rep)
        check(all(not (set(items[x].subject_id) & set(items[y].subject_id)) for x in items for y in items if x < y), "Medical-Diff-VQA: no patient in two splits", rep)
        check(all("difference" not in set(d.question_type) for d in items.values()), "Medical-Diff-VQA: difference questions excluded (single-image subset)", rep)
        # ---------- 3. arrows from the items + synthetic PNGs
        PR = f"{V7}/png";
        for s, d in items.items():
            os.makedirs(f"{PR}/{s}", exist_ok=True); pmm = {}
            for dcm in sorted(set(d.dicom_id)): b = png(hash(dcm) % 1000); open(f"{PR}/{s}/{dcm}.png", "wb").write(b); pmm[dcm] = {"png": f"{dcm}.png", "sha256": hashlib.sha256(b).hexdigest()}
            json.dump({"items": pmm}, open(f"{PR}/png_manifest_{s}.json", "w"))
        r = subprocess.run([sys.executable, f"{HERE}/medical_diff_vqa/05_make_arrows.py", "--items-dir", f"{V7}/data", "--png-root", PR, "--out-dir", f"{V7}/data"], env=env, capture_output=True, text=True)
        ok = check(r.returncode == 0 and os.path.exists(f"{V7}/data/vqa_mimic_test.arrow"), "Medical-Diff-VQA: 05_make_arrows.py writes the three arrows + vocabulary", rep)
        if not ok: print(r.stdout[-800:], r.stderr[-2000:])
        else:
            te_ar = read_arrow(f"{V7}/data/vqa_mimic_test.arrow"); check(list(te_ar.columns) == ARROW_COLUMNS and count_questions(te_ar) == len(items["test"]), "Medical-Diff-VQA: test arrow schema and question count", rep)
            # ---------- 4. grey image + closed-only cut
            blank_out_images(f"{V7}/data/vqa_mimic_test.arrow", f"{V7}/data/vqa_mimic_test_C1.arrow"); g = read_arrow(f"{V7}/data/vqa_mimic_test_C1.arrow")
            check(len({hashlib.sha256(x).hexdigest() for x in g["image"]}) == 1 and list(g["question_id"].map(list)) == list(te_ar["question_id"].map(list)), "grey image: one constant image, questions unchanged", rep)
            # ---------- 5. swaps
            from scripts_credentialed.image_swap_v2 import swap_mimic, swap_heal
            mm = swap_mimic(f"{V7}/data/vqa_mimic_test.arrow", f"{V7}/data/items_test.parquet", f"{V7}/swap", 42)
            check(mm["questions_same_image_id"] == 0 and mm["questions_same_patient"] == 0, "Medical-Diff-VQA swap: no row keeps its image, no same-patient donor", rep)
    # HEAL swap on a synthetic test arrow with image_id heal/<radiograph>#<region>
    rows = []
    for i in range(12):
        for reg in ("Aorta", "Heart"):
            rows.append({"image": png(100 + i), "questions": [f"q{i}{reg}"], "answers": [["yes"]], "answer_labels": [[1]], "answer_scores": [[1.0]], "image_id": f"heal/rad{i:03d}#{reg}", "question_id": [3000000 + 2 * i + (reg == "Heart")], "answer_type": [0], "split": "test"})
    write_arrow(pd.DataFrame(rows), f"{H}/vqa_heal_test.arrow"); from scripts_credentialed.image_swap_v2 import swap_heal
    hm = swap_heal(f"{H}/vqa_heal_test.arrow", f"{H}/swap", 42)
    check(hm["questions_same_radiograph"] == 0 and hm["n_radiographs"] == 12, "HEAL swap: derangement per radiograph, no question keeps its radiograph", rep)
    rep["passed"] = all(c["ok"] for c in rep["checks"]); json.dump(rep, open(f"{T}/selftest_report.json", "w"), indent=1)
    print(("ALL CHECKS PASSED" if rep["passed"] else "SOME CHECKS FAILED") + f" ({sum(c['ok'] for c in rep['checks'])}/{len(rep['checks'])}); report: {T}/selftest_report.json")
    if not a.keep and rep["passed"]: shutil.rmtree(T)
    sys.exit(0 if rep["passed"] else 1)
if __name__ == "__main__": main()
