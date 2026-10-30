# Rebuilding the HEAL-MedVQA and Medical-Diff-VQA material from the official files

Nothing derived from these two benchmarks is distributed (no image, no question or answer text, no identifier). A user with
PhysioNet credentials rebuilds, with the scripts here, exactly the subset, the split and the test conditions used in the paper.
Seeds are fixed in the scripts; the sha256 of every file we built is recorded in `results/aggregates/*` and in the numbers sheet,
so a rebuild can be checked file by file. `selftest_synthetic.py` runs the pipeline on fabricated inputs (schema only) and is
part of the repository's tests.

## HEAL-MedVQA (images from VinDr-CXR, PhysioNet)

1. `heal/heal_ids.py` — image identifiers used by HEAL-MedVQA, read from the HEAL-MedVQA annotation parquet files (column projection; images are never read).
2. `heal/vindr_split_ids.py` — which VinDr-CXR partition (train/test) each identifier belongs to, from `image_labels_{train,test}.csv` of VinDr-CXR.
3. `heal/dicom_to_png.py` — DICOM → 8-bit PNG (longest side 1024, VOI LUT or min–max windowing, MONOCHROME1 inverted), with a manifest (sha256, sizes, scale).
4. `heal/inventory_heal.py` — the closed-form test questions of the official HEAL-MedVQA test split, with the answer normalisation (`normalize_answer`) and the "evaluable" flag that defines the 1 446 test questions of the paper (closed-form questions whose anatomy mask admits a valid placement for the companion occlusion study; open questions are excluded).
5. `heal/make_heal_train_arrows.py` — training and validation arrows (M3AE arrow schema): atomic answers, 10 % of the training images held out for validation (stratified by anatomy and presence of a finding, seed 20260913), answer vocabulary with `no`=0, `yes`=1.
6. `heal/make_grey_image_test_arrow_heal.py` — grey-image test arrow (condition C1).
7. `image_swap_v2.py heal` — image swap by radiograph (condition C2v2, seed 42); `image_swap_heal_mimic_original.py` is the script that actually ran (hard-coded paths, kept for the record).
8. `heal/tfidf_classifier_heal.py`, `heal/tfidf_align_heal.py` — TF-IDF question classifier trained on the training split only.

The HEAL-MedVQA training/validation split by image is reproduced by step 5; the test arrow (condition C0) is the closed-form
evaluable questions of step 4 with the PNGs of step 3 (the companion script `make_heal_conditions.py` of the occlusion study builds it; its C0 output is what we used).

## Medical-Diff-VQA (images from MIMIC-CXR-JPG, PhysioNet)

1. `medical_diff_vqa/00_schema_stats.py` — schema statistics and the entity list (`entities.json`) from `all_diseases.json`.
2. `medical_diff_vqa/01b_make_subset.py` — the single-image subset: difference questions removed; closed = presence and yes/no view questions; open = location/level/type with a fixed vocabulary; one image per study, PA/AP only; split by patient from `mimic-cxr-2.0.0-split.csv.gz`; stratified sampling by template × answer, seed 20260913 (sizes per split in the numbers sheet, section 8, and in the report the script writes). Raw-table and output roots can be overridden with `MEDDIFF_RAW`, `MIMICCXR_RAW`, `MIMIC_OUT` (used by the self-test).
3. `medical_diff_vqa/02_png_from_jpg.py` — JPG → PNG with the same pipeline as HEAL (longest side 1024).
4. `medical_diff_vqa/05_make_arrows.py`, `05b_validate_arrows.py` — arrows per split, answer vocabulary.
5. `medical_diff_vqa/make_grey_image_closed_test_arrow_mimic.py` — the 1 499 closed-form test questions with the grey image (C1); `make_grey_training_arrows_mimic.py` — grey training/validation arrows of the grey M3AE text variant.
6. `image_swap_v2.py mimic` — image swap with same-patient exclusion (C2v2, seed 42).
7. `medical_diff_vqa/06_tfidf_classifier_mimic.py` — TF-IDF question classifier.

The 1 499 closed-form test questions of the paper are the `answer_type == closed` rows of `items_test.parquet` produced by step 2.
