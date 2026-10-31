import csv
import json
import os

from harness.prepro.build_kvasir_clean_split import build_clean_arrows
from harness.prepro.kvasir_split import split_records_by_image
from harness.prepro.prepare_kvasir_arrow import build_image_records


def _fake_hf_row(img_id, pathology, extra_q_a):
    original = [{"q": "Are there any abnormalities in the image? Check all that are present.", "a": pathology}]
    original += extra_q_a
    import json as _json
    return {
        "img_id": img_id,
        "image": f"https://example.com/{img_id}.jpg",
        "original": _json.dumps(original),
    }


def test_build_clean_arrows_writes_three_files_with_zero_overlap(tmp_path):
    train_rows = [_fake_hf_row(f"img{i}", "polyp" if i % 3 == 0 else "normal",
                                [{"q": "Is there a green/black box artefact?", "a": "no"}])
                  for i in range(60)]
    # 30 images from the HF "test" -- 20 of them repeat img_id from the HF "train" (simulates the
    # official leakage), only 10 are genuinely new.
    test_rows = [_fake_hf_row(f"img{i}", "polyp" if i % 3 == 0 else "normal",
                               [{"q": "Is there a green/black box artefact?", "a": "no"}])
                 for i in range(40, 70)]
    # Image whose category template-question came with an empty answer --
    # exercises the exclusion of this case from the labels JSON (Issue 1 of the review).
    train_rows.append(_fake_hf_row("img_no_pathology", "",
                                    [{"q": "Is there a green/black box artefact?", "a": "no"}]))

    def fake_download(url):
        return b"\xff\xd8fake-jpeg-bytes"

    manifest = build_clean_arrows(
        train_rows=train_rows,
        test_rows=test_rows,
        output_dir=str(tmp_path),
        ratios=(0.7, 0.1, 0.2),
        seed=42,
        download_fn=fake_download,
    )

    assert set(manifest["counts"].keys()) == {"train", "val", "test"}
    total_images = sum(manifest["counts"].values())
    # 71 unique img_id in total (60 from train + 10 new from test, 20 repeated,
    # + 1 without pathology_label)
    assert total_images == 71

    for split in ("train", "val", "test"):
        assert os.path.exists(f"{tmp_path}/vqa_kvasir_{split}_clean.arrow")
    assert os.path.exists(f"{tmp_path}/kvasir_clean_split_map.csv")
    assert os.path.exists(f"{tmp_path}/vqa_kvasir_clean_pathology_labels.json")

    all_img_ids = {row["img_id"] for row in train_rows} | {row["img_id"] for row in test_rows}

    # CSV content, not just existence: every img_id of the split map belongs
    # to the set of img_id of the fixtures, and every split is one of the 3 expected.
    with open(f"{tmp_path}/kvasir_clean_split_map.csv") as fp:
        csv_rows = list(csv.DictReader(fp))
    csv_map = {r["img_id"]: r["split"] for r in csv_rows}
    assert set(csv_map) == all_img_ids
    assert set(csv_map.values()) <= {"train", "val", "test"}

    # Confere valores ESPECÍFICOS de img_id->split, comparando contra o
    # direct computation from the same pure functions (build_image_records +
    # split_records_by_image, same seed) -- not just the aggregate count.
    records = build_image_records(train_rows + test_rows)
    expected_assignment = split_records_by_image(records, ratios=(0.7, 0.1, 0.2), seed=42)
    expected_map = {
        rec["img_id"]: split_name
        for split_name, split_records in expected_assignment.items()
        for rec in split_records
    }
    assert csv_map == expected_map

    # Content of the pathology labels JSON (consumed by C5 downstream):
    # correct "kvasir/" prefix, correct value, and exclusion of empty label.
    with open(f"{tmp_path}/vqa_kvasir_clean_pathology_labels.json") as fp:
        labels = json.load(fp)
    assert labels["kvasir/img0"] == "polyp"  # i=0, 0 % 3 == 0
    assert labels["kvasir/img1"] == "normal"  # i=1, 1 % 3 != 0
    assert "kvasir/img_no_pathology" not in labels


def test_build_clean_arrows_split_map_has_no_duplicate_img_id(tmp_path):
    train_rows = [_fake_hf_row(f"img{i}", "normal",
                                [{"q": "Is there a green/black box artefact?", "a": "no"}])
                  for i in range(30)]

    def fake_download(url):
        return b"\xff\xd8fake-jpeg-bytes"

    build_clean_arrows(
        train_rows=train_rows, test_rows=[], output_dir=str(tmp_path),
        ratios=(0.7, 0.1, 0.2), seed=42, download_fn=fake_download,
    )

    with open(f"{tmp_path}/kvasir_clean_split_map.csv") as fp:
        rows = list(csv.DictReader(fp))
    img_ids = [r["img_id"] for r in rows]
    assert len(img_ids) == len(set(img_ids))
    assert len(img_ids) == 30
    assert set(img_ids) == {row["img_id"] for row in train_rows}
    assert {r["split"] for r in rows} <= {"train", "val", "test"}
