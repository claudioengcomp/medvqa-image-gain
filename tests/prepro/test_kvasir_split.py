import pytest

from harness.prepro.kvasir_split import split_records_by_image


def _record(img_id, pathology_label):
    return {
        "img_id": img_id,
        "image_url": f"https://example.com/{img_id}.jpg",
        "questions": ["q1", "q2"],
        "answers": ["a1", "a2"],
        "pathology_label": pathology_label,
    }


def test_split_covers_every_record_exactly_once():
    records = [_record(f"img{i}", "polyp" if i % 3 == 0 else "normal") for i in range(100)]
    result = split_records_by_image(records, ratios=(0.7, 0.1, 0.2), seed=42)

    assert set(result.keys()) == {"train", "val", "test"}
    all_ids = [r["img_id"] for split in result for r in result[split]]
    assert sorted(all_ids) == sorted(r["img_id"] for r in records)
    assert len(all_ids) == len(records)


def test_split_has_zero_overlap_between_splits():
    records = [_record(f"img{i}", "polyp" if i % 3 == 0 else "normal") for i in range(100)]
    result = split_records_by_image(records, ratios=(0.7, 0.1, 0.2), seed=42)

    train_ids = {r["img_id"] for r in result["train"]}
    val_ids = {r["img_id"] for r in result["val"]}
    test_ids = {r["img_id"] for r in result["test"]}
    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()


def test_split_respects_approximate_ratios():
    records = [_record(f"img{i}", "polyp" if i % 3 == 0 else "normal") for i in range(1000)]
    result = split_records_by_image(records, ratios=(0.7, 0.1, 0.2), seed=42)

    n = len(records)
    assert abs(len(result["train"]) / n - 0.7) < 0.02
    assert abs(len(result["val"]) / n - 0.1) < 0.02
    assert abs(len(result["test"]) / n - 0.2) < 0.02


def test_split_is_deterministic_given_same_seed():
    records = [_record(f"img{i}", "polyp" if i % 3 == 0 else "normal") for i in range(100)]
    result_a = split_records_by_image(records, ratios=(0.7, 0.1, 0.2), seed=42)
    result_b = split_records_by_image(records, ratios=(0.7, 0.1, 0.2), seed=42)

    assert [r["img_id"] for r in result_a["train"]] == [r["img_id"] for r in result_b["train"]]


def test_split_stratifies_rare_label_across_all_three_splits():
    # 6 imagens raras (label "rare") -- estratificado, cada split deve ganhar
    # pelo menos 1 (sem estratificação, sorteio livre poderia jogar todas num só split).
    records = [_record(f"rare{i}", "rare") for i in range(6)]
    records += [_record(f"common{i}", "normal") for i in range(300)]
    result = split_records_by_image(records, ratios=(0.7, 0.1, 0.2), seed=42)

    rare_per_split = {
        split: sum(1 for r in recs if r["pathology_label"] == "rare")
        for split, recs in result.items()
    }
    assert all(count >= 1 for count in rare_per_split.values())


def test_split_rejects_ratios_not_summing_to_one():
    records = [_record("img0", "normal")]
    with pytest.raises(ValueError, match="ratios"):
        split_records_by_image(records, ratios=(0.7, 0.1, 0.1), seed=42)
