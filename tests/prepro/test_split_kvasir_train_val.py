import pytest

from tests.helpers import sample_rows
from harness.prepro.split_kvasir_train_val import split_train_val_df
import pandas as pd


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame(sample_rows(n))


def test_split_train_val_df_partitions_without_overlap_or_loss():
    df = _df(20)
    train_df, val_df = split_train_val_df(df, val_fraction=0.1, seed=42)
    assert len(train_df) + len(val_df) == 20
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(df["image_id"])


def test_split_train_val_df_respects_val_fraction_by_rounding():
    df = _df(20)
    train_df, val_df = split_train_val_df(df, val_fraction=0.1, seed=42)
    assert len(val_df) == 2  # round(20 * 0.1)


def test_split_train_val_df_is_deterministic_for_same_seed():
    df = _df(20)
    train1, val1 = split_train_val_df(df, val_fraction=0.1, seed=7)
    train2, val2 = split_train_val_df(df, val_fraction=0.1, seed=7)
    assert list(train1["image_id"]) == list(train2["image_id"])
    assert list(val1["image_id"]) == list(val2["image_id"])


def test_split_train_val_df_different_seeds_can_differ():
    df = _df(20)
    _, val_a = split_train_val_df(df, val_fraction=0.5, seed=1)
    _, val_b = split_train_val_df(df, val_fraction=0.5, seed=2)
    assert list(val_a["image_id"]) != list(val_b["image_id"])


def test_split_train_val_df_stamps_split_column():
    df = _df(10)
    train_df, val_df = split_train_val_df(df, val_fraction=0.2, seed=42)
    assert set(train_df["split"]) == {"train"}
    assert set(val_df["split"]) == {"val"}


def test_split_train_val_df_rejects_fraction_out_of_range():
    df = _df(5)
    with pytest.raises(ValueError):
        split_train_val_df(df, val_fraction=0.0, seed=42)
    with pytest.raises(ValueError):
        split_train_val_df(df, val_fraction=1.0, seed=42)
