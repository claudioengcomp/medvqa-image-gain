from unittest.mock import patch

import pytest

from med_vlm.data.loader import _INDIVIDUAL_DATASETS, _SPLIT_MAP, load_dataset_by_enum
from med_vlm.enums import DatasetName


def test_split_map_covers_all_individual_datasets():
    for name in _INDIVIDUAL_DATASETS:
        assert name in _SPLIT_MAP


def test_split_map_has_required_keys():
    for name, mapping in _SPLIT_MAP.items():
        assert "train" in mapping
        assert "validation" in mapping
        assert "test" in mapping


def test_load_dataset_by_enum_calls_load_dataset(mock_dataset):
    with patch(
        "med_vlm.data.loader.load_dataset", return_value=mock_dataset
    ) as mock_ld:
        load_dataset_by_enum(DatasetName.SLAKE, split="train")
        mock_ld.assert_called_once_with(DatasetName.SLAKE.value, split="train")


def test_combined_concatenates_datasets(mock_dataset):
    with (
        patch("med_vlm.data.loader.load_dataset", return_value=mock_dataset),
        patch("med_vlm.data.loader.concatenate_datasets") as mock_concat,
    ):
        mock_concat.return_value = mock_dataset
        load_dataset_by_enum(DatasetName.COMBINED, split="train")
        mock_concat.assert_called_once()
        # Should have attempted to load all 4 individual datasets
        parts = mock_concat.call_args[0][0]
        assert len(parts) == 4


def test_combined_raises_when_all_fail():
    with patch("med_vlm.data.loader.load_dataset", side_effect=Exception("network")):
        with pytest.raises(ValueError, match="No datasets could be loaded"):
            load_dataset_by_enum(DatasetName.COMBINED, split="train")
