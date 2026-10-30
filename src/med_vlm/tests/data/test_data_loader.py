from unittest.mock import patch

from med_vlm import enums
from med_vlm.data.loader import load_dataset_by_enum


def test_load_dataset_by_enum_default_split(mock_dataset):
    with patch("med_vlm.data.loader.load_dataset", return_value=mock_dataset):
        with patch("med_vlm.data.loader.build_normalizer") as mock_bn:
            mock_bn.return_value = mock_dataset.map.return_value
            result = load_dataset_by_enum(enums.DatasetName.VQA_RAD)

            assert result == mock_dataset
            mock_dataset.map.assert_called_once()
            assert (
                mock_dataset.map.call_args[1]["remove_columns"]
                == mock_dataset.column_names
            )


def test_load_dataset_by_enum_custom_split(mock_dataset):
    with patch("med_vlm.data.loader.load_dataset", return_value=mock_dataset):
        with patch("med_vlm.data.loader.build_normalizer"):
            result = load_dataset_by_enum(enums.DatasetName.VQA_RAD, split="test")

            assert result == mock_dataset


def test_load_dataset_by_enum_applies_normalizer(mock_dataset, mock_normalizer):
    with patch("med_vlm.data.loader.load_dataset", return_value=mock_dataset):
        with patch(
            "med_vlm.data.loader.build_normalizer", return_value=mock_normalizer
        ):
            load_dataset_by_enum(enums.DatasetName.VQA_RAD)

            mock_dataset.map.assert_called_once()
            call_kwargs = mock_dataset.map.call_args[1]
            assert call_kwargs["remove_columns"] == mock_dataset.column_names


def test_load_dataset_by_enum_removes_columns(mock_dataset):
    with patch("med_vlm.data.loader.load_dataset", return_value=mock_dataset):
        with patch("med_vlm.data.loader.build_normalizer"):
            load_dataset_by_enum(enums.DatasetName.VQA_RAD)

            call_kwargs = mock_dataset.map.call_args[1]
            assert "remove_columns" in call_kwargs
            assert call_kwargs["remove_columns"] == mock_dataset.column_names


def test_load_dataset_by_enum_dataset_enum_value_passed(mock_dataset):
    with patch(
        "med_vlm.data.loader.load_dataset", return_value=mock_dataset
    ) as mock_load:
        with patch("med_vlm.data.loader.build_normalizer"):
            load_dataset_by_enum(enums.DatasetName.VQA_RAD, split="validation")

            mock_load.assert_called_once_with(
                enums.DatasetName.VQA_RAD.value, split="validation"
            )


from unittest.mock import MagicMock, patch


def test_load_dataset_by_enum_kvasir_uses_clean_arrow(mock_dataset):
    with patch(
        "med_vlm.data.loader.load_condition_dataset", return_value=mock_dataset
    ) as mock_load_condition:
        with patch("med_vlm.data.loader.build_normalizer") as mock_bn:
            mock_bn.return_value = mock_dataset.map.return_value
            result = load_dataset_by_enum(enums.DatasetName.KVASIR_VQA, split="train")

            mock_load_condition.assert_called_once_with(
                "data/finetune_arrows/vqa_kvasir_train_clean.arrow"
            )
            assert result == mock_dataset.map.return_value


def test_load_dataset_by_enum_kvasir_validation_uses_val_arrow(mock_dataset):
    with patch(
        "med_vlm.data.loader.load_condition_dataset", return_value=mock_dataset
    ) as mock_load_condition:
        with patch("med_vlm.data.loader.build_normalizer"):
            load_dataset_by_enum(enums.DatasetName.KVASIR_VQA, split="validation")

            mock_load_condition.assert_called_once_with(
                "data/finetune_arrows/vqa_kvasir_val_clean.arrow"
            )


def test_load_dataset_by_enum_kvasir_test_uses_test_arrow(mock_dataset):
    with patch(
        "med_vlm.data.loader.load_condition_dataset", return_value=mock_dataset
    ) as mock_load_condition:
        with patch("med_vlm.data.loader.build_normalizer"):
            load_dataset_by_enum(enums.DatasetName.KVASIR_VQA, split="test")

            mock_load_condition.assert_called_once_with(
                "data/finetune_arrows/vqa_kvasir_test_clean.arrow"
            )


def test_load_dataset_by_enum_kvasir_normalizes_with_plural_answers_key(mock_dataset):
    with patch(
        "med_vlm.data.loader.load_condition_dataset", return_value=mock_dataset
    ):
        with patch("med_vlm.data.loader.build_normalizer") as mock_bn:
            mock_bn.return_value = mock_dataset.map.return_value
            load_dataset_by_enum(enums.DatasetName.KVASIR_VQA, split="train")

            mock_bn.assert_called_once_with(
                image_key="image",
                question_key="question",
                answers_key="answers",
                id_key="id",
                preprocessor=None,
            )
