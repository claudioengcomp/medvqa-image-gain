from unittest.mock import MagicMock, Mock

import pytest

from med_vlm.models.loader import load_model_bundle


@pytest.fixture
def mock_model_config():
    config = Mock()

    config.model_cls = Mock()
    config.model_cls.from_pretrained.return_value = Mock(name="model")

    config.tokenizer_cls = Mock()
    config.tokenizer_cls.from_pretrained.return_value = Mock(name="tokenizer")

    config.processor_cls = Mock()
    config.processor_cls.from_pretrained.return_value = Mock(name="processor")

    return config


@pytest.fixture
def mock_model():
    return Mock()


@pytest.fixture
def mock_dataset():
    ds = MagicMock()
    ds.column_names = ["question", "image", "answer"]
    ds.map.return_value = ds
    return ds


@pytest.fixture
def mock_normalizer():
    return Mock(return_value={"normalized": True})


@pytest.fixture(autouse=True)
def clear_model_loader_cache():
    load_model_bundle.cache_clear()
    yield
    load_model_bundle.cache_clear()
