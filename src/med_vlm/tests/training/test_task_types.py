import pytest

from med_vlm.enums import ModelName
from med_vlm.training.task_types import MODEL_TASK_TYPE, TaskType


def test_task_type_vqa_generation_value():
    assert TaskType.VQA_GENERATION.value == "vqa_generation"


def test_task_type_retrieval_value():
    assert TaskType.RETRIEVAL.value == "retrieval"


def test_task_type_is_string_enum():
    # if TaskType inherits from str, Enum (i.e. class TaskType(str, Enum))
    assert isinstance(TaskType.VQA_GENERATION, str)
    assert isinstance(TaskType.RETRIEVAL, str)


@pytest.mark.parametrize(
    "model,expected_task_type",
    [
        (ModelName.BLIP, TaskType.VQA_GENERATION),
        (ModelName.MEDVLM, TaskType.VQA_GENERATION),
        (ModelName.MEDCLIP, TaskType.RETRIEVAL),
        (ModelName.BIOMEDCLIP, TaskType.RETRIEVAL),
    ],
)
def test_model_task_type_mapping(model, expected_task_type):
    assert MODEL_TASK_TYPE[model] == expected_task_type


def test_model_task_type_all_models_mapped():
    # MODEL_TASK_TYPE is not read by the training or inference code. The models added to
    # ModelName after the mapping was written have no entry; they are listed here so that
    # any other model without an entry still fails the test.
    added_later = {ModelName.QWEN25VL, ModelName.LINGSHU, ModelName.HUATUO_QWEN25,
                   ModelName.INTERNVL3, ModelName.GEMMA3, ModelName.MEDGEMMA}
    assert set(MODEL_TASK_TYPE.keys()) <= set(ModelName)
    assert set(ModelName) - set(MODEL_TASK_TYPE.keys()) == added_later


def test_model_task_type_groups_are_correct():
    vqa_models = {ModelName.BLIP, ModelName.MEDVLM}
    retrieval_models = {ModelName.MEDCLIP, ModelName.BIOMEDCLIP}

    for model in vqa_models:
        assert MODEL_TASK_TYPE[model] == TaskType.VQA_GENERATION

    for model in retrieval_models:
        assert MODEL_TASK_TYPE[model] == TaskType.RETRIEVAL
