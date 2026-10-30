from enum import Enum

from med_vlm.enums import ModelName


class TaskType(str, Enum):
    VQA_GENERATION = "vqa_generation"
    RETRIEVAL = "retrieval"


MODEL_TASK_TYPE: dict[ModelName, TaskType] = {
    ModelName.BLIP: TaskType.VQA_GENERATION,
    ModelName.MEDVLM: TaskType.VQA_GENERATION,
    ModelName.MEDCLIP: TaskType.RETRIEVAL,
    ModelName.BIOMEDCLIP: TaskType.RETRIEVAL,
    ModelName.GRANITE_VISION: TaskType.VQA_GENERATION,
    ModelName.QWEN2VL: TaskType.VQA_GENERATION,
    ModelName.SMOLVLM: TaskType.VQA_GENERATION,
    ModelName.LLAVA_MED: TaskType.VQA_GENERATION,
    ModelName.MEDVLTHINKER: TaskType.VQA_GENERATION,
    ModelName.PUBMEDCLIP: TaskType.RETRIEVAL,
    ModelName.RAD_DINO: TaskType.RETRIEVAL,
    ModelName.PHIKON_V2: TaskType.RETRIEVAL,
}
