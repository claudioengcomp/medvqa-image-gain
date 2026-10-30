from typing import Any, Callable, Dict

from PIL import Image as PILImage


def build_normalizer(
    image_key: str,
    question_key: str,
    answers_key: str,
    id_key: str | None = None,
    preprocessor: Callable[[PILImage.Image], PILImage.Image] | None = None,
    drop_image: bool = False,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Builds a normalization function for VQA datasets based on the provided keys.
    Args:
        image_key (str): The key in the dataset that corresponds to the image data.
        question_key (str): The key in the dataset that corresponds to the question
        data.
        answers_key (str): The key in the dataset that corresponds to the answer data.
        id_key (str | None): The key in the dataset that corresponds to the unique
        identifier for each example. If None, the 'id' field will be set to None
        in the output.
        preprocessor: Optional callable applied to the PIL image after RGB conversion.
        Use :class:`~med_vlm.data.preprocessor.ImagePreprocessor` for CLAHE /
        denoising, or any ``PIL.Image -> PIL.Image`` function.
    Returns:
        A function that takes a dataset example as input and returns a normalized
        dictionary with the following structure:
        {
            "image": <value from image_key>,
            "question": <value from question_key>,
            "answers": <list of answers from answers_key>,
            "id": <value from id_key or None>
        }
    """

    def normalize(example: Dict[str, Any]) -> Dict[str, Any]:
        answers = example.get(answers_key)

        if isinstance(answers, str):
            answers = [answers]

        # drop_image (v7_txt, 2026-10-09): braco so-texto -- a imagem nao entra no
        # modelo, entao nao e decodificada, pre-processada nem re-encodada no cache.
        image = None if drop_image else example[image_key]
        if isinstance(image, PILImage.Image):
            image = image.convert("RGB")
            if preprocessor is not None:
                image = preprocessor(image)

        return {
            "image": image,
            "question": example[question_key],
            "answers": answers,
            "id": example.get(id_key) if id_key else None,
        }

    return normalize


DEFAULT_VQA_NORMALIZER = build_normalizer(
    image_key="image",
    question_key="question",
    answers_key="answer",
    id_key=None,
)
