import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib
import os
from typing import Callable

from datasets import concatenate_datasets, load_dataset
from PIL import Image as PILImage

import med_vlm.enums
from harness.circle_data_bridge import load_condition_dataset
from med_vlm.data.normalizer import build_normalizer

# Individual datasets that compose the COMBINED virtual dataset
_INDIVIDUAL_DATASETS = [
    med_vlm.enums.DatasetName.SLAKE,
    med_vlm.enums.DatasetName.VQA_MED,
    med_vlm.enums.DatasetName.PATH_VQA,
    med_vlm.enums.DatasetName.VQA_RAD,
]

# Maps the canonical split name to the actual split available on HuggingFace.
# Some datasets only have train/test (no validation split).
_SPLIT_MAP: dict[med_vlm.enums.DatasetName, dict[str, str]] = {
    med_vlm.enums.DatasetName.SLAKE: {
        "train": "train",
        "validation": "validation",
        "test": "test",
    },
    med_vlm.enums.DatasetName.VQA_MED: {
        "train": "test",
        "validation": "test",
        "test": "test",
    },
    med_vlm.enums.DatasetName.PATH_VQA: {
        "train": "train",
        "validation": "test",
        "test": "test",
    },
    med_vlm.enums.DatasetName.VQA_RAD: {
        "train": "train",
        "validation": "test",
        "test": "test",
    },
    # Kvasir-VQA-x1 is intentionally absent here: it's intercepted earlier in
    # load_dataset_by_enum via _KVASIR_CLEAN_ARROW_PATH and never reaches the
    # generic load_dataset() path, so it doesn't need a split-name mapping.
}


def _normalize(
    ds,
    preprocessor: Callable[[PILImage.Image], PILImage.Image] | None = None,
    answers_key: str = "answer",
    id_key: str | None = None,
    cache_file_name: str | None = None,
    drop_image: bool = False,
):
    normalizer = build_normalizer(
        image_key="image",
        question_key="question",
        answers_key=answers_key,
        id_key=id_key,
        preprocessor=preprocessor,
        drop_image=drop_image,
    )
    map_kwargs = {
        "remove_columns": ds.column_names,
        # MED_VLM_NUM_PROC (15/09/2026): workers do map morreram no ws3 e no gpu1 com 8;
        # permite reduzir sem tocar no default.
        "num_proc": int(os.environ.get("MED_VLM_NUM_PROC", min(os.cpu_count() or 1, 8))),
    }
    if cache_file_name is not None:
        map_kwargs["cache_file_name"] = cache_file_name
    return ds.map(normalizer, **map_kwargs)


# Kvasir-VQA-x1 não usa mais o split oficial do HF Hub para o treino
# generativo -- ver docs/superpowers/specs/2026-08-30-kvasir-clean-split-design.md
# (split oficial tinha 94% de overlap de imagem treino/teste, e o campo
# `answer` do HF é composto/verboso, incompatível com o formato usado na
# avaliação). Os 3 arrows abaixo são construídos por
# harness.prepro.build_kvasir_clean_split e usam a MESMA fonte que a
# avaliação (harness.circle_data_bridge.load_condition_dataset).
_KVASIR_CLEAN_ARROW_PATH = {
    "train": "data/finetune_arrows/vqa_kvasir_train_clean.arrow",
    "validation": "data/finetune_arrows/vqa_kvasir_val_clean.arrow",
    "test": "data/finetune_arrows/vqa_kvasir_test_clean.arrow",
}

_KVASIR_PREPROCESS_CACHE_DIR = "data/finetune_arrows/preprocessed_cache"

# HEAL (E5, 15/09/2026): mesmo formato M3AE dos arrows limpos do Kvasir; train/val
# gerados por harness/isbi_occlusion/heal/make_heal_train_arrows.py, test = C0 do
# braco occlusion_study (vqa_heal_occlusion_study_C0.arrow) copiado com este nome.
_HEAL_ARROW_PATH = {
    "train": "data/finetune_arrows/vqa_heal_train.arrow",
    "validation": "data/finetune_arrows/vqa_heal_val.arrow",
    "test": "data/finetune_arrows/vqa_heal_test.arrow",
}

# MIMIC (P4, 2026-10-10): mesmo formato M3AE; arrows em v7_mimic/data (sha256 em manifest_mimic_<split>.json).
_MIMIC_ARROW_PATH = {
    "train": f"{_MEDVQA_ROOT}/v7_mimic/data/vqa_mimic_train.arrow",
    "validation": f"{_MEDVQA_ROOT}/v7_mimic/data/vqa_mimic_val.arrow",
    "test": f"{_MEDVQA_ROOT}/v7_mimic/data/vqa_mimic_test.arrow",
}


def _kvasir_preprocess_cache_path(
    arrow_path: str,
    preprocessor: Callable[[PILImage.Image], PILImage.Image] | None,
) -> str | None:
    """Caminho estável de cache pro resultado do `.map()` de
    normalização+pré-processamento do Kvasir.

    Achado real (campanha de 31/08/2026): `load_condition_dataset` monta o
    dataset via `Dataset.from_dict(...)` (listas Python em memória), que
    não tem `cache_files` persistente -- sem um `cache_file_name` explícito,
    `datasets.map()` grava em um arquivo temporário novo a cada processo,
    então as 3 seeds de cada um dos 12 treinos da campanha recomputavam
    denoise+CLAHE do zero (~35-90min de CPU cada, mesma transformação
    idêntica -- é uma transformação da IMAGEM, não do modelo, e é
    exatamente a mesma entre LoRA/DoRA e entre modelos diferentes).

    Deriva um nome de arquivo determinístico a partir do arrow de origem
    (caminho + mtime, pra invalidar sozinho se o arrow for reconstruído)
    e da configuração exata do preprocessador (via `repr()` do dataclass
    `ImagePreprocessor`, que inclui todos os campos) -- qualquer mudança
    em um dos dois muda o hash e gera um cache novo, nunca serve resultado
    velho por engano.

    Retorna None se `preprocessor` for None (nada a cachear) ou se o
    arquivo de origem não existir ainda (deixa o `datasets.map()` cair no
    comportamento padrão em vez de gerar um caminho baseado em mtime
    inexistente)."""
    if preprocessor is None:
        return None
    if not os.path.exists(arrow_path):
        return None

    key = "|".join([
        os.path.abspath(arrow_path),
        str(os.path.getmtime(arrow_path)),
        repr(preprocessor),
    ])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(arrow_path))[0]
    os.makedirs(_KVASIR_PREPROCESS_CACHE_DIR, exist_ok=True)
    return f"{_KVASIR_PREPROCESS_CACHE_DIR}/{base}_{digest}.arrow"


def load_dataset_by_enum(
    dataset_enum: med_vlm.enums.DatasetName,
    split: str = "train",
    preprocessor: Callable[[PILImage.Image], PILImage.Image] | None = None,
    drop_image: bool = False,
    arrow_paths: dict | None = None,
):
    """Loads a dataset based on the provided enum value and applies a normalization
    function to it.

    drop_image (v7_txt, 2026-10-09): braco so-texto; `image` sai como None e o
    cache de pre-processamento (so de imagem) nao e usado.

    When ``dataset_enum`` is :attr:`~med_vlm.enums.DatasetName.COMBINED` all four
    individual medical VQA datasets are loaded (with appropriate split fallbacks) and
    concatenated into a single dataset.

    Args:
        dataset_enum: Which dataset (or COMBINED) to load.
        split: Canonical split name – ``"train"``, ``"validation"``, or ``"test"``.
        preprocessor: Optional image preprocessor applied after RGB conversion.
            Pass an :class:`~med_vlm.data.preprocessor.ImagePreprocessor` instance
            to enable CLAHE / denoising.
    """
    if dataset_enum == med_vlm.enums.DatasetName.COMBINED:
        parts = []
        for name in _INDIVIDUAL_DATASETS:
            actual_split = _SPLIT_MAP[name].get(split, split)
            try:
                ds = load_dataset(name.value, split=actual_split)
                parts.append(_normalize(ds, preprocessor=preprocessor))
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load dataset {name.value!r}"
                    f" (split={actual_split!r}): {e}"
                )
        if not parts:
            raise ValueError(
                f"No datasets could be loaded for COMBINED split='{split}'"
            )
        return concatenate_datasets(parts)

    if drop_image:
        preprocessor = None

    # v7_txt (2026-10-15, VQA-RAD fonte unica): `data.arrows` do YAML aponta o split para um arrow local no formato M3AE
    # (mesma leitura de Kvasir/HEAL/MIMIC); o enum do dataset continua o mesmo, entao prompt e rescoring nao mudam.
    if arrow_paths and split in arrow_paths:
        arrow_path = arrow_paths[split]
        ds = load_condition_dataset(arrow_path)
        cache_file_name = _kvasir_preprocess_cache_path(arrow_path, preprocessor)
        return _normalize(
            ds,
            preprocessor=preprocessor,
            answers_key="answers",
            id_key="id",
            cache_file_name=cache_file_name,
            drop_image=drop_image,
        )   # so toca a imagem; sem imagem, sem cache de pre-processamento

    if dataset_enum == med_vlm.enums.DatasetName.KVASIR_VQA:
        arrow_path = _KVASIR_CLEAN_ARROW_PATH[split]
        ds = load_condition_dataset(arrow_path)
        cache_file_name = _kvasir_preprocess_cache_path(arrow_path, preprocessor)
        return _normalize(
            ds,
            preprocessor=preprocessor,
            answers_key="answers",
            id_key="id",
            cache_file_name=cache_file_name,
            drop_image=drop_image,
        )

    if dataset_enum == med_vlm.enums.DatasetName.HEAL:
        arrow_path = _HEAL_ARROW_PATH[split]
        ds = load_condition_dataset(arrow_path)
        cache_file_name = _kvasir_preprocess_cache_path(arrow_path, preprocessor)
        return _normalize(
            ds,
            preprocessor=preprocessor,
            answers_key="answers",
            id_key="id",
            cache_file_name=cache_file_name,
            drop_image=drop_image,
        )

    if dataset_enum == med_vlm.enums.DatasetName.MIMIC:
        arrow_path = _MIMIC_ARROW_PATH[split]
        ds = load_condition_dataset(arrow_path)
        cache_file_name = _kvasir_preprocess_cache_path(arrow_path, preprocessor)
        return _normalize(
            ds,
            preprocessor=preprocessor,
            answers_key="answers",
            id_key="id",
            cache_file_name=cache_file_name,
            drop_image=drop_image,
        )

    ds = load_dataset(dataset_enum.value, split=split)
    return _normalize(ds, preprocessor=preprocessor, drop_image=drop_image)
