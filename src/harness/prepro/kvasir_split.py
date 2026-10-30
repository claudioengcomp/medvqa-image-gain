"""Split determinístico e estratificado por imagem do Kvasir-VQA-x1 --
lógica pura, sem rede/GPU (separada do script CLI em
build_kvasir_clean_split.py, que faz o download de verdade).

Motivação: o split oficial do HuggingFace (train/test) tem 94% de overlap
de imagem entre os dois lados -- ver docs/superpowers/specs/
2026-08-30-kvasir-clean-split-design.md §1.B. Este módulo agrupa TODAS as
imagens únicas (de train+test juntos) e sorteia um split novo, disjunto
por img_id, estratificado por pathology_label pra não deixar uma
categoria clínica rara inteira de um lado só.
"""
from __future__ import annotations

import random
from collections import defaultdict


def split_records_by_image(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Separa `records` (saída de
    harness.prepro.prepare_kvasir_arrow.build_image_records -- um dict por
    imagem única, com `pathology_label`) em train/val/test, disjunto por
    `img_id`, estratificado por `pathology_label` (imagens sem
    pathology_label caem no estrato `None`, tratado como qualquer outro).

    `ratios` = (train, val, test), deve somar 1.0 (tolerância 1e-9).
    Determinístico: mesma `seed` produz sempre a mesma atribuição.
    """
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios devem somar 1.0, somaram {total}")

    rng = random.Random(seed)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_label[rec["pathology_label"]].append(rec)

    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        result["train"].extend(shuffled[:n_train])
        result["val"].extend(shuffled[n_train:n_train + n_val])
        result["test"].extend(shuffled[n_train + n_val:])

    return result
