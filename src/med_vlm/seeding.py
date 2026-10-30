"""Seed única de treino -- garante que 3 runs com seeds diferentes (0, 1,
2) do MESMO config realmente divergem (init de peso, ordem de
embaralhamento do DataLoader, dropout), e que o mesmo seed é
reprodutível. Antes deste módulo, run_experiment.py não chamava nenhuma
seed global -- só havia seeds fixos (42/123) para subsample de eval, sem
relação com o treino em si. Ver docs/superpowers/specs/
2026-08-30-kvasir-clean-split-design.md §13.1.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Semeia random/numpy/torch (CPU e todas as GPUs visíveis, se houver).
    `torch.utils.data.DataLoader(shuffle=True)` sem `generator=` explícito
    usa o gerador global do torch -- chamar isto antes de construir o
    DataLoader de treino já garante ordem de embaralhamento determinística
    por seed, sem precisar passar `generator=` em cada lugar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
