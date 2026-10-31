"""Single training seed -- ensures that 3 runs with different seeds (0, 1,
2) of the SAME config actually diverge (weight init, DataLoader
shuffling order, dropout), and that the same seed is
reproducible. Before this module, run_experiment.py did not call any
global seed -- there were only fixed seeds (42/123) for eval subsampling, with no
relation to the training itself.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds random/numpy/torch (CPU and all visible GPUs, if any).
    `torch.utils.data.DataLoader(shuffle=True)` without an explicit `generator=`
    uses torch's global generator -- calling this before building the
    training DataLoader already guarantees deterministic shuffling order
    by seed, without needing to pass `generator=` everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
