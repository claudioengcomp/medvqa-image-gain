import random

import numpy as np
import torch

from med_vlm.seeding import set_seed


def test_set_seed_makes_python_random_deterministic():
    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b


def test_set_seed_makes_numpy_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert (a == b).all()


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_produce_different_sequences():
    set_seed(0)
    a = torch.rand(5)
    set_seed(1)
    b = torch.rand(5)
    assert not torch.equal(a, b)
