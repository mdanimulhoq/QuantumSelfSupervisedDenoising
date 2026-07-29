"""Test seeding reproducibility."""
import torch
import numpy as np
import random
from src.utils.seeding import set_seed

def test_torch_seed_reproducibility():
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)

def test_numpy_seed_reproducibility():
    set_seed(42)
    a = np.random.randn(10)
    set_seed(42)
    b = np.random.randn(10)
    assert np.allclose(a, b)
