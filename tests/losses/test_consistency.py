"""Tests for consistency loss (TDD §4.1)."""

import torch
import pytest

from src.losses.consistency import consistency_loss, ConsistencyLoss


def test_consistency_identical():
    """Consistency loss should be zero for identical inputs."""
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    loss = consistency_loss(x, x)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_consistency_different():
    """Consistency loss should be positive for different inputs."""
    x = torch.tensor([[0.5, 0.5]])
    y = torch.tensor([[0.7, 0.3]])
    loss = consistency_loss(x, y)
    assert loss > 0


def test_consistency_module():
    criterion = ConsistencyLoss()
    x = torch.tensor([[0.2, 0.8]])
    y = torch.tensor([[0.3, 0.7]])
    loss = criterion(x, y)
    assert loss > 0
    assert loss.ndim == 0
