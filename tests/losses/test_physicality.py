"""Tests for physicality loss (TDD §4.1)."""

import torch
import pytest

from src.losses.physicality import physicality_loss, PhysicalityLoss


def test_physicality_zero_on_valid_distribution():
    """Physicality loss should be zero for valid distributions."""
    p = torch.tensor([[0.2, 0.3, 0.5]])
    loss = physicality_loss(p)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_physicality_penalizes_negative():
    """Should penalize negative values."""
    p = torch.tensor([[0.5, -0.1, 0.6]])
    loss = physicality_loss(p)
    assert loss > 0


def test_physicality_penalizes_non_normalized():
    """Should penalize sum != 1."""
    p = torch.tensor([[0.2, 0.3, 0.4]])  # sum = 0.9
    loss = physicality_loss(p, lambda_neg=0.0, lambda_norm=1.0)
    assert loss > 0


def test_physicality_module():
    criterion = PhysicalityLoss(lambda_neg=1.0, lambda_norm=1.0)
    p = torch.tensor([[0.1, 0.9]])
    loss = criterion(p)
    assert loss == 0.0
