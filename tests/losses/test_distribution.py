"""Analytical tests for distribution losses (TDD §4.1)."""

import torch
import pytest

from src.losses.distribution import (
    kl_loss,
    tvd_loss,
    chi2_loss,
    composite_dist_loss,
    CompositeDistributionLoss,
)


def test_kl_identical_distributions():
    """KL(identical || identical) should be 0."""
    p = torch.tensor([[0.2, 0.3, 0.5]])
    loss = kl_loss(p, p)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_kl_numpy_reference():
    """Compare against analytical KL value: KL(target || pred)."""
    target = torch.tensor([[0.5, 0.5]])
    pred = torch.tensor([[0.9, 0.1]])
    loss = kl_loss(pred, target)
    # KL(0.5,0.5 || 0.9,0.1) = 0.5*log(0.5/0.9) + 0.5*log(0.5/0.1)
    expected = 0.5 * torch.log(torch.tensor(0.5) / 0.9) + \
               0.5 * torch.log(torch.tensor(0.5) / 0.1)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_kl_positive():
    """KL should be non-negative."""
    p = torch.tensor([[0.7, 0.3]])
    q = torch.tensor([[0.4, 0.6]])
    loss = kl_loss(p, q)
    assert loss >= 0


def test_tvd_identical():
    """TVD(identical, identical) should be 0."""
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    loss = tvd_loss(p, p)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_tvd_maximum():
    """TVD between disjoint distributions should be 1."""
    p = torch.tensor([[1.0, 0.0]])
    q = torch.tensor([[0.0, 1.0]])
    loss = tvd_loss(p, q)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)


def test_tvd_symmetric():
    """TVD is symmetric."""
    p = torch.tensor([[0.7, 0.3]])
    q = torch.tensor([[0.2, 0.8]])
    assert torch.allclose(tvd_loss(p, q), tvd_loss(q, p), atol=1e-5)


def test_chi2_identical():
    """Chi-squared(identical, identical) should be 0."""
    p = torch.tensor([[0.5, 0.5]])
    loss = chi2_loss(p, p)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_composite_identical():
    """Composite loss on identical distributions should be 0."""
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    loss = composite_dist_loss(p, p)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_composite_with_zeros():
    """Composite loss with all weights zero should be 0."""
    p = torch.tensor([[0.9, 0.1]])
    q = torch.tensor([[0.5, 0.5]])
    loss = composite_dist_loss(p, q, alpha=0.0, beta=0.0, gamma=0.0)
    assert loss == 0.0


def test_batch_mean():
    """Loss should be mean over batch."""
    p = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    q = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    loss1 = tvd_loss(p[0:1], q[0:1])
    loss2 = tvd_loss(p[1:2], q[1:2])
    loss_batch = tvd_loss(p, q)
    expected = (loss1 + loss2) / 2
    assert torch.allclose(loss_batch, expected, atol=1e-5)


def test_composite_loss_module():
    """Test the nn.Module wrapper."""
    criterion = CompositeDistributionLoss(alpha=1.0, beta=0.5, gamma=0.1)
    p = torch.tensor([[0.7, 0.3]])
    q = torch.tensor([[0.5, 0.5]])
    loss = criterion(p, q)
    assert loss > 0
    assert loss.ndim == 0  # scalar
