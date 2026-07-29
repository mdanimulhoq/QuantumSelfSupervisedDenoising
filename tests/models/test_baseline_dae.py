"""Tests for Baseline Denoising Autoencoder (TDD §3.5)."""

import torch
import pytest

from src.models.baseline_dae import DenoisingAutoencoder, create_dae_for_qubits


@pytest.fixture
def dae_4q():
    return create_dae_for_qubits(4)


@pytest.fixture
def dae_2q():
    return create_dae_for_qubits(2)


@pytest.fixture
def sample_dist_4q():
    # Batch of 3, 16-dim probability vectors
    x = torch.rand(3, 16)
    return x / x.sum(dim=-1, keepdim=True)


@pytest.fixture
def sample_dist_2q():
    x = torch.rand(5, 4)
    return x / x.sum(dim=-1, keepdim=True)


def test_dae_instantiation_4q():
    dae = create_dae_for_qubits(4)
    assert dae.input_dim == 16


def test_dae_instantiation_8q():
    dae = create_dae_for_qubits(8)
    assert dae.input_dim == 256


def test_dae_forward_shape_4q(dae_4q, sample_dist_4q):
    sn, hn = dae_4q(sample_dist_4q)
    assert sn.shape == (3, 16)
    assert hn.shape == (3, 16)


def test_dae_forward_shape_2q(dae_2q, sample_dist_2q):
    sn, hn = dae_2q(sample_dist_2q)
    assert sn.shape == (5, 4)
    assert hn.shape == (5, 4)


def test_dae_outputs_sum_to_one_4q(dae_4q, sample_dist_4q):
    sn, hn = dae_4q(sample_dist_4q)
    assert torch.allclose(sn.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(hn.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_dae_outputs_non_negative_4q(dae_4q, sample_dist_4q):
    sn, hn = dae_4q(sample_dist_4q)
    assert (sn >= 0).all()
    assert (hn >= 0).all()


def test_dae_heads_different(dae_4q, sample_dist_4q):
    sn, hn = dae_4q(sample_dist_4q)
    assert not torch.allclose(sn, hn, atol=1e-5)


def test_dae_2q_instantiation():
    dae = DenoisingAutoencoder(n_qubits=2, hidden_dims=[32, 16], bottleneck_dim=8)
    x = torch.rand(2, 4)
    x = x / x.sum(dim=-1, keepdim=True)
    sn, hn = dae(x)
    assert sn.shape == (2, 4)
