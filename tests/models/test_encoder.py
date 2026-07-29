"""Tests for Bitstring Encoder (TDD §3.2)."""

import torch
import pytest

from src.models.encoder import BitstringEncoder, PositionalEncoding


@pytest.fixture
def encoder():
    return BitstringEncoder(d_model=64, n_max_qubits=32)


@pytest.fixture
def sample_bitstrings():
    # Batch of 2, 5 bitstrings, 4 qubits
    return torch.tensor([
        [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]],
        [[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 1]],
    ])


def test_output_shape(encoder, sample_bitstrings):
    out = encoder(sample_bitstrings)
    B, M, n_qubits = sample_bitstrings.shape
    assert out.shape == (B, M, encoder.d_model)


def test_invariant_under_shot_permutation(encoder):
    """Output should be invariant under shot-axis shuffle."""
    bs = torch.tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])  # (2, 2, 2)
    out1 = encoder(bs)

    # Shuffle along shot axis (dim=1)
    perm = torch.randperm(bs.shape[1])
    bs_shuffled = bs[:, perm, :]
    out2 = encoder(bs_shuffled)

    # Each row in out2 should match the permuted rows in out1
    assert torch.allclose(out1[:, perm, :], out2, atol=1e-5)


def test_invariant_under_qubit_permutation(encoder):
    """Output should be invariant under qubit-axis shuffle (sum-pooling is symmetric)."""
    bs = torch.tensor([[[0, 1, 1], [1, 0, 0]]])  # (1, 2, 3)

    out1 = encoder(bs)

    # Shuffle along qubit axis (dim=2)
    perm = torch.tensor([2, 0, 1])
    bs_shuffled = bs[:, :, perm]
    out2 = encoder(bs_shuffled)

    # Sum-pooling makes this invariant
    assert torch.allclose(out1, out2, atol=1e-5)


def test_positional_encoding_shape():
    pe = PositionalEncoding(d_model=64, max_qubits=32)
    x = torch.randn(4, 6, 8, 64)  # (B, M, n_qubits, d_model)
    out = pe(x, dim=2)
    assert out.shape == x.shape


def test_qubit_embed_values():
    encoder = BitstringEncoder(d_model=64)
    bs = torch.tensor([[[0], [1]]])  # (1, 2, 1)
    out = encoder(bs)
    # Both bitstrings should produce valid embeddings (no NaN)
    assert not torch.isnan(out).any()
    # Different bitstrings should have different embeddings
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-5)
