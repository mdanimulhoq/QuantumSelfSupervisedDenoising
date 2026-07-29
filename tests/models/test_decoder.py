"""Tests for Dual-Head Decoder (TDD §3.4)."""

import torch
import pytest

from src.models.decoder import DistributionHead, DualHeadDecoder, generate_all_bitstrings


@pytest.fixture
def decoder():
    return DualHeadDecoder(d_model=64, hidden_dim=128, n_max_qubits=8)


@pytest.fixture
def latent_vector():
    return torch.randn(2, 64)


@pytest.fixture
def candidate_bitstrings():
    return generate_all_bitstrings(2).unsqueeze(0).expand(2, -1, -1)  # (2, 4, 2)


def test_distribution_head_output_sums_to_one():
    head = DistributionHead(d_model=64, hidden_dim=128)
    z = torch.randn(1, 64)
    cand_emb = torch.randn(1, 8, 64)
    probs = head(z, cand_emb)
    assert probs.shape == (1, 8)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0), atol=1e-5)
    assert (probs >= 0).all()


def test_dual_head_shapes(decoder, latent_vector, candidate_bitstrings):
    sn, hn = decoder(latent_vector, candidate_bitstrings)
    B, M = candidate_bitstrings.shape[:2]
    assert sn.shape == (B, M)
    assert hn.shape == (B, M)


def test_dual_head_outputs_sum_to_one(decoder, latent_vector, candidate_bitstrings):
    sn, hn = decoder(latent_vector, candidate_bitstrings)
    assert torch.allclose(sn.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(hn.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_dual_head_outputs_non_negative(decoder, latent_vector, candidate_bitstrings):
    sn, hn = decoder(latent_vector, candidate_bitstrings)
    assert (sn >= 0).all()
    assert (hn >= 0).all()


def test_heads_produce_different_outputs(decoder, latent_vector):
    """SN-D and HN-E should produce different distributions (different weights)."""
    cand = generate_all_bitstrings(3).unsqueeze(0).expand(2, -1, -1)
    sn, hn = decoder(latent_vector, cand)
    assert not torch.allclose(sn, hn, atol=1e-5)


def test_generate_all_bitstrings():
    bs = generate_all_bitstrings(2)
    assert bs.shape == (4, 2)
    assert torch.equal(bs[0], torch.tensor([0, 0]))
    assert torch.equal(bs[3], torch.tensor([1, 1]))


def test_batch_independence(decoder):
    """Different latent vectors should produce different distributions."""
    z1 = torch.randn(1, 64)
    z2 = torch.randn(1, 64)
    cand = generate_all_bitstrings(3).unsqueeze(0)  # (1, 8, 3)

    sn1, _ = decoder(z1, cand)
    sn2, _ = decoder(z2, cand)

    assert not torch.allclose(sn1, sn2, atol=1e-3)
