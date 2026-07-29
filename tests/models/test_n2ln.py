"""Tests for N2LN-QEM Model Wrapper (TDD §2.6)."""

import torch
import pytest

from src.models.n2ln import N2LNQEM
from src.models.decoder import generate_all_bitstrings


@pytest.fixture
def model():
    return N2LNQEM(
        d_model=64,
        n_heads=4,
        n_isab=2,
        n_sab=1,
        d_ff=128,
        m=8,
        decoder_hidden=128,
        n_max_qubits=8,
        dropout=0.0,
    )


@pytest.fixture
def sample_input():
    # Batch of 2, 5 observed bitstrings, 4 qubits
    bs = torch.tensor([
        [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]],
        [[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 1]],
    ])
    counts = torch.tensor([
        [[0.40], [0.25], [0.15], [0.12], [0.08]],
        [[0.35], [0.30], [0.20], [0.10], [0.05]],
    ])
    return bs, counts


def test_model_instantiation():
    model = N2LNQEM()
    assert model.d_model == 64


def test_unified_mode(model, sample_input):
    bs, counts = sample_input
    sn, hn = model(bs, counts, mode="unified")
    B, M_in, n_qubits = bs.shape
    M_out = 2 ** n_qubits  # all bitstrings
    assert sn.shape == (B, M_out)
    assert hn.shape == (B, M_out)
    assert torch.allclose(sn.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert torch.allclose(hn.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_sn_d_only_mode(model, sample_input):
    bs, counts = sample_input
    sn, hn = model(bs, counts, mode="sn_d_only")
    assert sn is not None
    assert hn is None
    assert torch.allclose(sn.sum(dim=-1), torch.ones(bs.shape[0]), atol=1e-5)


def test_hn_e_only_mode(model, sample_input):
    bs, counts = sample_input
    sn, hn = model(bs, counts, mode="hn_e_only")
    assert sn is None
    assert hn is not None
    assert torch.allclose(hn.sum(dim=-1), torch.ones(bs.shape[0]), atol=1e-5)


def test_with_explicit_candidates(model, sample_input):
    bs, counts = sample_input
    candidates = generate_all_bitstrings(2).unsqueeze(0).expand(2, -1, -1)  # 2 qubit candidates
    # Adjust input to 2 qubits
    bs_2q = torch.tensor([[[0, 0], [1, 1]], [[0, 1], [1, 0]]])
    cnt_2q = torch.tensor([[[0.5], [0.5]], [[0.5], [0.5]]])
    sn, hn = model(bs_2q, cnt_2q, candidate_bitstrings=candidates, mode="unified")
    assert sn.shape == (2, 4)
    assert hn.shape == (2, 4)


def test_get_global_latent(model, sample_input):
    bs, counts = sample_input
    z = model.get_global_latent(bs, counts)
    assert z.shape == (2, model.d_model)


def test_outputs_non_negative(model, sample_input):
    bs, counts = sample_input
    sn, hn = model(bs, counts, mode="unified")
    assert (sn >= 0).all()
    assert (hn >= 0).all()
