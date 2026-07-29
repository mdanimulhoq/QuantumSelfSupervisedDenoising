"""Tests for Count-Weighted Set Transformer (TDD §3.3)."""

import torch
import pytest

from src.models.set_transformer import (
    MultiheadAttentionBlock,
    InducedSetAttentionBlock,
    SelfAttentionBlock,
    PoolingByMultiheadAttention,
    CountWeightedSetTransformer,
)


@pytest.fixture
def transformer():
    return CountWeightedSetTransformer(
        d_model=64, n_heads=4, n_isab=2, n_sab=1, d_ff=128, m=8, dropout=0.0
    )


def test_mab_shape():
    mab = MultiheadAttentionBlock(64, 4, 128)
    q = torch.randn(2, 10, 64)
    kv = torch.randn(2, 15, 64)
    out = mab(q, kv)
    assert out.shape == q.shape


def test_isab_shape():
    isab = InducedSetAttentionBlock(64, 4, 128, m=8)
    x = torch.randn(2, 10, 64)
    out = isab(x)
    assert out.shape == x.shape


def test_sab_shape():
    sab = SelfAttentionBlock(64, 4, 128)
    x = torch.randn(2, 10, 64)
    out = sab(x)
    assert out.shape == x.shape


def test_pma_shape():
    pma = PoolingByMultiheadAttention(64, 4, k=1, d_ff=128)
    x = torch.randn(2, 10, 64)
    out = pma(x)
    assert out.shape == (2, 1, 64)


def test_transformer_output_shape(transformer):
    embeddings = torch.randn(2, 8, 64)
    counts = torch.rand(2, 8, 1)
    out = transformer(embeddings, counts)
    assert out.shape == (2, 64)


def test_transformer_permutation_invariant(transformer):
    """Output should be invariant to input order."""
    embeddings = torch.randn(1, 5, 64)
    counts = torch.rand(1, 5, 1)

    out1 = transformer(embeddings, counts)

    # Permute along set axis
    perm = torch.randperm(5)
    embeddings_shuffled = embeddings[:, perm, :]
    counts_shuffled = counts[:, perm, :]
    out2 = transformer(embeddings_shuffled, counts_shuffled)

    assert torch.allclose(out1, out2, atol=1e-4)


def test_count_weight_sanity(transformer):
    """Zeroing a frequent bitstring should change output more than zeroing a rare one."""
    torch.manual_seed(42)
    embeddings = torch.randn(1, 5, 64)
    # Normalized counts: bitstring 0 is 50%, bitstring 4 is 5%
    counts = torch.tensor([[[0.50], [0.25], [0.15], [0.05], [0.05]]])

    ref_out = transformer(embeddings, counts)

    # Zero the frequent one (index 0)
    cnt_zero_freq = counts.clone()
    cnt_zero_freq[:, 0, :] = 0.0
    out_zero_freq = transformer(embeddings, cnt_zero_freq)

    # Zero the rare one (index 4)
    cnt_zero_rare = counts.clone()
    cnt_zero_rare[:, 4, :] = 0.0
    out_zero_rare = transformer(embeddings, cnt_zero_rare)

    diff_freq = torch.norm(ref_out - out_zero_freq)
    diff_rare = torch.norm(ref_out - out_zero_rare)

    # Zeroing frequent should cause larger change
    assert diff_freq > diff_rare, f"diff_freq={diff_freq:.4f}, diff_rare={diff_rare:.4f}"
