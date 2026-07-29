"""Set Transformer with count-weighted attention (TDD §3.3).

Implements ISAB (Induced Set Attention Block) and PMA (Pooling by Multihead Attention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHA(nn.Module):
    """Multi-head attention with separate projections for Q, K, V."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.shape
        B, Lk, _ = key.shape

        q = self.q_proj(query)  # (B, Lq, d_model)
        k = self.k_proj(key)    # (B, Lk, d_model)
        v = self.v_proj(value)  # (B, Lk, d_model)

        q = q.view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out(out)


class SAB(nn.Module):
    """Self-Attention Block."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.mha(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout2(self.ff(x))
        x = self.norm2(x)
        return x


class ISAB(nn.Module):
    """Induced Set Attention Block."""
    def __init__(self, d_model, n_heads, d_ff, m=16, dropout=0.1):
        super().__init__()
        self.m = m
        self.inducing_points = nn.Parameter(torch.randn(1, m, d_model))
        self.mha1 = MHA(d_model, n_heads, dropout)
        self.mha2 = MHA(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        B, M, _ = x.shape
        I = self.inducing_points.expand(B, -1, -1)  # (B, m, d)

        # Inducing points attend to input
        H = self.mha1(I, x, x)  # (B, m, d)
        H = I + self.dropout1(H)
        H = self.norm1(H)
        H = H + self.dropout2(self.ff(H))
        H = self.norm2(H)

        # Input attends to inducing points
        out = self.mha2(x, H, H)  # (B, M, d)
        out = x + self.dropout3(out)
        out = self.norm3(out)
        return out


class PMA(nn.Module):
    """Pooling by Multihead Attention."""
    def __init__(self, d_model, n_heads, num_seeds=1, dropout=0.1):
        super().__init__()
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.mha = MHA(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, _, _ = x.shape
        S = self.seeds.expand(B, -1, -1)
        out = self.mha(S, x, x)
        out = self.norm(out)
        return out


class CountWeightedSetTransformer(nn.Module):
    """Set Transformer with count-weighted attention (TDD §3.3)."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_ISAB=2,
        n_SAB=1,
        d_ff=256,
        m=16,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = nn.ModuleList([
            ISAB(d_model, n_heads, d_ff, m=m, dropout=dropout)
            for _ in range(n_ISAB)
        ])
        self.pma = PMA(d_model, n_heads, num_seeds=1, dropout=dropout)
        self.decoder = nn.ModuleList([
            SAB(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_SAB)
        ])
        self.count_proj = nn.Linear(1, d_model)

    def forward(self, embeddings, counts):
        x = embeddings + self.count_proj(counts)
        for isab in self.encoder:
            x = isab(x)
        z = self.pma(x)
        for sab in self.decoder:
            z = sab(z)
        return z.squeeze(1)