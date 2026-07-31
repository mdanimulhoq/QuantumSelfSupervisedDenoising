"""Count-Weighted Set Transformer with mask (view + contiguous)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Q, _ = query.shape
        _, K, _ = key.shape

        # 🔥 view + contiguous
        q = self.W_q(query).contiguous().view(B, Q, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_k(key).contiguous().view(B, K, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_v(value).contiguous().view(B, K, self.n_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, Q, -1)
        return self.W_o(out)

class SAB(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn = self.mha(x, x, x, mask=mask)
        x = self.norm1(x + attn)
        ff = self.ff(x)
        x = self.norm2(x + ff)
        return x

class ISAB(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, m, dropout=0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, m, d_model))
        self.mha1 = MultiHeadAttention(d_model, n_heads, dropout)
        self.mha2 = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, key_mask=None):
        B, M, _ = x.shape
        I = self.inducing.expand(B, -1, -1)
        attn_mask = None
        if key_mask is not None:
            attn_mask = (~key_mask).float()[:, None, None, :]  # (B,1,1,M)
        H = self.mha1(I, x, x, mask=attn_mask)
        H = self.norm1(I + H)
        H = H + self.ff(H)
        H = self.norm2(H)
        out = self.mha2(x, H, H)
        out = self.norm3(x + out)
        return out

class PMA(nn.Module):
    def __init__(self, d_model, n_heads, n_seeds, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model))
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_mask=None):
        B, _, _ = x.shape
        S = self.seeds.expand(B, -1, -1)
        attn_mask = None
        if key_mask is not None:
            attn_mask = (~key_mask).float()[:, None, None, :]
        out = self.mha(S, x, x, mask=attn_mask)
        out = self.norm(out)
        return out

class CountWeightedSetTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_ISAB, n_SAB, d_ff, m, dropout=0.1):
        super().__init__()
        self.count_proj = nn.Linear(1, d_model)
        self.encoder = nn.ModuleList([
            ISAB(d_model, n_heads, d_ff, m, dropout) for _ in range(n_ISAB)
        ])
        self.pma = PMA(d_model, n_heads, 1, dropout)
        self.decoder = nn.ModuleList([
            SAB(d_model, n_heads, d_ff, dropout) for _ in range(n_SAB)
        ])

    def forward(self, embeddings, counts, mask=None):
        x = embeddings + self.count_proj(counts)
        for isab in self.encoder:
            x = isab(x, key_mask=mask)
        z = self.pma(x, key_mask=mask)
        for sab in self.decoder:
            z = sab(z)
        return z.squeeze(1)
