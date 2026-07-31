"""Dual-head decoder with dot-product scorer (temporarily)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualHeadDecoder(nn.Module):
    def __init__(self, d_model=64, n_max_qubits=20, hidden_size=128,
                 temperature=0.1, dropout=0.1, use_mlp_scorer=False,
                 temperature_floor=0.1):
        super().__init__()
        self.d_model = d_model
        self.temperature_floor = temperature_floor
        self.register_buffer('temperature', torch.tensor(temperature))
        self.sn_head = nn.Sequential(
            nn.Linear(d_model, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, d_model)
        )
        self.hn_head = nn.Sequential(
            nn.Linear(d_model, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, d_model)
        )
        # 🔥 Use dot-product scorer (no MLP)
        self.sn_scorer = None
        self.hn_scorer = None
        self.bitstring_encoder = None

    def set_bitstring_encoder(self, encoder):
        self.bitstring_encoder = encoder

    def _safe_temperature(self):
        return self.temperature.clamp(min=self.temperature_floor, max=2.0)

    def _score(self, z_per_m, bs_emb, scorer):
        # Dot-product fallback
        return (z_per_m * bs_emb).sum(dim=-1)

    def forward(self, z, bitstrings, mask=None):
        if self.bitstring_encoder is None:
            raise RuntimeError("Encoder not set")
        bs_emb, enc_mask = self.bitstring_encoder(bitstrings)
        if mask is None:
            mask = enc_mask
        B, M, _ = bitstrings.shape
        sn_z = self.sn_head(z).unsqueeze(1).expand(B, M, -1)
        sn_logits = self._score(sn_z, bs_emb, self.sn_scorer)
        sn_logits = sn_logits.masked_fill(mask, -1e9)
        T = self._safe_temperature()
        sn_dist = F.softmax(sn_logits / T, dim=-1)
        hn_z = self.hn_head(z).unsqueeze(1).expand(B, M, -1)
        hn_logits = self._score(hn_z, bs_emb, self.hn_scorer)
        hn_logits = hn_logits.masked_fill(mask, -1e9)
        hn_dist = F.softmax(hn_logits / T, dim=-1)
        return sn_dist, hn_dist
