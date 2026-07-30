"""Dual-head decoder for N2LN-QEM (TDD §3.4) — REVISED with MLP Scorer.

Changes (2026-07-30):
  1. MLP scorer (concat + non-linear) instead of pure dot-product.
  2. temperature_floor=0.3 prevents logit explosion.
  3. use_mlp_scorer flag to toggle between MLP and dot-product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadDecoder(nn.Module):
    """Dual-head decoder with separate heads for SN-D and HN-E.

    Now supports MLP-based scoring for sharper probability peaks.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_max_qubits: int = 20,
        hidden_size: int = 128,
        temperature: float = 1.0,
        dropout: float = 0.1,
        use_mlp_scorer: bool = True,
        temperature_floor: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_max_qubits = n_max_qubits
        self.use_mlp_scorer = use_mlp_scorer
        self.temperature_floor = temperature_floor

        # Learnable temperature with floor
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # SN-D head (projects global z to head-specific space)
        self.sn_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
        )

        # HN-E head
        self.hn_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
        )

        # 🔥 NEW: MLP scorer (non-linear compatibility between z and bitstring)
        if use_mlp_scorer:
            self.sn_scorer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self.hn_scorer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        else:
            self.sn_scorer = None
            self.hn_scorer = None

        # Will be set externally via set_bitstring_encoder()
        self.bitstring_encoder = None

    def set_bitstring_encoder(self, encoder):
        """Set the bitstring encoder (from main model)."""
        self.bitstring_encoder = encoder

    def _safe_temperature(self):
        """Clamp temperature to prevent over-sharp (logit explosion) and over-smooth."""
        return self.temperature.clamp(min=self.temperature_floor, max=5.0)

    def _score(self, z_per_m: torch.Tensor, bs_emb: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
        """Compute logits with MLP-scorer (concat) or dot-product (fallback).

        Args:
            z_per_m: (B, M, d) — same z broadcast to all M bitstrings
            bs_emb: (B, M, d) — bitstring embeddings
            scorer: nn.Sequential or None

        Returns:
            logits: (B, M)
        """
        if scorer is not None:
            # MLP over [z, bs_emb] concatenation
            h = torch.cat([z_per_m, bs_emb], dim=-1)  # (B, M, 2d)
            logits = scorer(h).squeeze(-1)             # (B, M)
        else:
            # Dot-product fallback (original behaviour)
            logits = (z_per_m * bs_emb).sum(dim=-1)    # (B, M)
        return logits

    def forward(self, z: torch.Tensor, bitstrings: torch.Tensor):
        """Forward pass.

        Args:
            z: (B, d) global latent vector from Set Transformer
            bitstrings: (B, M, n) candidate bitstrings

        Returns:
            sn_dist: (B, M) SN-D distribution (shot-noise denoised)
            hn_dist: (B, M) HN-E distribution (hardware-noise mitigated)
        """
        if self.bitstring_encoder is None:
            raise RuntimeError(
                "Decoder.bitstring_encoder is not set. "
                "Please call set_bitstring_encoder() after decoder creation."
            )

        B, M, n = bitstrings.shape

        # Encode bitstrings using the shared encoder
        bs_emb = self.bitstring_encoder(bitstrings)  # (B, M, d)

        # --- SN-D Branch ---
        sn_z = self.sn_head(z)  # (B, d)
        sn_z_per_m = sn_z.unsqueeze(1).expand(B, M, self.d_model)  # (B, M, d)
        sn_logits = self._score(sn_z_per_m, bs_emb, self.sn_scorer)
        T = self._safe_temperature()
        sn_dist = F.softmax(sn_logits / T, dim=-1)

        # --- HN-E Branch ---
        hn_z = self.hn_head(z)  # (B, d)
        hn_z_per_m = hn_z.unsqueeze(1).expand(B, M, self.d_model)  # (B, M, d)
        hn_logits = self._score(hn_z_per_m, bs_emb, self.hn_scorer)
        hn_dist = F.softmax(hn_logits / T, dim=-1)

        return sn_dist, hn_dist