"""Dual-head decoder for N2LN-QEM (TDD §3.4).

Produces both SN-D (shot-noise denoised) and HN-E (hardware-noise mitigated) distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadDecoder(nn.Module):
    """Dual-head decoder with separate heads for SN-D and HN-E."""

    def __init__(
        self,
        d_model: int = 64,
        n_max_qubits: int = 20,
        hidden_size: int = 128,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_max_qubits = n_max_qubits
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.dropout = nn.Dropout(dropout)

        # SN-D head
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

        # For positional encoding of bitstrings (if needed)
        self.bitstring_encoder = None  # Will be set externally
        
        # Dictionary to store projections for different qubit counts
        self._fallback_projs = nn.ModuleDict()

    def set_bitstring_encoder(self, encoder):
        """Set the bitstring encoder (from main model)."""
        self.bitstring_encoder = encoder

    def _get_fallback_proj(self, n_qubits: int, device: torch.device):
        """Get or create a fallback projection for given qubit count."""
        key = f"proj_{n_qubits}"
        if key not in self._fallback_projs:
            proj = nn.Linear(n_qubits, self.d_model).to(device)
            self._fallback_projs[key] = proj
        return self._fallback_projs[key]

    def forward(self, z: torch.Tensor, bitstrings: torch.Tensor):
        """Forward pass.

        Args:
            z: (B, d) global latent vector
            bitstrings: (B, M, n) candidate bitstrings

        Returns:
            sn_dist: (B, M) SN-D distribution
            hn_dist: (B, M) HN-E distribution
        """
        B, M, n = bitstrings.shape
        device = z.device

        # Encode bitstrings if encoder is available, else use simple projection
        if self.bitstring_encoder is not None:
            # Encode each bitstring to d-dim features
            bs_embeddings = self.bitstring_encoder(bitstrings)  # (B, M, d)
        else:
            # Fallback: linear projection for each qubit count
            flat_bs = bitstrings.float().view(B * M, n)  # (B*M, n)
            proj = self._get_fallback_proj(n, device)
            bs_embeddings = proj(flat_bs).view(B, M, self.d_model)  # (B, M, d)

        # SN-D branch
        sn_z = self.sn_head(z)  # (B, d)
        sn_z_exp = sn_z.unsqueeze(1)  # (B, 1, d)
        sn_logits = torch.sum(sn_z_exp * bs_embeddings, dim=-1)  # (B, M)
        sn_logits = sn_logits / self.temperature
        sn_dist = F.softmax(sn_logits, dim=-1)

        # HN-E branch
        hn_z = self.hn_head(z)  # (B, d)
        hn_z_exp = hn_z.unsqueeze(1)  # (B, 1, d)
        hn_logits = torch.sum(hn_z_exp * bs_embeddings, dim=-1)  # (B, M)
        hn_logits = hn_logits / self.temperature
        hn_dist = F.softmax(hn_logits, dim=-1)

        return sn_dist, hn_dist