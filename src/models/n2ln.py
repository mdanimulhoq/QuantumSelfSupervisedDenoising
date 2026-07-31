"""N2LN-QEM: Main model wrapper (TDD §2.6 + §3.1).

Assembles: BitstringEncoder + CountWeightedSetTransformer + DualHeadDecoder.
Supports three modes: sn_only, hn_only, unified.
"""

import torch
import torch.nn as nn
from src.models.encoder import BitstringEncoder
from src.models.set_transformer import CountWeightedSetTransformer
from src.models.decoder import DualHeadDecoder


class N2LNQEM(nn.Module):
    """N2LN-QEM model with dual-head output.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_isab: Number of Induced Set Attention Blocks
        n_sab: Number of Self-Attention Blocks
        d_ff: Feed-forward dimension
        m: Number of inducing points
        decoder_hidden: Hidden dimension in decoder
        dropout: Dropout rate
        max_qubits: Maximum qubits (for positional encoding)
        use_mlp_scorer: Use MLP scorer instead of dot-product (TDD §3.4)
        temperature_floor: Minimum temperature to prevent logit explosion
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_isab: int = 2,
        n_sab: int = 1,
        d_ff: int = 256,
        m: int = 16,
        decoder_hidden: int = 128,
        dropout: float = 0.1,
        max_qubits: int = 20,
        use_mlp_scorer: bool = True,
        temperature_floor: float = 0.3,
    ):
        super().__init__()

        # Bitstring encoder
        self.encoder = BitstringEncoder(
            d_model=d_model,
            n_max_qubits=max_qubits,
        )

        # Set Transformer
        self.transformer = CountWeightedSetTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_ISAB=n_isab,
            n_SAB=n_sab,
            d_ff=d_ff,
            m=m,
            dropout=dropout,
        )

        # Dual-head decoder
        self.decoder = DualHeadDecoder(
            d_model=d_model,
            n_max_qubits=max_qubits,
            hidden_size=decoder_hidden,
            temperature=1.0,
            dropout=dropout,
            use_mlp_scorer=use_mlp_scorer,
            temperature_floor=temperature_floor,
        )

        # Share encoder with decoder
        self.decoder.set_bitstring_encoder(self.encoder)

        self.mode = 'unified'  # 'sn_only' | 'hn_only' | 'unified'

    def forward(self, bitstrings, counts, mode=None):
        """Forward pass.

        Args:
            bitstrings: (B, M, n) bitstrings
            counts: (B, M, 1) normalized counts
            mode: 'sn_only' | 'hn_only' | 'unified' (default: self.mode)

        Returns:
            sn_dist: (B, K) SN-D output distribution
            hn_dist: (B, K) HN-E output distribution
        """
        if mode is None:
            mode = self.mode

        # Encode bitstrings
        embeddings = self.encoder(bitstrings)  # (B, M, d)

        # Global representation
        z = self.transformer(embeddings, counts)  # (B, d)

        # Decode
        sn_dist, hn_dist = self.decoder(z, bitstrings)  # (B, K), (B, K)

        # Apply mode
        if mode == 'sn_only':
            return sn_dist, None
        elif mode == 'hn_only':
            return None, hn_dist
        else:  # unified
            return sn_dist, hn_dist

    def set_mode(self, mode: str):
        """Set model mode."""
        assert mode in ['sn_only', 'hn_only', 'unified']
        self.mode = mode

    def get_mode(self) -> str:
        return self.mode
