"""Denoising Autoencoder (DAE) baseline (TDD §3.5).

Simple MLP autoencoder operating on full distribution vectors.
Used for comparison against Set Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionAutoencoder(nn.Module):
    """Denoising Autoencoder baseline."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        bottleneck_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Encoder
        layers = []
        d = input_dim
        for h in hidden_dims + [bottleneck_dim]:
            layers.extend([nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)])
            d = h
        self.encoder = nn.Sequential(*layers)
        
        # SN-D Decoder
        self.sn_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], input_dim),
        )
        
        # HN-E Decoder
        self.hn_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], input_dim),
        )
    
    def forward(self, x: torch.Tensor):
        """Forward pass.
        
        Args:
            x: (B, D) distribution vector
        
        Returns:
            sn_out: (B, D) SN-D output
            hn_out: (B, D) HN-E output
        """
        z = self.encoder(x)
        sn_logits = self.sn_decoder(z)
        hn_logits = self.hn_decoder(z)
        return F.softmax(sn_logits, dim=-1), F.softmax(hn_logits, dim=-1)
