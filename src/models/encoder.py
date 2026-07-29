"""Bitstring Encoder for N2LN-QEM (TDD §3.2).

Encodes bitstrings using per-qubit embedding + positional encoding.
Permutation-invariant by construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for qubit positions."""
    
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, n, d/2)
        B, M, n, d = x.shape
        # Add positional encoding to qubit dimension
        pe = self.pe[:, :n, :d].unsqueeze(1)  # (1, 1, n, d)
        return x + pe


class BitstringEncoder(nn.Module):
    """Encodes bitstrings using per-qubit embedding + positional encoding."""
    
    def __init__(self, d_model: int = 64, n_max_qubits: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n_max_qubits = n_max_qubits
        # Embedding for bit values 0 and 1 (and -1 for padding, but we handle it)
        self.qubit_embed = nn.Embedding(2, d_model // 2)
        self.pos_encoding = PositionalEncoding(d_model // 2, n_max_qubits)
        self.projection = nn.Linear(d_model // 2, d_model)
    
    def forward(self, bitstrings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bitstrings: (B, M, n) with values 0, 1, or -1 (padding)
        Returns:
            embeddings: (B, M, d)
        """
        B, M, n = bitstrings.shape
        
        # Handle padding: replace -1 with 0 (safe embedding)
        bitstrings_clean = bitstrings.clone()
        bitstrings_clean[bitstrings_clean == -1] = 0
        
        # Per-qubit embedding: (B, M, n, d//2)
        qubit_emb = self.qubit_embed(bitstrings_clean.long())
        
        # Add positional encoding
        qubit_emb = self.pos_encoding(qubit_emb)
        
        # Pool over qubits (sum = permutation-invariant)
        pooled = qubit_emb.sum(dim=2)  # (B, M, d//2)
        
        # Project to d_model
        embeddings = self.projection(pooled)  # (B, M, d)
        
        # Create mask for padding (where all bits are -1)
        mask = (bitstrings == -1).all(dim=-1)  # (B, M)
        # Set masked positions to zero
        embeddings = embeddings * (~mask).unsqueeze(-1).float()
        
        return embeddings