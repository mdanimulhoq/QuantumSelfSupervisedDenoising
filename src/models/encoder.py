"""Bitstring Encoder with position-aware embeddings (fixes Hamming-weight collapse)."""
import torch
import torch.nn as nn

class BitstringEncoder(nn.Module):
    def __init__(self, d_model=64, n_max_qubits=20):
        super().__init__()
        self.d_model = d_model
        self.n_max_qubits = n_max_qubits
        # প্রতিটা (position, bit-value) জোড়ার জন্য আলাদা vector
        self.qubit_embed = nn.Embedding(2 * n_max_qubits, d_model)

    def forward(self, bitstrings):
        B, M, n = bitstrings.shape
        pad_mask = (bitstrings == -1)
        bits = bitstrings.clone()
        bits[pad_mask] = 0

        positions = torch.arange(n, device=bitstrings.device).view(1, 1, n)
        idx = positions * 2 + bits.long()
        qubit_emb = self.qubit_embed(idx)
        qubit_emb = qubit_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        pooled = qubit_emb.sum(dim=2)
        mask = pad_mask.all(dim=-1)
        embeddings = pooled * (~mask).unsqueeze(-1).float()
        return embeddings, mask
