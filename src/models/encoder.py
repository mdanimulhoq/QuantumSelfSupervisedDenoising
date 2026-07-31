"""Bitstring Encoder with mask support."""
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, n_max_qubits=32):
        super().__init__()
        pe = torch.zeros(n_max_qubits, d_model)
        position = torch.arange(0, n_max_qubits, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        B, M, n, d = x.shape
        pe = self.pe[:n, :].unsqueeze(0).unsqueeze(0)
        return x + pe

class BitstringEncoder(nn.Module):
    def __init__(self, d_model=64, n_max_qubits=20):
        super().__init__()
        self.d_model = d_model
        self.qubit_embed = nn.Embedding(2, d_model)
        self.pos_encoding = PositionalEncoding(d_model, n_max_qubits)

    def forward(self, bitstrings):
        B, M, n = bitstrings.shape
        bitstrings_clean = bitstrings.clone()
        bitstrings_clean[bitstrings_clean == -1] = 0
        qubit_emb = self.qubit_embed(bitstrings_clean.long())
        qubit_emb = self.pos_encoding(qubit_emb)
        pooled = qubit_emb.mean(dim=2)
        mask = (bitstrings == -1).all(dim=-1)
        embeddings = pooled * (~mask).unsqueeze(-1).float()
        return embeddings, mask
