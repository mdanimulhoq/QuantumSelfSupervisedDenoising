"""Simple MLP for sanity check (no transformer)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, n_qubits=4, d_model=32, hidden=64):
        super().__init__()
        # Input: bitstrings (B, M, n) + counts (B, M, 1)
        # Flatten: (B, M, n+1)
        self.fc1 = nn.Linear(4 + 1, hidden)  # n=4 qubits + count
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.temperature = 0.1

    def forward(self, bitstrings, counts):
        # bitstrings: (B, M, n), counts: (B, M, 1)
        x = torch.cat([bitstrings.float(), counts], dim=-1)  # (B, M, n+1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        logits = self.fc3(x).squeeze(-1)  # (B, M)
        return F.softmax(logits / self.temperature, dim=-1)
