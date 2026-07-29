"""Data encoding: counts → tensor (TDD §5.4)."""

import torch
import numpy as np
from typing import Dict, Tuple

def counts_to_tensor(
    counts_dict: Dict[str, int],
    n_qubits: int,
    max_bitstrings: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Qiskit Counts dict to model input format.
    
    Args:
        counts_dict: {'01': 45, '10': 55, ...}
        n_qubits: number of qubits
        max_bitstrings: cap on number of bitstrings
    
    Returns:
        bitstrings: (M, n) int tensor
        probs: (M,) float tensor (normalized)
    """
    total = sum(counts_dict.values())
    items = sorted(counts_dict.items(), key=lambda x: -x[1])[:max_bitstrings]
    
    bitstrings = []
    probs = []
    for bs, count in items:
        bitstrings.append([int(b) for b in bs.zfill(n_qubits)])
        probs.append(count / total)
    
    return torch.tensor(bitstrings, dtype=torch.long), torch.tensor(probs, dtype=torch.float32)

def tensor_to_counts(
    bitstrings: torch.Tensor,
    probs: torch.Tensor,
    shots: int = 1000,
) -> Dict[str, int]:
    """Convert tensor back to counts dictionary."""
    counts = {}
    for i in range(len(bitstrings)):
        bs = ''.join(str(b.item()) for b in bitstrings[i])
        counts[bs] = int(probs[i].item() * shots)
    return counts

def pad_bitstrings(
    bitstrings: torch.Tensor,
    max_len: int,
    pad_value: int = -1,
) -> torch.Tensor:
    """Pad bitstrings to fixed length."""
    M = bitstrings.shape[0]
    if M >= max_len:
        return bitstrings[:max_len]
    padded = torch.full((max_len, bitstrings.shape[1]), pad_value, dtype=bitstrings.dtype)
    padded[:M] = bitstrings
    return padded
