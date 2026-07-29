"""
Noise-Aware Data Augmentation for N2LN-QEM (TDD §4.2.2).

Bootstrap resampling: generate low-shot samples from high-shot data.
"""

import numpy as np
import torch
from typing import Dict, List, Optional


def bootstrap_resample(
    high_shot_counts: Dict[str, int],
    n_qubits: int,
    low_shots: int = 100,
    num_samples: int = 1,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Generate low-shot samples via multinomial resampling."""
    if seed is not None:
        np.random.seed(seed)

    total = sum(high_shot_counts.values())
    bitstrings = list(high_shot_counts.keys())
    probs = np.array([count / total for count in high_shot_counts.values()])

    samples = []
    for _ in range(num_samples):
        indices = np.random.choice(len(bitstrings), size=low_shots, p=probs, replace=True)
        unique, counts = np.unique(indices, return_counts=True)
        sampled = {bitstrings[i]: counts[j] / low_shots for j, i in enumerate(unique)}
        samples.append(sampled)

    return samples


def interpolate_noise_levels(
    counts1: Dict[str, int],
    counts2: Dict[str, int],
    alpha: float = 0.5,
) -> Dict[str, int]:
    """Interpolate between two noise levels."""
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    probs1 = {k: v / total1 for k, v in counts1.items()}
    probs2 = {k: v / total2 for k, v in counts2.items()}

    all_keys = set(probs1.keys()) | set(probs2.keys())
    interpolated = {}
    for k in all_keys:
        p1 = probs1.get(k, 0.0)
        p2 = probs2.get(k, 0.0)
        interpolated[k] = (1 - alpha) * p1 + alpha * p2

    total_shots = int((total1 + total2) / 2)
    return {k: max(1, int(v * total_shots)) for k, v in interpolated.items() if v > 0}
