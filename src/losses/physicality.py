"""Physicality regularization loss (TDD §4.1).

Penalizes non-physical outputs: negative probabilities and non-normalized distributions.
"""

import torch
import torch.nn as nn


def physicality_loss(
    pred: torch.Tensor,
    lambda_neg: float = 1.0,
    lambda_norm: float = 1.0,
) -> torch.Tensor:
    """Penalize non-physical probability distributions.

    L_phys = lambda_neg * max(0, -min(p))^2 + lambda_norm * |sum(p) - 1|^2

    Args:
        pred: Predicted probability distribution (B, D).
        lambda_neg: Weight for negativity penalty.
        lambda_norm: Weight for normalization penalty.

    Returns:
        Scalar physicality loss (mean over batch).
    """
    # Negativity penalty: penalize any negative probability mass
    min_vals = pred.min(dim=-1).values  # (B,)
    neg_penalty = (torch.clamp(-min_vals, min=0) ** 2).mean()

    # Normalization penalty: penalize deviation from sum=1
    sums = pred.sum(dim=-1)  # (B,)
    norm_penalty = ((sums - 1.0) ** 2).mean()

    return lambda_neg * neg_penalty + lambda_norm * norm_penalty


class PhysicalityLoss(nn.Module):
    """Trainable physicality regularization module.

    Args:
        lambda_neg: Weight for negativity penalty.
        lambda_norm: Weight for normalization penalty.
    """

    def __init__(self, lambda_neg: float = 1.0, lambda_norm: float = 1.0):
        super().__init__()
        self.lambda_neg = lambda_neg
        self.lambda_norm = lambda_norm

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return physicality_loss(pred, self.lambda_neg, self.lambda_norm)
