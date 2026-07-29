"""Cross-stage consistency loss (TDD §4.1).

Encourages HN-E head to produce consistent results whether given
raw low-shot input or SN-D-denoised input.
"""

import torch
import torch.nn as nn

from src.losses.distribution import tvd_loss


def consistency_loss(
    hn_from_sn: torch.Tensor,
    hn_from_raw: torch.Tensor,
) -> torch.Tensor:
    """Cross-stage consistency: TVD between two HN-E pathways.

    L_consist = TVD(HN-E(SN-D(x_low)), HN-E(x_high))

    Args:
        hn_from_sn: HN-E output when given SN-D-denoised input (B, D).
        hn_from_raw: HN-E output when given raw high-shot input (B, D).

    Returns:
        Scalar consistency loss.
    """
    return tvd_loss(hn_from_sn, hn_from_raw)


class ConsistencyLoss(nn.Module):
    """Trainable consistency loss module."""

    def forward(self, hn_from_sn: torch.Tensor, hn_from_raw: torch.Tensor) -> torch.Tensor:
        return consistency_loss(hn_from_sn, hn_from_raw)
