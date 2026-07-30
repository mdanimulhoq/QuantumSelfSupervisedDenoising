"""Distribution losses for N2LN-QEM (TDD §4.1.1).

KL divergence, TVD, Chi-squared, Sharpness, and composite distribution loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL divergence loss: KL(target || pred).
    
    Args:
        pred: Predicted distribution (B, M)
        target: Target distribution (B, M)
        eps: Small epsilon for numerical stability
    
    Returns:
        KL divergence (scalar)
    """
    pred_safe = torch.clamp(pred, min=eps, max=1.0)
    target_safe = torch.clamp(target, min=eps, max=1.0)
    kl = target_safe * (torch.log(target_safe + eps) - torch.log(pred_safe + eps))
    return kl.sum(dim=-1).mean()


def tvd_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Total Variation Distance: 0.5 * sum(|pred - target|).
    
    Args:
        pred: Predicted distribution (B, M)
        target: Target distribution (B, M)
    
    Returns:
        TVD (scalar)
    """
    return 0.5 * torch.abs(pred - target).sum(dim=-1).mean()


def chi2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Chi-squared divergence: sum((pred - target)^2 / target).
    
    Args:
        pred: Predicted distribution (B, M)
        target: Target distribution (B, M)
        eps: Small epsilon for numerical stability
    
    Returns:
        Chi-squared divergence (scalar)
    """
    target_safe = torch.clamp(target, min=eps, max=1.0)
    chi2 = ((pred - target_safe) ** 2) / (target_safe + eps)
    return chi2.sum(dim=-1).mean()


# =============================================================================
# 🔥 NEW: Sharpness & Entropy Regularizers (Step C)
# =============================================================================

def sharpness_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    """Encourage pred_max - target_max to be >= margin (sharp peaks).

    Loss = ReLU(margin - (pred_max - target_max))

    Args:
        pred: Predicted distribution (B, M)
        target: Target distribution (B, M)
        margin: Minimum desired difference between pred_max and target_max

    Returns:
        Sharpness loss (scalar)
    """
    pred_max = pred.max(dim=-1).values
    target_max = target.max(dim=-1).values
    diff = pred_max - target_max
    return F.relu(margin - diff).mean()


def entropy_floor_loss(pred: torch.Tensor, target: torch.Tensor, tolerance: float = 0.05) -> torch.Tensor:
    """Penalize if pred_entropy > target_entropy + tolerance (model too uniform).

    Args:
        pred: Predicted distribution (B, M)
        target: Target distribution (B, M)
        tolerance: Allowed entropy difference in nats

    Returns:
        Entropy floor loss (scalar)
    """
    eps = 1e-8
    pred_h = -(pred * torch.log(pred + eps)).sum(dim=-1)          # (B,)
    target_h = -(target * torch.log(target + eps)).sum(dim=-1)    # (B,)
    return F.relu(pred_h - target_h + tolerance).mean()


# =============================================================================
# Composite Loss with Sharpness and Entropy Floor
# =============================================================================

def composite_dist_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    sharpness: float = 0.0,
    entropy_floor: float = 0.0,
    sharpness_margin: float = 0.02,
    entropy_tolerance: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Composite distribution loss: alpha*KL + beta*TVD + gamma*Chi2 + sharpness*Sharpness + entropy_floor*EntropyFloor.
    
    Args:
        pred: Predicted distribution (B, M)
        target: Target distribution (B, M)
        alpha: KL divergence weight
        beta: TVD weight
        gamma: Chi-squared weight
        sharpness: Sharpness loss weight (0 to disable)
        entropy_floor: Entropy floor loss weight (0 to disable)
        sharpness_margin: Margin for sharpness loss
        entropy_tolerance: Tolerance for entropy floor loss
        eps: Small epsilon for numerical stability
    
    Returns:
        Composite loss (scalar)
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch in composite_dist_loss: "
            f"pred={pred.shape}, target={target.shape}. "
            "Ensure both distributions have the same support size."
        )

    # Renormalize to ensure proper probability distributions
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    loss = 0.0
    if alpha > 0:
        loss += alpha * kl_loss(pred, target, eps)
    if beta > 0:
        loss += beta * tvd_loss(pred, target)
    if gamma > 0:
        loss += gamma * chi2_loss(pred, target, eps)
    if sharpness > 0:
        loss += sharpness * sharpness_loss(pred, target, sharpness_margin)
    if entropy_floor > 0:
        loss += entropy_floor * entropy_floor_loss(pred, target, entropy_tolerance)
    return loss


class CompositeDistributionLoss(nn.Module):
    """Module wrapper for composite distribution loss with sharpness and entropy floor."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        sharpness: float = 0.0,
        entropy_floor: float = 0.0,
        sharpness_margin: float = 0.02,
        entropy_tolerance: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sharpness = sharpness
        self.entropy_floor = entropy_floor
        self.sharpness_margin = sharpness_margin
        self.entropy_tolerance = entropy_tolerance
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return composite_dist_loss(
            pred,
            target,
            self.alpha,
            self.beta,
            self.gamma,
            self.sharpness,
            self.entropy_floor,
            self.sharpness_margin,
            self.entropy_tolerance,
            self.eps,
        )