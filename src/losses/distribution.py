"""Distribution losses for N2LN-QEM (TDD §4.1.1).

KL divergence, TVD, Chi-squared, and composite distribution loss.
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


def composite_dist_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Composite distribution loss: alpha*KL + beta*TVD + gamma*Chi2.
    
    Args:
        pred: Predicted distribution (B, M)
        target: Target distribution (B, M)
        alpha: KL divergence weight
        beta: TVD weight
        gamma: Chi-squared weight
        eps: Small epsilon for numerical stability
    
    Returns:
        Composite loss (scalar)
    """
    # Ensure pred and target have same shape
    if pred.shape != target.shape:
        # Handle mismatched shapes by truncating to min size
        min_dim = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_dim]
        target = target[:, :min_dim]
    
    loss = 0.0
    if alpha > 0:
        loss += alpha * kl_loss(pred, target, eps)
    if beta > 0:
        loss += beta * tvd_loss(pred, target)
    if gamma > 0:
        loss += gamma * chi2_loss(pred, target, eps)
    return loss


class CompositeDistributionLoss(nn.Module):
    """Module wrapper for composite distribution loss."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return composite_dist_loss(pred, target, self.alpha, self.beta, self.gamma, self.eps)