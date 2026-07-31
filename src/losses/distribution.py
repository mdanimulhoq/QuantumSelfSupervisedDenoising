"""Distribution losses for N2LN-QEM (TDD §4.1.1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def tvd_loss(pred, target, eps=1e-8):
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return 0.5 * torch.abs(pred - target).sum(dim=-1).mean()

def kl_loss(pred, target, eps=1e-8):
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = target * (torch.log(target + eps) - torch.log(pred + eps))
    return kl.sum(dim=-1).mean()

def chi2_loss(pred, target, eps=1e-8):
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return ((pred - target) ** 2 / (target + eps)).sum(dim=-1).mean()

def sharpness_loss(pred, target, margin=0.02):
    pred_max = pred.max(dim=-1).values
    target_max = target.max(dim=-1).values
    return F.relu(margin - (pred_max - target_max)).mean()

def entropy_floor_loss(pred, target, tolerance=0.05):
    eps = 1e-8
    pred_h = -(pred * torch.log(pred + eps)).sum(dim=-1)
    target_h = -(target * torch.log(target + eps)).sum(dim=-1)
    return F.relu(pred_h - target_h + tolerance).mean()

def composite_dist_loss(pred, target, alpha=0.1, beta=2.0, gamma=0.0,
                        sharpness=1.5, entropy_floor=0.0,
                        sharpness_margin=0.02, entropy_tolerance=0.05,
                        eps=1e-8):
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
    def __init__(self, alpha=0.1, beta=2.0, gamma=0.0,
                 sharpness=1.5, entropy_floor=0.0,
                 sharpness_margin=0.02, entropy_tolerance=0.05,
                 eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sharpness = sharpness
        self.entropy_floor = entropy_floor
        self.sharpness_margin = sharpness_margin
        self.entropy_tolerance = entropy_tolerance
        self.eps = eps

    def forward(self, pred, target):
        return composite_dist_loss(
            pred, target,
            self.alpha, self.beta, self.gamma,
            self.sharpness, self.entropy_floor,
            self.sharpness_margin, self.entropy_tolerance,
            self.eps
        )
