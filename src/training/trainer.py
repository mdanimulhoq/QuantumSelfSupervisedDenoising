"""Training loop for N2LN-QEM (TDD §4.2).

AdamW optimizer, gradient clipping, wandb logging, checkpointing.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.distribution import CompositeDistributionLoss
from src.losses.physicality import PhysicalityLoss
from src.losses.consistency import ConsistencyLoss
from src.utils.device import get_device
from src.utils.seeding import set_seed


class N2LNTrainer:
    """Trainer for N2LN-QEM model.

    Args:
        model: N2LNQEM model instance.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm.
        loss_weights: Dict with keys:
            'kl', 'tvd', 'chi2', 'sharpness', 'entropy_floor',
            'sharpness_margin', 'entropy_tolerance',
            'consistency', 'physicality'.
        checkpoint_dir: Directory for saving checkpoints.
        device: torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
        checkpoint_dir: str = "./checkpoints",
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = get_device() if device is None else torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss weights with 🔥 NEW: sharpness and entropy_floor
        if loss_weights is None:
            loss_weights = {
                "kl": 0.1,              # কমিয়ে দেওয়া হয়েছে (KL spread করে)
                "tvd": 2.0,             # বাড়ানো হয়েছে (ভুল জায়গায় mass দিলে শাস্তি)
                "chi2": 0.0,
                "sharpness": 0.5,       # 🔥 NEW: max_pred কে target_max-এর কাছে ধাক্কা দেবে
                "entropy_floor": 0.0,   # 🔥 NEW: (ঐচ্ছিক) এনট্রপি নিয়ন্ত্রণ
                "sharpness_margin": 0.02,
                "entropy_tolerance": 0.05,
                "consistency": 0.3,
                "physicality": 0.0,
            }
        self.loss_weights = loss_weights

        # Loss modules - 🔥 PASS sharpness & entropy_floor to distribution loss
        self.dist_loss_fn = CompositeDistributionLoss(
            alpha=loss_weights.get("kl", 1.0),
            beta=loss_weights.get("tvd", 0.5),
            gamma=loss_weights.get("chi2", 0.1),
            sharpness=loss_weights.get("sharpness", 0.0),
            entropy_floor=loss_weights.get("entropy_floor", 0.0),
            sharpness_margin=loss_weights.get("sharpness_margin", 0.02),
            entropy_tolerance=loss_weights.get("entropy_tolerance", 0.05),
        )
        self.phys_loss_fn = PhysicalityLoss()
        self.consist_loss_fn = ConsistencyLoss()

        self.current_epoch = 0
        self.best_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        mode: str = "sn_d_only",
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader yielding dict with 'bitstrings', 'counts', 'target_sn', 'target_hn'
            mode: Training mode - "sn_only", "hn_only", or "unified"
            epoch: Current epoch number.

        Returns:
            Dict of average loss values.
        """
        self.model.train()
        total_loss = 0.0
        total_dist = 0.0
        total_phys = 0.0
        total_consist = 0.0
        n_batches = 0

        for batch in train_loader:
            # Extract tensors from dict
            bitstrings = batch['bitstrings'].to(self.device)
            counts = batch['counts'].to(self.device)
            target_sn = batch['target_sn'].to(self.device)
            target_hn = batch['target_hn'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            sn_out, hn_out = self.model(bitstrings, counts, mode=mode)

            # Compute losses
            loss = 0.0
            dist_loss = 0.0
            phys_loss = 0.0
            consist_loss = 0.0

            if mode in ["sn_only", "unified"] and sn_out is not None:
                dist_loss += self.dist_loss_fn(sn_out, target_sn)
                phys_loss += self.phys_loss_fn(sn_out)

            if mode in ["hn_only", "unified"] and hn_out is not None:
                dist_loss += self.dist_loss_fn(hn_out, target_hn)
                phys_loss += self.phys_loss_fn(hn_out)

            loss += dist_loss

            if self.loss_weights.get("physicality", 0) > 0:
                loss += self.loss_weights["physicality"] * phys_loss

            # Consistency loss (unified mode only)
            if mode == "unified" and sn_out is not None and hn_out is not None:
                consist_loss = self.consist_loss_fn(hn_out, sn_out)
                if self.loss_weights.get("consistency", 0) > 0:
                    loss += self.loss_weights["consistency"] * consist_loss

            # Backward
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            total_dist += dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss
            total_phys += phys_loss.item() if isinstance(phys_loss, torch.Tensor) else phys_loss
            total_consist += consist_loss.item() if isinstance(consist_loss, torch.Tensor) else consist_loss
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "dist_loss": total_dist / n_batches,
            "phys_loss": total_phys / n_batches,
            "consist_loss": total_consist / n_batches,
        }

    def validate(
        self, val_loader: DataLoader, mode: str = "sn_only"
    ) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            val_loader: Validation DataLoader.
            mode: Evaluation mode.

        Returns:
            Dict of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                bitstrings = batch['bitstrings'].to(self.device)
                counts = batch['counts'].to(self.device)
                target_sn = batch['target_sn'].to(self.device)
                target_hn = batch['target_hn'].to(self.device)

                sn_out, hn_out = self.model(bitstrings, counts, mode=mode)

                loss = 0.0
                if mode in ["sn_only", "unified"] and sn_out is not None:
                    loss += self.dist_loss_fn(sn_out, target_sn)
                if mode in ["hn_only", "unified"] and hn_out is not None:
                    loss += self.dist_loss_fn(hn_out, target_hn)

                total_loss += loss.item()
                n_batches += 1

        return {"val_loss": total_loss / n_batches}

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> str:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Full path to saved checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            },
            path,
        )
        return path

    def load_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

    def set_lr(self, lr: float) -> None:
        """Update learning rate.

        Args:
            lr: New learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr