"""Sanity tests for training loop (TDD §4.2)."""

import os
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.models.n2ln import N2LNQEM
from src.training.trainer import N2LNTrainer


@pytest.fixture
def tiny_model():
    return N2LNQEM(d_model=32, n_heads=2, n_isab=1, n_sab=1, d_ff=64, m=4, decoder_hidden=64, dropout=0.0)


@pytest.fixture
def tiny_dataset():
    """Create a tiny synthetic dataset for overfit test."""
    B, M, n_qubits = 8, 4, 3
    bitstrings = torch.randint(0, 2, (B, M, n_qubits))
    counts = torch.rand(B, M, 1)
    counts = counts / counts.sum(dim=1, keepdim=True)
    target_sn = torch.rand(B, 2**n_qubits)
    target_sn = target_sn / target_sn.sum(dim=-1, keepdim=True)
    target_hn = torch.rand(B, 2**n_qubits)
    target_hn = target_hn / target_hn.sum(dim=-1, keepdim=True)
    dataset = TensorDataset(bitstrings, counts, target_sn, target_hn)
    return DataLoader(dataset, batch_size=8)


def test_trainer_instantiation(tiny_model):
    trainer = N2LNTrainer(tiny_model, lr=1e-3)
    assert trainer is not None


def test_overfit_tiny_batch(tiny_model, tiny_dataset):
    """Model should overfit a tiny batch quickly."""
    trainer = N2LNTrainer(tiny_model, lr=1e-3, weight_decay=0.0, grad_clip=0.0)

    initial_loss = None
    for epoch in range(100):
        metrics = trainer.train_epoch(tiny_dataset, mode="sn_d_only", epoch=epoch)
        if initial_loss is None:
            initial_loss = metrics["loss"]
        trainer.current_epoch = epoch

    final_loss = metrics["loss"]
    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.5, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


def test_validate_runs(tiny_model, tiny_dataset):
    trainer = N2LNTrainer(tiny_model, lr=1e-3)
    metrics = trainer.validate(tiny_dataset, mode="sn_d_only")
    assert "val_loss" in metrics


def test_save_and_load_checkpoint(tiny_model, tiny_dataset, tmp_path):
    trainer = N2LNTrainer(tiny_model, lr=1e-3, checkpoint_dir=str(tmp_path))
    trainer.train_epoch(tiny_dataset, mode="sn_d_only", epoch=0)
    path = trainer.save_checkpoint("test.pt")
    assert os.path.exists(path)

    # Create new trainer and load
    model2 = N2LNQEM(d_model=32, n_heads=2, n_isab=1, n_sab=1, d_ff=64, m=4, decoder_hidden=64, dropout=0.0)
    trainer2 = N2LNTrainer(model2, lr=1e-3, checkpoint_dir=str(tmp_path))
    trainer2.load_checkpoint("test.pt")
    assert trainer2.current_epoch == 0


def test_set_lr(tiny_model):
    trainer = N2LNTrainer(tiny_model, lr=1e-3)
    trainer.set_lr(1e-4)
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-4
