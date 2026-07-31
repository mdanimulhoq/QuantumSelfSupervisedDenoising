"""Train SN-D head for Experiment 1 (TDD §4.2, §6.4)."""

import os
import sys
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import h5py
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.n2ln import N2LNQEM
from src.training.trainer import N2LNTrainer
from src.utils.seeding import set_seed
from src.utils.device import get_device
from src.losses.distribution import CompositeDistributionLoss
from src.losses.physicality import PhysicalityLoss


def align_support(low_bs, low_probs, high_bs, high_probs):
    def to_key(row):
        return tuple(int(x) for x in row.tolist())

    support = {}
    for bs, p in zip(low_bs, low_probs.squeeze(-1)):
        key = to_key(bs)
        if key not in support:
            support[key] = [0.0, 0.0]
        support[key][0] = float(p)

    for bs, p in zip(high_bs, high_probs):
        key = to_key(bs)
        if key not in support:
            support[key] = [0.0, 0.0]
        support[key][1] = float(p)

    keys = sorted(support.keys())
    union_bs = torch.tensor(keys, dtype=torch.long)
    low_aligned = torch.tensor([support[k][0] for k in keys], dtype=torch.float32)
    high_aligned = torch.tensor([support[k][1] for k in keys], dtype=torch.float32)

    # 🔥 NO renormalization
    return union_bs, low_aligned.unsqueeze(1), high_aligned


class SNDDataset(Dataset):
    def __init__(self, h5_path: str, n_qubits: int):
        self.h5_path = h5_path
        self.n_qubits = n_qubits
        self.data = []
        with h5py.File(h5_path, 'r') as f:
            n_samples = len(f['n_qubits'])
            for i in range(n_samples):
                try:
                    actual_n = f['n_qubits'][i]
                    low_bs_flat = f['low_bitstrings'][i]
                    low_shape = f['low_bitstrings_shape'][i]
                    if len(low_shape) == 2 and low_bs_flat.size == low_shape[0] * low_shape[1]:
                        low_bs = low_bs_flat.reshape(low_shape[0], low_shape[1])
                    else:
                        low_bs = low_bs_flat.reshape(-1, actual_n)
                    low_probs = f['low_probs'][i].astype(np.float32)

                    high_bs_flat = f['high_bitstrings'][i]
                    high_shape = f['high_bitstrings_shape'][i]
                    if len(high_shape) == 2 and high_bs_flat.size == high_shape[0] * high_shape[1]:
                        high_bs = high_bs_flat.reshape(high_shape[0], high_shape[1])
                    else:
                        high_bs = high_bs_flat.reshape(-1, actual_n)
                    high_probs = f['high_probs'][i].astype(np.float32)

                    self.data.append({
                        'low_bitstrings': torch.tensor(low_bs, dtype=torch.long),
                        'low_probs': torch.tensor(low_probs, dtype=torch.float32).unsqueeze(1),
                        'high_bitstrings': torch.tensor(high_bs, dtype=torch.long),
                        'high_probs': torch.tensor(high_probs, dtype=torch.float32),
                        'n_qubits': actual_n,
                    })
                except Exception as e:
                    print(f"⚠️ Skipping sample {i}: {e}")
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        bitstrings, counts, target_sn = align_support(
            item['low_bitstrings'],
            item['low_probs'],
            item['high_bitstrings'],
            item['high_probs'],
        )
        return {
            'bitstrings': bitstrings,
            'counts': counts,
            'target_sn': target_sn,
            'target_hn': target_sn.clone(),
            'n_qubits': item['n_qubits'],
        }


def collate_snd_batch(batch):
    max_n_qubits = max([b['n_qubits'] for b in batch])
    max_bs = max([b['bitstrings'].shape[0] for b in batch])
    max_bs_target = max([b['target_sn'].shape[0] for b in batch])
    batched_bitstrings, batched_counts, batched_target_sn, batched_target_hn, batched_n_qubits = [], [], [], [], []

    for b in batch:
        n_q = b['n_qubits']
        bs = b['bitstrings']
        if bs.shape[1] < max_n_qubits:
            pad_len = max_n_qubits - bs.shape[1]
            bs_pad = torch.full((bs.shape[0], pad_len), -1, dtype=bs.dtype)
            bs = torch.cat([bs, bs_pad], dim=1)
        pad_len = max_bs - bs.shape[0]
        if pad_len > 0:
            bs_pad = torch.full((pad_len, max_n_qubits), -1, dtype=bs.dtype)
            bs = torch.cat([bs, bs_pad], dim=0)
        batched_bitstrings.append(bs)

        counts = b['counts']
        pad_len_count = max_bs - counts.shape[0]
        if pad_len_count > 0:
            counts_pad = torch.zeros((pad_len_count, 1), dtype=counts.dtype)
            counts = torch.cat([counts, counts_pad], dim=0)
        batched_counts.append(counts)

        target = b['target_sn']
        pad_len_target = max_bs_target - target.shape[0]
        if pad_len_target > 0:
            target_pad = torch.zeros((pad_len_target,), dtype=target.dtype)
            target = torch.cat([target, target_pad], dim=0)
        batched_target_sn.append(target)
        batched_target_hn.append(target)
        batched_n_qubits.append(n_q)

    return {
        'bitstrings': torch.stack(batched_bitstrings),
        'counts': torch.stack(batched_counts),
        'target_sn': torch.stack(batched_target_sn),
        'target_hn': torch.stack(batched_target_hn),
        'n_qubits': torch.tensor(batched_n_qubits, dtype=torch.long),
    }


def train_snd(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(config['data']['data_dir'])
    train_path = data_dir / 'exp1_snd_train.h5'
    val_path = data_dir / 'exp1_snd_val.h5'

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    train_dataset = SNDDataset(str(train_path), config['data']['n_qubits'][0])
    val_dataset = SNDDataset(str(val_path), config['data']['n_qubits'][0]) if val_path.exists() else None

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_snd_batch,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_snd_batch,
        )

    # 🔥 FIX: সরাসরি BIG MODEL প্যারামিটার ব্যবহার করুন (config উপেক্ষা করে)
    print("🔥 Using BIG MODEL: d_model=128, n_heads=8, n_isab=2")
    model = N2LNQEM(
        d_model=128,
        n_heads=8,
        n_isab=2,
        n_sab=1,
        d_ff=512,
        m=16,
        decoder_hidden=256,
        dropout=0.1,
        max_qubits=config['data']['n_qubits'][-1],
        use_mlp_scorer=False,
        temperature_floor=0.1,
    )
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_weights = {
        'kl': config['loss']['alpha'],
        'tvd': config['loss']['beta'],
        'chi2': config['loss']['gamma'],
        'physicality': config['loss']['physicality'],
        'consistency': 0.0,
        'sharpness': config['loss'].get('sharpness', 0.0),
        'sharpness_margin': config['loss'].get('sharpness_margin', 0.02),
        'entropy_floor': config['loss'].get('entropy_floor', 0.0),
        'entropy_tolerance': config['loss'].get('entropy_tolerance', 0.05),
    }

    trainer = N2LNTrainer(
        model=model,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        grad_clip=config['training']['grad_clip'],
        loss_weights=loss_weights,
        checkpoint_dir='checkpoints/exp1_snd',
        device=str(device),
    )

    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = config['training']['early_stop_patience']

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader, mode='sn_only', epoch=epoch)
        if val_loader:
            val_metrics = trainer.validate(val_loader, mode='sn_only')
            val_loss = val_metrics['val_loss']
        else:
            val_loss = train_metrics['loss']

        log_msg = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss:.4f}"
        print(log_msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            trainer.save_checkpoint('best_model.pt')
            print(f"   ✅ New best model saved (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break

    trainer.load_checkpoint('best_model.pt')
    test_path = data_dir / 'exp1_snd_test.h5'
    if test_path.exists():
        test_dataset = SNDDataset(str(test_path), config['data']['n_qubits'][0])
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_snd_batch,
        )
        test_metrics = trainer.validate(test_loader, mode='sn_only')
        print(f"\n📊 Test Loss: {test_metrics['val_loss']:.4f}")

    print("\n✅ Training complete!")
    print(f"   Best model saved: checkpoints/exp1_snd/best_model.pt")


if __name__ == "__main__":
    config_path = "experiments/exp1_snd/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    train_snd(config_path)
