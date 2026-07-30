"""Train HN-E head for Experiment 2 (TDD §4.2, §6.4).

Trains the HN-E (Hardware-Noise Extrapolation) head on noise-scaled pairs.
Loads SN-D checkpoint for initialization (optional).
Phase 2: Joint training — all parameters (encoder, transformer, both heads) are trainable.
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.n2ln import N2LNQEM
from src.training.trainer import N2LNTrainer
from src.utils.seeding import set_seed
from src.utils.device import get_device


# =============================================================================
# Helper: Align support (same as SN-D)
# =============================================================================

def align_support(low_bs, low_probs, high_bs, high_probs):
    """Align low and high distributions to a union support."""
    def to_key(row):
        return tuple(int(x) for x in row.tolist())

    support = {}

    # Low shot entries (low noise target)
    for bs, p in zip(low_bs, low_probs.squeeze(-1)):
        key = to_key(bs)
        if key not in support:
            support[key] = [0.0, 0.0]
        support[key][0] = float(p)

    # High shot entries (high noise input)
    for bs, p in zip(high_bs, high_probs):
        key = to_key(bs)
        if key not in support:
            support[key] = [0.0, 0.0]
        support[key][1] = float(p)

    # Deterministic ordering
    keys = sorted(support.keys())

    union_bs = torch.tensor(keys, dtype=torch.long)
    low_aligned = torch.tensor([support[k][0] for k in keys], dtype=torch.float32)
    high_aligned = torch.tensor([support[k][1] for k in keys], dtype=torch.float32)

    # Normalize
    low_aligned = low_aligned / low_aligned.sum().clamp_min(1e-8)
    high_aligned = high_aligned / high_aligned.sum().clamp_min(1e-8)

    return union_bs, low_aligned.unsqueeze(1), high_aligned


# =============================================================================
# Dataset Class
# =============================================================================

class HNEDataset(Dataset):
    """Dataset for HN-E training."""
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.data = []
        
        with h5py.File(h5_path, 'r') as f:
            # Detect noise scales from keys
            bitstring_keys = [key for key in f.keys() if key.startswith('bitstrings_')]
            scales = []
            for key in bitstring_keys:
                try:
                    scale_str = key.split('_')[1]
                    scales.append(float(scale_str))
                except:
                    pass
            scales = sorted(scales)
            if not scales:
                raise ValueError("No noise scales found in HDF5 file.")
            
            self.high_scale = max(scales)
            self.low_scale = min(scales)
            
            n_samples = len(f['n_qubits'])
            for i in range(n_samples):
                try:
                    # Get actual qubit count
                    n_qubits = f['n_qubits'][i]
                    
                    # High-noise data (input)
                    high_bs_flat = f[f'bitstrings_{self.high_scale}'][i]
                    high_shape = f[f'bitstrings_{self.high_scale}_shape'][i]
                    # Handle flat data properly
                    if len(high_shape) == 2 and high_bs_flat.size == high_shape[0] * high_shape[1]:
                        high_bs = high_bs_flat.reshape(high_shape[0], high_shape[1])
                    else:
                        high_bs = high_bs_flat.reshape(-1, n_qubits)
                    
                    high_probs = f[f'probs_{self.high_scale}'][i]
                    high_probs = high_probs.astype(np.float32)
                    
                    # Low-noise data (target)
                    low_bs_flat = f[f'bitstrings_{self.low_scale}'][i]
                    low_shape = f[f'bitstrings_{self.low_scale}_shape'][i]
                    if len(low_shape) == 2 and low_bs_flat.size == low_shape[0] * low_shape[1]:
                        low_bs = low_bs_flat.reshape(low_shape[0], low_shape[1])
                    else:
                        low_bs = low_bs_flat.reshape(-1, n_qubits)
                    
                    low_probs = f[f'probs_{self.low_scale}'][i]
                    low_probs = low_probs.astype(np.float32)
                    
                    self.data.append({
                        'high_bs': torch.tensor(high_bs, dtype=torch.long),
                        'high_probs': torch.tensor(high_probs, dtype=torch.float32).unsqueeze(1),
                        'low_bs': torch.tensor(low_bs, dtype=torch.long),
                        'low_probs': torch.tensor(low_probs, dtype=torch.float32),
                        'n_qubits': n_qubits,
                    })
                except Exception as e:
                    print(f"⚠️ Skipping sample {i}: {e}")
                    continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Align support: high (input) and low (target)
        bitstrings, counts, target_sn = align_support(
            item['high_bs'],      # input high-noise bitstrings
            item['high_probs'],   # input high-noise probs
            item['low_bs'],       # target low-noise bitstrings
            item['low_probs'],    # target low-noise probs
        )
        
        # For HN-E, we use high-noise as input, low-noise as target
        return {
            'bitstrings': bitstrings,
            'counts': counts,                        # high-noise counts
            'target_sn': target_sn,                  # not used (SN-D frozen)
            'target_hn': target_sn,                  # low-noise target for HN-E
            'n_qubits': item['n_qubits'],
        }


def collate_hne_batch(batch):
    """Collate function for HN-E batch."""
    max_n_qubits = max([b['n_qubits'] for b in batch])
    max_bs = max([b['bitstrings'].shape[0] for b in batch])
    max_bs_target = max([b['target_hn'].shape[0] for b in batch])
    
    batched_bitstrings = []
    batched_counts = []
    batched_target_sn = []
    batched_target_hn = []
    batched_n_qubits = []
    
    for b in batch:
        n_q = b['n_qubits']
        
        # Pad bitstrings
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
        
        # Pad counts
        counts = b['counts']
        pad_len_count = max_bs - counts.shape[0]
        if pad_len_count > 0:
            counts_pad = torch.zeros((pad_len_count, 1), dtype=counts.dtype)
            counts = torch.cat([counts, counts_pad], dim=0)
        batched_counts.append(counts)
        
        # Pad target
        target = b['target_hn']
        pad_len_target = max_bs_target - target.shape[0]
        if pad_len_target > 0:
            target_pad = torch.zeros((pad_len_target,), dtype=target.dtype)
            target = torch.cat([target, target_pad], dim=0)
        batched_target_sn.append(target)  # not used
        batched_target_hn.append(target)
        
        batched_n_qubits.append(n_q)
    
    return {
        'bitstrings': torch.stack(batched_bitstrings),
        'counts': torch.stack(batched_counts),
        'target_sn': torch.stack(batched_target_sn),
        'target_hn': torch.stack(batched_target_hn),
        'n_qubits': torch.tensor(batched_n_qubits, dtype=torch.long),
    }


# =============================================================================
# Training Function
# =============================================================================

def train_hne(config_path: str):
    """Main training function for HN-E experiment."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data paths
    data_dir = Path(config['data']['data_dir'])
    train_path = data_dir / 'exp2_hne_train.h5'
    val_path = data_dir / 'exp2_hne_val.h5'
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    # Create datasets
    train_dataset = HNEDataset(str(train_path))
    val_dataset = HNEDataset(str(val_path)) if val_path.exists() else None
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_hne_batch,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_hne_batch,
        )
    
    # Create model
    model = N2LNQEM(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_isab=config['model']['n_isab'],
        n_sab=config['model']['n_sab'],
        d_ff=config['model']['d_ff'],
        m=config['model']['m'],
        decoder_hidden=config['model']['decoder_hidden'],
        dropout=config['model']['dropout'],
        max_qubits=8,
    )
    model.to(device)
    
    # Load SN-D checkpoint for initialization (optional, TDD Phase 2 uses it)
    sn_checkpoint_path = Path('checkpoints/exp1_snd/best_model.pt')
    if sn_checkpoint_path.exists():
        print(f"Loading SN-D checkpoint from: {sn_checkpoint_path}")
        checkpoint = torch.load(sn_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("   ✅ SN-D weights loaded for initialization")
    else:
        print(f"⚠️ SN-D checkpoint not found: {sn_checkpoint_path}")
        print("   Training HN-E from scratch")
    
    # 🔥 TDD-সঙ্গতিপূর্ণ: সব প্যারামিটার trainable রাখুন (কোনও ফ্রিজ নয়)
    for param in model.parameters():
        param.requires_grad = True
    print("   ✅ All parameters (encoder, transformer, both heads) are trainable")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss weights (Phase 2: HN-E only, but backbone updates)
    loss_weights = {
        'kl': config['loss']['alpha'],
        'tvd': config['loss']['beta'],
        'chi2': config['loss']['gamma'],
        'physicality': config['loss']['physicality'],
        'consistency': 0.0,  # Phase 2 doesn't use consistency loss
    }
    
    # Create trainer
    trainer = N2LNTrainer(
        model=model,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        grad_clip=config['training']['grad_clip'],
        loss_weights=loss_weights,
        checkpoint_dir='checkpoints/exp2_hne',
        device=str(device),
    )
    
    # Training loop
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = config['training']['early_stop_patience']
    
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Train HN-E only (but backbone updates because all params are trainable)
        train_metrics = trainer.train_epoch(
            train_loader,
            mode='hn_only',
            epoch=epoch,
        )
        
        # Validate
        if val_loader:
            val_metrics = trainer.validate(val_loader, mode='hn_only')
            val_loss = val_metrics['val_loss']
        else:
            val_loss = train_metrics['loss']
        
        # Log
        log_msg = f"Epoch {epoch+1:3d}/{epochs} | "
        log_msg += f"Train Loss: {train_metrics['loss']:.4f} | "
        log_msg += f"Val Loss: {val_loss:.4f}"
        print(log_msg)
        
        # Early stopping
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
    
    print("\n✅ Training complete!")
    print(f"   Best model saved: checkpoints/exp2_hne/best_model.pt")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    config_path = "experiments/exp2_hne/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    train_hne(config_path)