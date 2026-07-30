"""Joint fine-tuning of SN-D and HN-E for Unified N2LN (TDD §4.2 Phase 3).

Loads SN-D and HN-E checkpoints, unfreezes both heads,
and fine-tunes with consistency loss.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import h5py

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

    low_aligned = low_aligned / low_aligned.sum().clamp_min(1e-8)
    high_aligned = high_aligned / high_aligned.sum().clamp_min(1e-8)

    return union_bs, low_aligned.unsqueeze(1), high_aligned


# =============================================================================
# Dataset for Unified Training
# =============================================================================

class UnifiedDataset(Dataset):
    """Dataset for Unified N2LN training (uses SN-D dataset)."""
    
    def __init__(self, h5_path: str, n_qubits: int):
        self.h5_path = h5_path
        self.n_qubits = n_qubits
        self.data = []
        
        with h5py.File(h5_path, 'r') as f:
            n_samples = len(f['n_qubits'])
            
            for i in range(n_samples):
                try:
                    actual_n = f['n_qubits'][i]
                    
                    # Low-shot data (input for SN-D)
                    low_bs_flat = f['low_bitstrings'][i]
                    low_shape = f['low_bitstrings_shape'][i]
                    if len(low_shape) == 2 and low_bs_flat.size == low_shape[0] * low_shape[1]:
                        low_bs = low_bs_flat.reshape(low_shape[0], low_shape[1])
                    else:
                        low_bs = low_bs_flat.reshape(-1, actual_n)
                    
                    low_probs = f['low_probs'][i]
                    low_probs = low_probs.astype(np.float32)
                    
                    # High-shot data (target for SN-D)
                    high_bs_flat = f['high_bitstrings'][i]
                    high_shape = f['high_bitstrings_shape'][i]
                    if len(high_shape) == 2 and high_bs_flat.size == high_shape[0] * high_shape[1]:
                        high_bs = high_bs_flat.reshape(high_shape[0], high_shape[1])
                    else:
                        high_bs = high_bs_flat.reshape(-1, actual_n)
                    
                    high_probs = f['high_probs'][i]
                    high_probs = high_probs.astype(np.float32)
                    
                    self.data.append({
                        'low_bs': torch.tensor(low_bs, dtype=torch.long),
                        'low_probs': torch.tensor(low_probs, dtype=torch.float32).unsqueeze(1),
                        'high_bs': torch.tensor(high_bs, dtype=torch.long),
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
        
        # Align support: low and high
        bitstrings, counts, target_sn = align_support(
            item['low_bs'],
            item['low_probs'],
            item['high_bs'],
            item['high_probs'],
        )
        
        return {
            'bitstrings': bitstrings,
            'counts': counts,
            'target_sn': target_sn,
            'target_hn': target_sn.clone(),  # Same target for HN-E in unified mode
            'n_qubits': item['n_qubits'],
        }


def collate_unified_batch(batch):
    """Collate function for unified training."""
    max_n_qubits = max([b['n_qubits'] for b in batch])
    max_bs = max([b['bitstrings'].shape[0] for b in batch])
    max_bs_target = max([b['target_sn'].shape[0] for b in batch])
    
    batched_bitstrings = []
    batched_counts = []
    batched_target_sn = []
    batched_target_hn = []
    batched_n_qubits = []
    
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


# =============================================================================
# Main Training Function
# =============================================================================

def train_unified(config_path: str):
    """Main training function for Unified N2LN."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    # Data paths
    data_dir = Path(config['data']['snd_data_dir'])
    train_path = data_dir / 'exp1_snd_train.h5'
    val_path = data_dir / 'exp1_snd_val.h5'
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    # Create datasets
    train_dataset = UnifiedDataset(str(train_path), config['data']['n_qubits'][0])
    val_dataset = UnifiedDataset(str(val_path), config['data']['n_qubits'][0]) if val_path.exists() else None
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_unified_batch,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_unified_batch,
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
    
    # Load SN-D checkpoint
    sn_path = Path(config['checkpoints']['sn_checkpoint'])
    hne_path = Path(config['checkpoints']['hne_checkpoint'])
    
    if sn_path.exists():
        print(f"Loading SN-D checkpoint from: {sn_path}")
        sn_ckpt = torch.load(sn_path, map_location=device)
        model.load_state_dict(sn_ckpt['model_state_dict'], strict=False)
        print("   ✅ SN-D weights loaded")
    else:
        print(f"⚠️ SN-D checkpoint not found: {sn_path}")
    
    if hne_path.exists():
        print(f"Loading HN-E checkpoint from: {hne_path}")
        hne_ckpt = torch.load(hne_path, map_location=device)
        model.load_state_dict(hne_ckpt['model_state_dict'], strict=False)
        print("   ✅ HN-E weights loaded")
    else:
        print(f"⚠️ HN-E checkpoint not found: {hne_path}")
    
    # Unfreeze both heads (they were frozen in HN-E training)
    for name, param in model.named_parameters():
        if 'hn_head' in name or 'sn_head' in name:
            param.requires_grad = True
    print("   ✅ Both heads unfrozen for joint fine-tuning")
    
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss weights (Phase 3: Unified with consistency)
    loss_weights = {
        'kl': config['loss']['alpha'],
        'tvd': config['loss']['beta'],
        'chi2': config['loss']['gamma'],
        'consistency': config['loss']['consistency'],
        'physicality': config['loss']['physicality'],
    }
    
    # Create trainer
    trainer = N2LNTrainer(
        model=model,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        grad_clip=config['training']['grad_clip'],
        loss_weights=loss_weights,
        checkpoint_dir=config['checkpoints']['output_dir'],
        device=str(device),
    )
    
    # Training loop
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = config['training']['early_stop_patience']
    
    print(f"\nStarting unified training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Train unified mode
        train_metrics = trainer.train_epoch(
            train_loader,
            mode='unified',
            epoch=epoch,
        )
        
        # Validate
        if val_loader:
            val_metrics = trainer.validate(val_loader, mode='unified')
            val_loss = val_metrics['val_loss']
        else:
            val_loss = train_metrics['loss']
        
        # Log
        log_msg = f"Epoch {epoch+1:3d}/{epochs} | "
        log_msg += f"Train Loss: {train_metrics['loss']:.4f} | "
        log_msg += f"Val Loss: {val_loss:.4f} | "
        log_msg += f"Consist: {train_metrics.get('consist_loss', 0):.4f}"
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
    
    print("\n✅ Unified training complete!")
    print(f"   Best model saved: {config['checkpoints']['output_dir']}/best_model.pt")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    config_path = "experiments/exp3_unified/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    train_unified(config_path)