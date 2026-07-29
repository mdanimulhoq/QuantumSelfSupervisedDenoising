"""Train HN-E head for Experiment 2 (TDD §4.2, §6.4).

Trains the HN-E (Hardware-Noise Extrapolation) head on noise-scaled pairs.
Loads SN-D checkpoint from Experiment 1 and freezes it.
Uses Phase 2 curriculum: Joint training (SN-D frozen, HN-E train).
"""

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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.n2ln import N2LNQEM
from src.training.trainer import N2LNTrainer
from src.utils.seeding import set_seed
from src.utils.device import get_device


# =============================================================================
# Dataset Class for HN-E
# =============================================================================

class HNEDataset(Dataset):
    """Dataset for HN-E training with noise-scaled pairs."""
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.data = []
        
        with h5py.File(h5_path, 'r') as f:
            n_samples = len(f['n_qubits'])
            noise_scales = json.loads(f.attrs['noise_scales'])
            
            # Use λ=3.0 as high noise, λ=1.0 as low noise (target)
            self.high_scale = max(noise_scales)
            self.low_scale = min(noise_scales)
            
            for i in range(n_samples):
                # Load high-noise data (λ=3.0)
                high_bs = f[f'bitstrings_{self.high_scale}'][i]
                high_shape = f[f'bitstrings_{self.high_scale}_shape'][i]
                high_bs = high_bs.reshape(high_shape)
                high_probs = f[f'probs_{self.high_scale}'][i]
                high_probs = high_probs.astype(np.float32)
                
                # Load low-noise data (λ=1.0) - target
                low_bs = f[f'bitstrings_{self.low_scale}'][i]
                low_shape = f[f'bitstrings_{self.low_scale}_shape'][i]
                low_bs = low_bs.reshape(low_shape)
                low_probs = f[f'probs_{self.low_scale}'][i]
                low_probs = low_probs.astype(np.float32)
                
                self.data.append({
                    'high_bitstrings': torch.tensor(high_bs, dtype=torch.long),
                    'high_probs': torch.tensor(high_probs, dtype=torch.float32).unsqueeze(1),
                    'low_bitstrings': torch.tensor(low_bs, dtype=torch.long),
                    'low_probs': torch.tensor(low_probs, dtype=torch.float32),
                    'n_qubits': f['n_qubits'][i],
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'bitstrings': item['high_bitstrings'],
            'counts': item['high_probs'],
            'target_sn': item['low_probs'],  # Not used (SN-D frozen)
            'target_hn': item['low_probs'],
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
        
        target = b['target_hn']
        pad_len_target = max_bs_target - target.shape[0]
        if pad_len_target > 0:
            target_pad = torch.zeros((pad_len_target,), dtype=target.dtype)
            target = torch.cat([target, target_pad], dim=0)
        batched_target_sn.append(target)  # Not used
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
    
    # Load SN-D checkpoint (Step 4.2)
    sn_checkpoint_path = Path('checkpoints/exp1_snd/best_model.pt')
    if sn_checkpoint_path.exists():
        print(f"Loading SN-D checkpoint from: {sn_checkpoint_path}")
        checkpoint = torch.load(sn_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Freeze SN-D head and encoder/transformer
        for name, param in model.named_parameters():
            if 'sn_head' in name or 'encoder' in name or 'transformer' in name:
                param.requires_grad = False
        print("   ✅ SN-D head, encoder, and transformer frozen")
    else:
        print(f"⚠️ SN-D checkpoint not found: {sn_checkpoint_path}")
        print("   Training HN-E from scratch")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss weights (Phase 2: HN-E only)
    loss_weights = {
        'kl': config['loss']['alpha'],
        'tvd': config['loss']['beta'],
        'chi2': config['loss']['gamma'],
        'physicality': config['loss']['physicality'],
        'consistency': 0.0,  # Not used in Phase 2
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
        # Train HN-E only
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
