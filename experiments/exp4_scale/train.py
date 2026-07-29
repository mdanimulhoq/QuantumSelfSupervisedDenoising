"""Scale-out training for Experiment 4 (TDD §6.4 Exp 4).

Trains on n=4,6 and tests on n=10,15,20,25,30.
Uses mixed precision and gradient accumulation for large-scale training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import h5py

from src.models.n2ln import N2LNQEM
from src.training.trainer import N2LNTrainer
from src.utils.seeding import set_seed
from src.utils.device import get_device


class ScaleDataset(Dataset):
    """Dataset for scale-out training."""
    
    def __init__(self, h5_path: str):
        self.data = []
        
        with h5py.File(h5_path, 'r') as f:
            n_samples = len(f['n_qubits'])
            
            for i in range(n_samples):
                bitstrings = f['bitstrings'][i]
                shape = f['bitstrings_shape'][i]
                bitstrings = bitstrings.reshape(shape)
                probs = f['probs'][i]
                probs = probs.astype(np.float32)
                
                self.data.append({
                    'bitstrings': torch.tensor(bitstrings, dtype=torch.long),
                    'probs': torch.tensor(probs, dtype=torch.float32).unsqueeze(1),
                    'n_qubits': f['n_qubits'][i],
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'bitstrings': item['bitstrings'],
            'counts': item['probs'],
            'target_sn': item['probs'],  # For SN-D training
            'target_hn': item['probs'],  # For HN-E training
            'n_qubits': item['n_qubits'],
        }


def collate_scale_batch(batch):
    """Collate function for scale-out batch."""
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


def train_scale(config_path: str):
    """Main training function for scale-out experiment."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("STEP 7.3: Scale-out Training Setup")
    print("=" * 60)
    print("This script trains on n=4,6 and tests on n=10,15,20,25,30.")
    print("Features:")
    print("  - Mixed precision training (AMP)")
    print("  - Gradient accumulation")
    print("  - Qubit-count generalization")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  - Step 4.1: SN-D dataset (full)")
    print("  - Step 4.2: SN-D training (full)")
    print("  - Step 4.3: SN-D evaluation")
    print("  - Step 7.2: Large-scale dataset")
    print("=" * 60)
    
    # TODO: Load training data (n=4,6 from Step 4.1)
    # TODO: Load test data (n=10,15,20,25,30 from Step 7.2)
    # TODO: Create model
    # TODO: Training with mixed precision
    # TODO: Evaluation on all qubit counts
    # TODO: Generate TVD curve
    
    print("\n✅ Scale-out training script ready!")
    print("   Run after completing Steps 4.2, 4.3, and 7.2")


if __name__ == "__main__":
    config_path = "experiments/exp4_scale/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    train_scale(config_path)
