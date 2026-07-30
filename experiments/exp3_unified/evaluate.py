"""Evaluate Unified N2LN model (TDD §6.4 Exp 3).

Compares Unified N2LN against:
- Raw
- SN-D only
- HN-E only
- ZNE
- Ideal-supervised oracle (if available)
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import h5py
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.n2ln import N2LNQEM
from src.utils.seeding import set_seed
from src.utils.device import get_device


# =============================================================================
# Helper: Align support
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
# Dataset Class
# =============================================================================

class EvalDataset(Dataset):
    """Dataset for evaluation."""
    
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
                    
                    low_probs = f['low_probs'][i]
                    low_probs = low_probs.astype(np.float32)
                    
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
            'target_hn': target_sn.clone(),
            'n_qubits': item['n_qubits'],
        }


def collate_eval_batch(batch):
    """Collate function for evaluation."""
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
# Metrics
# =============================================================================

def compute_tvd(pred, target, eps=1e-8):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return 0.5 * torch.abs(pred - target).sum(dim=-1).mean().item()


def compute_fidelity(pred, target, eps=1e-8):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (torch.sqrt(pred * target + eps).sum(dim=-1) ** 2).mean().item()


def compute_mae(pred, target, eps=1e-8):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return torch.abs(pred - target).mean().item()


def compute_metrics(pred, target):
    return {
        'tvd': compute_tvd(pred, target),
        'fidelity': compute_fidelity(pred, target),
        'mae': compute_mae(pred, target),
    }


# =============================================================================
# Model Loader
# =============================================================================

def load_model(checkpoint_path, device, max_qubits=8):
    """Load a model from checkpoint."""
    model = N2LNQEM(
        d_model=64,
        n_heads=4,
        n_isab=2,
        n_sab=1,
        d_ff=256,
        m=16,
        decoder_hidden=128,
        dropout=0.1,
        max_qubits=max_qubits,
    )
    model.to(device)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        return model
    return None


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_unified(config_path: str):
    """Main evaluation function for Unified N2LN."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    # Data paths
    data_dir = Path(config['data']['snd_data_dir'])
    test_path = data_dir / 'exp1_snd_test.h5'
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    # Dataset
    test_dataset = EvalDataset(str(test_path), config['data']['n_qubits'][0])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_eval_batch,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load models
    models = {}
    
    # SN-D
    sn_path = Path('checkpoints/exp1_snd/best_model.pt')
    if sn_path.exists():
        models['sn_d'] = load_model(sn_path, device)
        print("✅ SN-D model loaded")
    
    # HN-E
    hne_path = Path('checkpoints/exp2_hne/best_model.pt')
    if hne_path.exists():
        models['hn_e'] = load_model(hne_path, device)
        print("✅ HN-E model loaded")
    
    # Unified
    unified_path = Path('checkpoints/exp3_unified/best_model.pt')
    if unified_path.exists():
        models['unified'] = load_model(unified_path, device)
        print("✅ Unified model loaded")
    
    # ZNE results (load from previous run)
    zne_results = None
    zne_file = Path('experiments/exp2_hne/baselines/zne_results.json')
    if zne_file.exists():
        with open(zne_file, 'r') as f:
            zne_results = json.load(f)
        print("✅ ZNE results loaded")
    
    # Evaluate
    results = {
        'raw': [],
        'sn_d': [],
        'hn_e': [],
        'unified': [],
    }
    
    print("\nEvaluating on test set...")
    print("=" * 60)
    
    with torch.no_grad():
        for batch in test_loader:
            bitstrings = batch['bitstrings'].to(device)
            counts = batch['counts'].to(device)
            target = batch['target_sn'].to(device)
            
            # Raw
            raw_dist = counts.squeeze(-1)
            results['raw'].append(compute_metrics(raw_dist, target))
            
            # SN-D
            if 'sn_d' in models:
                sn_out, _ = models['sn_d'](bitstrings, counts, mode='sn_only')
                results['sn_d'].append(compute_metrics(sn_out, target))
            
            # HN-E
            if 'hn_e' in models:
                _, hn_out = models['hn_e'](bitstrings, counts, mode='hn_only')
                results['hn_e'].append(compute_metrics(hn_out, target))
            
            # Unified
            if 'unified' in models:
                unified_out, _ = models['unified'](bitstrings, counts, mode='unified')
                results['unified'].append(compute_metrics(unified_out, target))
    
    # Average results
    avg_results = {}
    for key, values in results.items():
        if values:
            avg_metrics = {}
            for metric in values[0].keys():
                avg_metrics[metric] = np.mean([v[metric] for v in values])
            avg_results[key] = avg_metrics
    
    # Print results
    print("\n📊 Results:")
    print("-" * 70)
    print(f"{'Method':<15} {'TVD':<12} {'Fidelity':<12} {'MAE':<12}")
    print("-" * 70)
    
    for method in ['raw', 'sn_d', 'hn_e', 'unified']:
        if method in avg_results:
            r = avg_results[method]
            print(f"{method.upper():<15} {r['tvd']:<12.4f} {r['fidelity']:<12.4f} {r['mae']:<12.4f}")
    
    if zne_results:
        print(f"{'ZNE':<15} {zne_results['zne_tvd']:<12.4f} {'-':<12} {'-':<12}")
    
    print("-" * 70)
    
    # Compute improvements
    if 'raw' in avg_results and 'unified' in avg_results:
        raw_tvd = avg_results['raw']['tvd']
        unified_tvd = avg_results['unified']['tvd']
        improvement = ((raw_tvd - unified_tvd) / raw_tvd * 100) if raw_tvd > 0 else 0
        print(f"\n📈 Unified TVD Improvement vs Raw: {improvement:.1f}%")
        
        if improvement >= 40:
            print("✅ Success criterion met: Unified TVD Improvement >= 40%")
        else:
            print(f"⚠️ Success criterion not met: Unified TVD Improvement = {improvement:.1f}% < 40%")
    
    # Save results
    output_dir = Path('experiments/exp3_unified')
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(avg_results, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved: {output_dir / 'eval_results.json'}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    config_path = "experiments/exp3_unified/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    evaluate_unified(config_path)