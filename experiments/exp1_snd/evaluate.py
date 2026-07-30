"""Evaluate SN-D model for Experiment 1 (TDD §6.2, §6.4).

Computes TVD, Fidelity, MAE for SN-D output vs raw low-shot vs high-shot.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.n2ln import N2LNQEM
from src.training.trainer import N2LNTrainer
from src.utils.seeding import set_seed
from src.utils.device import get_device
from src.losses.distribution import tvd_loss, kl_loss

# =============================================================================
# Helper: Align low and high distributions to union support
# =============================================================================

def align_support(low_bs, low_probs, high_bs, high_probs):
    """Align low and high distributions to a union support.

    Args:
        low_bs: (M1, n) bitstrings from low-shot measurement
        low_probs: (M1, 1) probabilities from low-shot measurement
        high_bs: (M2, n) bitstrings from high-shot measurement
        high_probs: (M2,) probabilities from high-shot measurement

    Returns:
        union_bs: (M, n) sorted union of all bitstrings
        low_aligned: (M, 1) probabilities aligned to union support
        high_aligned: (M,) probabilities aligned to union support
    """
    def to_key(row):
        return tuple(int(x) for x in row.tolist())

    support = {}

    # Low shot entries
    for bs, p in zip(low_bs, low_probs.squeeze(-1)):
        key = to_key(bs)
        if key not in support:
            support[key] = [0.0, 0.0]
        support[key][0] = float(p)

    # High shot entries
    for bs, p in zip(high_bs, high_probs):
        key = to_key(bs)
        if key not in support:
            support[key] = [0.0, 0.0]
        support[key][1] = float(p)

    # Deterministic ordering for reproducibility
    keys = sorted(support.keys())

    union_bs = torch.tensor(keys, dtype=torch.long)
    low_aligned = torch.tensor([support[k][0] for k in keys], dtype=torch.float32)
    high_aligned = torch.tensor([support[k][1] for k in keys], dtype=torch.float32)

    # Normalize (ensure sums to 1)
    low_aligned = low_aligned / low_aligned.sum().clamp_min(1e-8)
    high_aligned = high_aligned / high_aligned.sum().clamp_min(1e-8)

    return union_bs, low_aligned.unsqueeze(1), high_aligned


# =============================================================================
# Dataset Class (same as train.py)
# =============================================================================

class SNDDataset(Dataset):
    """Dataset for SN-D evaluation."""
    
    def __init__(self, h5_path: str, n_qubits: int):
        self.h5_path = h5_path
        self.n_qubits = n_qubits
        self.data = []
        
        with h5py.File(h5_path, 'r') as f:
            n_samples = len(f['n_qubits'])
            
            for i in range(n_samples):
                try:
                    # Read the actual number of qubits for this sample
                    actual_n = f['n_qubits'][i]
                    
                    # Load low-bitstrings - flatten and reshape using actual_n
                    low_bs_flat = f['low_bitstrings'][i]
                    low_shape = f['low_bitstrings_shape'][i]
                    if len(low_shape) == 2 and low_bs_flat.size == low_shape[0] * low_shape[1]:
                        low_bs = low_bs_flat.reshape(low_shape[0], low_shape[1])
                    else:
                        low_bs = low_bs_flat.reshape(-1, actual_n)
                    
                    # Load low-probs
                    low_probs = f['low_probs'][i]
                    low_probs = low_probs.astype(np.float32)
                    
                    # Load high-bitstrings - flatten and reshape using actual_n
                    high_bs_flat = f['high_bitstrings'][i]
                    high_shape = f['high_bitstrings_shape'][i]
                    if len(high_shape) == 2 and high_bs_flat.size == high_shape[0] * high_shape[1]:
                        high_bs = high_bs_flat.reshape(high_shape[0], high_shape[1])
                    else:
                        high_bs = high_bs_flat.reshape(-1, actual_n)
                    
                    # Load high-probs
                    high_probs = f['high_probs'][i]
                    high_probs = high_probs.astype(np.float32)
                    
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

        # 🔥 FIX: Align low and high to union support
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
    """Collate function for SN-D batch."""
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
# Evaluation Functions - 🔥 FIXED: No truncation, proper normalization
# =============================================================================

def compute_tvd(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute TVD between two distributions."""
    # Ensure same shape
    if pred.shape != target.shape:
        # This should not happen after alignment, but if it does, raise error
        raise ValueError(f"Shape mismatch in compute_tvd: pred={pred.shape}, target={target.shape}")

    # Renormalize to handle any numerical drift
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    return 0.5 * torch.abs(pred - target).sum(dim=-1).mean().item()


def compute_fidelity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute fidelity between two distributions."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in compute_fidelity: pred={pred.shape}, target={target.shape}")

    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    fidelity = (torch.sqrt(pred * target + eps).sum(dim=-1) ** 2).mean().item()
    return fidelity


def compute_mae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute MAE between two distributions."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in compute_mae: pred={pred.shape}, target={target.shape}")

    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    return torch.abs(pred - target).mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute all metrics."""
    return {
        'tvd': compute_tvd(pred, target),
        'fidelity': compute_fidelity(pred, target),
        'mae': compute_mae(pred, target),
    }


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_snd(config_path: str):
    """Main evaluation function for SN-D experiment."""
    
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
    test_path = data_dir / 'exp1_snd_test.h5'
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    # Create dataset
    test_dataset = SNDDataset(str(test_path), config['data']['n_qubits'][0])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_snd_batch,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
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
        max_qubits=config['data']['n_qubits'][-1],
    )
    model.to(device)
    
    # Load best model
    checkpoint_path = Path('checkpoints/exp1_snd/best_model.pt')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")
    
    # Evaluate
    all_metrics = {
        'raw_vs_high': [],
        'sn_vs_high': [],
        'sn_vs_ideal': [],
    }
    
    print("\nEvaluating on test set...")
    print("=" * 60)
    
    with torch.no_grad():
        for batch in test_loader:
            bitstrings = batch['bitstrings'].to(device)
            counts = batch['counts'].to(device)
            target_high = batch['target_sn'].to(device)
            
            # Raw low-shot (input)
            low_dist = counts.squeeze(-1)  # (B, M)
            
            # SN-D output
            sn_out, _ = model(bitstrings, counts, mode='sn_only')
            
            # High-shot (target)
            high_dist = target_high
            
            # Compute metrics
            raw_vs_high = compute_metrics(low_dist, high_dist)
            sn_vs_high = compute_metrics(sn_out, high_dist)
            
            all_metrics['raw_vs_high'].append(raw_vs_high)
            all_metrics['sn_vs_high'].append(sn_vs_high)
    
    # Average metrics
    results = {}
    for key in all_metrics:
        if all_metrics[key]:
            avg_metrics = {}
            for metric in all_metrics[key][0].keys():
                values = [m[metric] for m in all_metrics[key]]
                avg_metrics[metric] = np.mean(values)
            results[key] = avg_metrics
        else:
            results[key] = {}
    
    # Save metrics
    output_dir = Path('experiments/exp1_snd')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n📊 Results:")
    print("-" * 60)
    print(f"{'Metric':<20} {'Raw vs High':<15} {'SN-D vs High':<15}")
    print("-" * 60)
    for metric in ['tvd', 'fidelity', 'mae']:
        raw_val = results['raw_vs_high'][metric]
        sn_val = results['sn_vs_high'][metric]
        print(f"{metric:<20} {raw_val:<15.4f} {sn_val:<15.4f}")
    print("-" * 60)
    
    # Compute improvement
    if results['raw_vs_high']['tvd'] > 0:
        tvd_improvement = (results['raw_vs_high']['tvd'] - results['sn_vs_high']['tvd']) / results['raw_vs_high']['tvd'] * 100
    else:
        tvd_improvement = 0
    print(f"\n📈 TVD Improvement: {tvd_improvement:.1f}%")
    
    # Check success criterion (TDD §1.3)
    if results['sn_vs_high']['tvd'] <= 0.5 * results['raw_vs_high']['tvd']:
        print("✅ Success criterion met: SN-D TVD <= 50% of raw TVD")
    else:
        print(f"⚠️ Success criterion not met: SN-D TVD ({results['sn_vs_high']['tvd']:.4f}) > 50% of raw ({results['raw_vs_high']['tvd']:.4f})")
    
    # Generate report
    generate_report(results, output_dir)
    generate_plots(results, output_dir, test_loader, model, device)
    
    print("\n✅ Evaluation complete!")
    print(f"   Report saved: {output_dir / 'REPORT.md'}")
    print(f"   Metrics saved: {output_dir / 'metrics.json'}")
    print(f"   Plots saved: {output_dir / 'plots/'}")


def generate_report(results: Dict, output_dir: Path):
    """Generate REPORT.md."""
    raw_tvd = results['raw_vs_high']['tvd']
    sn_tvd = results['sn_vs_high']['tvd']
    improvement = ((raw_tvd - sn_tvd) / raw_tvd * 100) if raw_tvd > 0 else 0
    
    report = f"""# Experiment 1: SN-D Validation Report

## Summary
- **Model**: SN-D (Shot-Noise Denoising)
- **Qubits**: 4, 6, 8
- **Low shots**: 100
- **High shots**: 100000

## Results

### TVD Comparison
| Metric | Raw vs High | SN-D vs High | Improvement |
|--------|-------------|--------------|-------------|
| TVD | {results['raw_vs_high']['tvd']:.4f} | {results['sn_vs_high']['tvd']:.4f} | {improvement:.1f}% |
| Fidelity | {results['raw_vs_high']['fidelity']:.4f} | {results['sn_vs_high']['fidelity']:.4f} | {((results['sn_vs_high']['fidelity'] - results['raw_vs_high']['fidelity']) / results['raw_vs_high']['fidelity'] * 100):.1f}% |
| MAE | {results['raw_vs_high']['mae']:.4f} | {results['sn_vs_high']['mae']:.4f} | {((results['raw_vs_high']['mae'] - results['sn_vs_high']['mae']) / results['raw_vs_high']['mae'] * 100):.1f}% |

## Success Criterion (TDD §1.3)
- **Goal**: SN-D TVD <= 50% of raw low-shot TVD
- **Result**: SN-D TVD = {sn_tvd:.4f} vs Raw TVD = {raw_tvd:.4f}
- **Status**: {'✅ PASSED' if sn_tvd <= 0.5 * raw_tvd else '❌ FAILED'}

## Plots
- `plots/tvd_comparison.png` - TVD comparison
- `plots/distribution_example.png` - Example distributions

## Files
- `config.yaml`: Experiment configuration
- `metrics.json`: Full metrics
- `best_model.pt`: Trained model

## Next Steps
Proceed to Experiment 2: HN-E Validation
"""
    
    with open(output_dir / 'REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)


def generate_plots(results: Dict, output_dir: Path, test_loader, model, device):
    """Generate plots for report."""
    plots_dir = output_dir / 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: TVD comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['tvd', 'fidelity', 'mae']
    raw_values = [results['raw_vs_high'][m] for m in metrics]
    sn_values = [results['sn_vs_high'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, raw_values, width, label='Raw vs High', color='red', alpha=0.7)
    ax.bar(x + width/2, sn_values, width, label='SN-D vs High', color='blue', alpha=0.7)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('SN-D Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'tvd_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: Example distribution
    with torch.no_grad():
        for batch in test_loader:
            bitstrings = batch['bitstrings'].to(device)
            counts = batch['counts'].to(device)
            target_high = batch['target_sn'].to(device)
            
            sn_out, _ = model(bitstrings, counts, mode='sn_only')
            
            # Take first sample
            low = counts[0, :].cpu().numpy().flatten()
            high = target_high[0, :].cpu().numpy().flatten()
            sn = sn_out[0, :].cpu().numpy().flatten()
            
            # Trim to min length
            min_len = min(len(low), len(high), len(sn))
            low = low[:min_len]
            high = high[:min_len]
            sn = sn[:min_len]
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            
            axes[0].bar(range(min_len), low, color='red', alpha=0.7)
            axes[0].set_title('Low-Shot Distribution (Raw)')
            axes[0].set_ylabel('Probability')
            
            axes[1].bar(range(min_len), high, color='green', alpha=0.7)
            axes[1].set_title('High-Shot Distribution (Target)')
            axes[1].set_ylabel('Probability')
            
            axes[2].bar(range(min_len), sn, color='blue', alpha=0.7)
            axes[2].set_title('SN-D Output')
            axes[2].set_xlabel('Bitstring Index')
            axes[2].set_ylabel('Probability')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'distribution_example.png', dpi=150)
            plt.close()
            break


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    config_path = "experiments/exp1_snd/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    evaluate_snd(config_path)