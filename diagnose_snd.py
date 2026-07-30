"""
Diagnostic script for SN-D model (TDD §6.4 Exp 1).

Measures:
- sn_logits_std: Standard deviation of logits (should be > 1.0 for sharp predictions)
- sn_entropy: Entropy of output distribution (should be < log(M) for non-uniform)
- tvd_high: TVD computed only on high-shot support (fair metric)
- tvd_union: TVD computed on union support (inflated metric)
- max_pred: Maximum probability in prediction
- max_tgt: Maximum probability in target
- support_high_frac: Fraction of high-shot support in union
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import h5py
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Import project modules
from src.models.n2ln import N2LNQEM
from src.losses.distribution import kl_loss, tvd_loss
from experiments.exp1_snd.evaluate import SNDDataset, collate_snd_batch


def diagnose():
    """Run diagnosis on SN-D model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt_path = "checkpoints/exp1_snd/best_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("   Please run Step 4.2 first.")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[OK] Loaded checkpoint from: {ckpt_path}")
    
    # Load config
    with open("experiments/exp1_snd/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
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
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print("[OK] Model loaded")
    
    # Load test data
    test_path = "data/raw/exp1_snd/exp1_snd_test.h5"
    if not os.path.exists(test_path):
        print(f"[ERROR] Test data not found: {test_path}")
        return
    
    ds = SNDDataset(test_path, n_qubits=8)
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_snd_batch,
    )
    print(f"[OK] Test samples: {len(ds)}")
    
    # Metrics
    metrics = {
        "sn_logits_std": [],
        "sn_entropy": [],
        "tvd_high": [],
        "tvd_union": [],
        "max_pred": [],
        "max_tgt": [],
        "support_high_frac": [],
    }
    
    print("\n[RUN] Running diagnosis...")
    print("=" * 60)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            bs = batch["bitstrings"].to(device)
            cnt = batch["counts"].to(device)
            tgt = batch["target_sn"].to(device)
            
            # Get model outputs
            embeddings = model.encoder(bs)
            z = model.transformer(embeddings, cnt)
            
            # Get SN-D logits (replicate decoder scoring)
            sn_z = model.decoder.sn_head(z).unsqueeze(1)  # (B, 1, d)
            bs_emb = model.encoder(bs)  # (B, M, d)
            sn_logits = (sn_z * bs_emb).sum(dim=-1) / model.decoder.temperature  # (B, M)
            
            # Get SN-D distribution
            sn_out, _ = model.decoder(z, bs)
            
            # --- Compute metrics ---
            
            # 1. Logits standard deviation
            logits_std = sn_logits.std(dim=-1).mean().item()
            metrics["sn_logits_std"].append(logits_std)
            
            # 2. Entropy
            entropy = -(sn_out * torch.log(sn_out + 1e-8)).sum(dim=-1).mean().item()
            metrics["sn_entropy"].append(entropy)
            
            # 3. TVD on union support
            tvd_union = 0.5 * torch.abs(sn_out - tgt).sum(dim=-1).mean().item()
            metrics["tvd_union"].append(tvd_union)
            
            # 4. TVD on high-support only
            mask = (tgt > 1e-6)
            if mask.any():
                sn_m = sn_out * mask.float()
                tg_m = tgt * mask.float()
                # Renormalize
                sn_m = sn_m / sn_m.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                tg_m = tg_m / tg_m.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                tvd_high = 0.5 * torch.abs(sn_m - tg_m).sum(dim=-1).mean().item()
                metrics["tvd_high"].append(tvd_high)
            else:
                metrics["tvd_high"].append(tvd_union)
            
            # 5. Max prediction vs max target
            metrics["max_pred"].append(sn_out.max(dim=-1).values.mean().item())
            metrics["max_tgt"].append(tgt.max(dim=-1).values.mean().item())
            
            # 6. Support fraction
            high_card = (tgt > 1e-6).sum(dim=-1).float().mean().item()
            union_card = bs.shape[1]
            metrics["support_high_frac"].append(high_card / union_card)
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    
    # Print results
    print("\n[RESULTS] DIAGNOSIS RESULTS:")
    print("=" * 60)
    print(f"  sn_logits_std:        {avg_metrics['sn_logits_std']:.4f}  (should be > 1.0 for sharp predictions)")
    print(f"  sn_entropy:           {avg_metrics['sn_entropy']:.4f}  (should be < log(M) for non-uniform)")
    print(f"  tvd_high (fair):      {avg_metrics['tvd_high']:.4f}  (raw will be higher)")
    print(f"  tvd_union (inflated): {avg_metrics['tvd_union']:.4f}")
    print(f"  max_pred:             {avg_metrics['max_pred']:.4f}")
    print(f"  max_tgt:              {avg_metrics['max_tgt']:.4f}  (should be > max_pred for sharp peaks)")
    print(f"  support_high_frac:    {avg_metrics['support_high_frac']:.4f}")
    print("=" * 60)
    
    # Diagnosis
    print("\n[INTERPRETATION]")
    print("-" * 60)
    
    if avg_metrics['sn_logits_std'] < 0.5:
        print("[WARN] logits_std < 0.5 -> output is too smooth. Temperature or architecture issue.")
    else:
        print("[OK] logits_std > 0.5 -> model has some sharpness.")
    
    if avg_metrics['tvd_high'] < avg_metrics['tvd_union']:
        print(f"[OK] tvd_high ({avg_metrics['tvd_high']:.4f}) < tvd_union ({avg_metrics['tvd_union']:.4f})")
        print("     -> Union support inflates TVD. Use high-support for fair evaluation.")
    else:
        print("[INFO] tvd_high approx tvd_union -> union inflation not significant.")
    
    if avg_metrics['max_pred'] < avg_metrics['max_tgt']:
        print(f"[WARN] max_pred ({avg_metrics['max_pred']:.4f}) < max_tgt ({avg_metrics['max_tgt']:.4f})")
        print("     -> Model is under-peaked. Need sharpness loss or MLP scorer.")
    else:
        print(f"[OK] max_pred ({avg_metrics['max_pred']:.4f}) >= max_tgt ({avg_metrics['max_tgt']:.4f})")
        print("     -> Model peaks are sharp enough.")
    
    # Success check
    print("\n" + "=" * 60)
    print("[TARGET] TDD SUCCESS CRITERION CHECK:")
    raw_tvd = avg_metrics['tvd_union']  # Using union for comparison with reported results
    sn_tvd = avg_metrics['tvd_high']    # Using high for fair comparison
    threshold = 0.5 * raw_tvd
    
    print(f"  Raw TVD (union):    {raw_tvd:.4f}")
    print(f"  SN-D TVD (high):    {sn_tvd:.4f}")
    print(f"  Target (50%% raw):   {threshold:.4f}")
    
    if sn_tvd <= threshold:
        print("  [PASS] SUCCESS CRITERION MET! SN-D TVD <= 50%% of raw TVD")
    else:
        print(f"  [FAIL] NOT MET: SN-D TVD ({sn_tvd:.4f}) > 50%% raw ({threshold:.4f})")
        print(f"     Improvement needed: {((sn_tvd - threshold) / threshold * 100):.1f}%%")
    
    # Recommendations
    print("\n[RECOMMENDATIONS]")
    if avg_metrics['sn_logits_std'] < 0.5:
        print("  1. Reduce temperature to 0.5 or 0.7")
    if avg_metrics['max_pred'] < avg_metrics['max_tgt']:
        print("  2. Add sharpness loss to trainer")
        print("  3. Consider MLP scorer instead of dot-product")
    if avg_metrics['tvd_high'] < avg_metrics['tvd_union']:
        print("  4. Use high-support TVD for evaluation (Step A in fix plan)")
    
    print("\n[OK] Diagnosis complete!")


if __name__ == "__main__":
    diagnose()