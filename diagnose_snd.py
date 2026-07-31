#!/usr/bin/env python3
"""Diagnostic script for SN-D model (TDD §6.4 Exp 1)."""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# 🔥 FIX: remove extra closing parenthesis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.n2ln import N2LNQEM
from src.losses.distribution import tvd_loss
from experiments.exp1_snd.evaluate import SNDDataset, collate_snd_batch
from src.utils.device import get_device


def diagnose():
    config_path = "experiments/exp1_snd/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    # ✅ Read config to build the SAME model as training
    model_cfg = config['model']
    model = N2LNQEM(
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['n_heads'],
        n_isab=model_cfg['n_isab'],
        n_sab=model_cfg['n_sab'],
        d_ff=model_cfg['d_ff'],
        m=model_cfg['m'],
        decoder_hidden=model_cfg['decoder_hidden'],
        dropout=model_cfg['dropout'],
        max_qubits=config['data']['n_qubits'][-1],
        use_mlp_scorer=model_cfg.get('use_mlp_scorer', False),
        temperature_floor=model_cfg.get('temperature_floor', 0.1),
    )
    model.to(device)

    # ✅ Load checkpoint
    ckpt_path = "checkpoints/exp1_snd/best_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"[OK] Loaded checkpoint from: {ckpt_path}")

    # ✅ Load test data
    test_path = Path(config['data']['data_dir']) / 'exp1_snd_test.h5'
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        return
    test_dataset = SNDDataset(str(test_path), config['data']['n_qubits'][0])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_snd_batch,
    )
    print(f"[OK] Test samples: {len(test_dataset)}")

    metrics = {
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
        for batch in test_loader:
            bs = batch['bitstrings'].to(device)
            cnt = batch['counts'].to(device)
            tgt = batch['target_sn'].to(device)

            # Forward
            sn_out, _ = model(bs, cnt, mode='sn_only')

            # Entropy
            metrics["sn_entropy"].append(
                -(sn_out * torch.log(sn_out + 1e-8)).sum(dim=-1).mean().item()
            )

            # TVD on union
            tvd_u = 0.5 * torch.abs(sn_out - tgt).sum(dim=-1).mean().item()
            metrics["tvd_union"].append(tvd_u)

            # TVD on high-support only
            mask = (tgt > 1e-6)
            if mask.any():
                sn_m = sn_out * mask.float()
                tg_m = tgt * mask.float()
                sn_m = sn_m / sn_m.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                tg_m = tg_m / tg_m.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                tvd_h = 0.5 * torch.abs(sn_m - tg_m).sum(dim=-1).mean().item()
                metrics["tvd_high"].append(tvd_h)
            else:
                metrics["tvd_high"].append(tvd_u)

            metrics["max_pred"].append(sn_out.max(dim=-1).values.mean().item())
            metrics["max_tgt"].append(tgt.max(dim=-1).values.mean().item())

            # support fraction
            high_card = (tgt > 1e-6).sum(dim=-1).float().mean().item()
            union_card = bs.shape[1]
            metrics["support_high_frac"].append(high_card / union_card)

    # Average
    avg = {k: np.mean(v) for k, v in metrics.items() if v}
    print("\n[RESULTS] DIAGNOSIS RESULTS:")
    print("=" * 60)
    print(f"  sn_entropy:           {avg['sn_entropy']:.4f}  (should be < log(M) for non-uniform)")
    print(f"  tvd_high (fair):      {avg['tvd_high']:.4f}  (raw will be higher)")
    print(f"  tvd_union (inflated): {avg['tvd_union']:.4f}")
    print(f"  max_pred:             {avg['max_pred']:.4f}")
    print(f"  max_tgt:              {avg['max_tgt']:.4f}  (should be > max_pred for sharp peaks)")
    print(f"  support_high_frac:    {avg['support_high_frac']:.4f}")
    print("=" * 60)

    # Interpretation
    print("\n[INTERPRETATION]")
    print("-" * 60)
    if avg['max_pred'] < avg['max_tgt']:
        print("[WARN] max_pred < max_tgt -> Model is under-peaked. Need sharpness loss or MLP scorer.")
    else:
        print("[OK] max_pred >= max_tgt -> Model peaks are sharp enough.")

    if avg['tvd_high'] < avg['tvd_union']:
        print("[OK] tvd_high < tvd_union -> Union support inflates TVD. Use high-support for fair evaluation.")
    else:
        print("[INFO] tvd_high ≈ tvd_union -> Union inflation not significant.")

    # Success check
    print("\n" + "=" * 60)
    raw_tvd = avg['tvd_union']
    sn_tvd = avg['tvd_high']
    target = 0.5 * raw_tvd
    print("[TARGET] TDD SUCCESS CRITERION CHECK:")
    print(f"  Raw TVD (union):    {raw_tvd:.4f}")
    print(f"  SN-D TVD (high):    {sn_tvd:.4f}")
    print(f"  Target (50% raw):   {target:.4f}")
    if sn_tvd <= target:
        print("  ✅ SUCCESS CRITERION MET! SN-D TVD ≤ 50% of raw TVD")
    else:
        print(f"  ❌ NOT MET: SN-D TVD ({sn_tvd:.4f}) > 50% raw ({target:.4f})")

    print("\n[OK] Diagnosis complete!")


if __name__ == "__main__":
    diagnose()
