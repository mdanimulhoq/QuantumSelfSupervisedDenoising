"""Run ablation studies (TDD §6.4 Exp 6).

Compares full N2LN against variants with components removed:
1. No positional encoding
2. No count weighting
3. No consistency loss
4. No curriculum learning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils.seeding import set_seed
from src.utils.device import get_device


def run_ablation(config_path: str):
    """Run all ablation experiments."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("STEP 10.1: Ablation Grid Setup")
    print("=" * 60)
    print("Ablations to run:")
    
    for key, ablation in config['ablations'].items():
        print(f"  - {ablation['name']}: {ablation['description']}")
    
    print("=" * 60)
    print("\nPrerequisites:")
    print("  - Step 4.1: SN-D dataset (full)")
    print("  - Step 4.2: SN-D training (full)")
    print("  - Step 5.1: HN-E dataset (full)")
    print("  - Step 5.2: HN-E training (full)")
    print("  - Step 6.1: Unified N2LN training")
    print("=" * 60)
    
    # TODO: For each ablation:
    # 1. Modify model/config
    # 2. Train model
    # 3. Evaluate
    # 4. Store results
    
    print("\n✅ Ablation script ready!")
    print("   Results will be saved to: experiments/exp7_ablation/results/")
    print("   Run after completing Steps 4.2, 5.2, and 6.1")


if __name__ == "__main__":
    config_path = "experiments/exp7_ablation/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    run_ablation(config_path)
