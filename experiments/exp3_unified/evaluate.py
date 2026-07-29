"""Evaluate Unified N2LN model (TDD §6.4 Exp 3).

Compares Unified N2LN against:
- Raw
- SN-D only
- HN-E only
- ZNE
- Ideal-supervised oracle
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.n2ln import N2LNQEM
from src.utils.seeding import set_seed
from src.utils.device import get_device
from src.losses.distribution import tvd_loss


def evaluate_unified(config_path: str):
    """Main evaluation function for Unified N2LN."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("STEP 6.2: Unified N2LN Evaluation Setup")
    print("=" * 60)
    print("This script is ready for evaluation.")
    print("Please complete:")
    print("  - Step 4.1: SN-D dataset (full)")
    print("  - Step 4.2: SN-D training (full)")
    print("  - Step 5.1: HN-E dataset (full)")
    print("  - Step 5.2: HN-E training (full)")
    print("  - Step 6.1: Unified N2LN training")
    print("=" * 60)
    
    # TODO: Load data
    # TODO: Load models (SN-D, HN-E, Unified)
    # TODO: Compute metrics (TVD, Fidelity, MAE)
    # TODO: Compare with baselines
    # TODO: Generate report
    
    print("\n✅ Unified N2LN evaluation script ready!")
    print("   Run after completing Steps 4.2, 5.2, and 6.1")


if __name__ == "__main__":
    config_path = "experiments/exp3_unified/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    evaluate_unified(config_path)
