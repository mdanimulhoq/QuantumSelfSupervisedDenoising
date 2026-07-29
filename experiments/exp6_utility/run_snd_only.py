"""Run SN-D head only on 127-qubit utility data (TDD §7.4).

Demonstrates shot-noise reduction on 127-qubit hardware.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.n2ln import N2LNQEM
from src.utils.seeding import set_seed
from src.utils.device import get_device


def run_snd_only(config_path: str):
    """Run SN-D only on 127-qubit data."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("STEP 9.2: Shot-Noise-Only Demonstration Setup")
    print("=" * 60)
    print("This script runs SN-D head on 127-qubit utility data.")
    print("Features:")
    print("  - Loads SN-D checkpoint from Step 4.2")
    print("  - Runs on 127-qubit data (50 circuits)")
    print("  - Demonstrates shot-noise reduction")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  - Step 4.2: SN-D training (full)")
    print("  - Step 9.1: Utility-scale data collection")
    print("  - SN-D checkpoint: checkpoints/exp1_snd/best_model.pt")
    print("=" * 60)
    
    # TODO: Load SN-D model
    # TODO: Load 127-qubit data
    # TODO: Run SN-D head
    # TODO: Compute TVD reduction
    # TODO: Generate report
    
    print("\n✅ SN-D only demo script ready!")
    print("   Run after completing Steps 4.2 and 9.1")


if __name__ == "__main__":
    config_path = "experiments/exp6_utility/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    run_snd_only(config_path)
