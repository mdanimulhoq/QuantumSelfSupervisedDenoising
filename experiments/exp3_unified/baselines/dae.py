"""Denoising Autoencoder (DAE) baseline for Experiment 3 (TDD §3.5).

Compares Set Transformer against simple MLP autoencoder.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.baseline_dae import DistributionAutoencoder
from src.utils.seeding import set_seed
from src.utils.device import get_device


class DAEBaseline:
    """DAE Baseline wrapper for Experiment 3."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        set_seed(self.config['seed'])
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # TODO: Create DAE model
        # TODO: Load data
        # TODO: Train DAE
        # TODO: Evaluate DAE
    
    def train(self):
        """Train DAE baseline."""
        print("\n" + "=" * 60)
        print("STEP 6.3: DAE Baseline Setup")
        print("=" * 60)
        print("This script is ready for DAE training.")
        print("Please complete:")
        print("  - Step 4.1: SN-D dataset (full)")
        print("  - Step 4.2: SN-D training (full)")
        print("  - Step 5.1: HN-E dataset (full)")
        print("  - Step 5.2: HN-E training (full)")
        print("  - Step 6.1: Unified N2LN training")
        print("=" * 60)
        
        print("\n✅ DAE baseline script ready!")
        print("   Run after completing Steps 4.2, 5.2, and 6.1")
    
    def evaluate(self):
        """Evaluate DAE baseline."""
        print("\n📊 DAE Baseline Evaluation")
        print("-" * 40)
        print("DAE vs Set Transformer comparison pending")
        print("Run after training both models")


if __name__ == "__main__":
    config_path = "experiments/exp3_unified/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    dae = DAEBaseline(config_path)
    dae.train()
    dae.evaluate()
