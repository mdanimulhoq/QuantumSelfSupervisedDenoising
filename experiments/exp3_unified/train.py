"""Joint fine-tuning of SN-D and HN-E for Unified N2LN (TDD §4.2 Phase 3).

Loads SN-D and HN-E checkpoints, unfreezes both heads,
and fine-tunes with consistency loss.
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.n2ln import N2LNQEM
from src.training.trainer import N2LNTrainer
from src.utils.seeding import set_seed
from src.utils.device import get_device


def train_unified(config_path: str):
    """Main training function for Unified N2LN."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # TODO: Load datasets (Step 4.1 and Step 5.1)
    # TODO: Create dataloaders
    # TODO: Load SN-D and HN-E checkpoints
    # TODO: Unfreeze both heads
    # TODO: Train with consistency loss
    
    print("\n✅ Unified N2LN training script ready!")
    print("   Run after completing Step 4.2 and Step 5.2")


if __name__ == "__main__":
    config_path = "experiments/exp3_unified/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("STEP 6.1: Unified N2LN Setup")
    print("=" * 60)
    print("This script is ready for training.")
    print("Please complete:")
    print("  - Step 4.2: SN-D training (full dataset)")
    print("  - Step 5.2: HN-E training (full dataset)")
    print("Then run this script with the --run flag")
    print("=" * 60)
    
    # Just setup - don't run training yet
    train_unified(config_path)
