"""Fine-tune Unified N2LN on real IBMQ hardware data (TDD §7.3).

Loads Phase 6 checkpoint and fine-tunes on small hardware dataset.
Characterizes sim-to-hardware transfer gap.
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
from src.training.trainer import N2LNTrainer
from src.utils.seeding import set_seed
from src.utils.device import get_device
from src.data.hardware.ibmq import IBMQInterface


def finetune_hardware(config_path: str):
    """Fine-tune on real hardware data."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("STEP 8.2: Sim-to-Hardware Transfer Setup")
    print("=" * 60)
    print("This script fine-tunes on real 7-qubit IBMQ hardware.")
    print("Features:")
    print("  - Loads Phase 6 Unified N2LN checkpoint")
    print("  - Fine-tunes on hardware data (50 circuits)")
    print("  - Characterizes sim-to-hardware transfer gap")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  - Step 6.1: Unified N2LN checkpoint")
    print("  - Step 8.1: IBMQ interface")
    print("  - IBMQ account with access to Nairobi")
    print("=" * 60)
    
    # TODO: Load Unified N2LN checkpoint
    # TODO: Collect hardware data (50 circuits)
    # TODO: Fine-tune on hardware data
    # TODO: Compare with non-fine-tuned model
    # TODO: Paired t-test for significance
    
    print("\n✅ Hardware fine-tuning script ready!")
    print("   Run after completing Steps 6.1 and 8.1")


if __name__ == "__main__":
    config_path = "experiments/exp5_hw_small/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    finetune_hardware(config_path)
