"""Run all ML-QEM baselines (TDD §6.4 Exp 7).

Compares N2LN against:
1. CDR (Clifford Data Regression)
2. IBM ML-QEM (RF, MLP, GNN)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
from pathlib import Path

from experiments.exp8_baselines.cdr import CDRBaseline
from experiments.exp8_baselines.ml_qem import IBM_MLQEM


def run_all_baselines(config_path: str):
    """Run all baseline experiments."""
    
    print("\n" + "=" * 60)
    print("STEP 10.2: ML-QEM Baselines Setup")
    print("=" * 60)
    print("Baselines to run:")
    print("  1. CDR (Clifford Data Regression)")
    print("  2. IBM ML-QEM (Random Forest)")
    print("  3. IBM ML-QEM (MLP)")
    print("  4. IBM ML-QEM (GNN - placeholder)")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  - Step 4.2: SN-D training (full)")
    print("  - Step 6.1: Unified N2LN training")
    print("  - sklearn installed: pip install scikit-learn")
    print("=" * 60)
    
    # TODO: Load data
    # TODO: Train each baseline
    # TODO: Evaluate and compare with N2LN
    # TODO: Generate head-to-head table
    
    print("\n✅ All baselines ready!")
    print("   Run after completing Steps 4.2 and 6.1")


if __name__ == "__main__":
    config_path = "experiments/exp8_baselines/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    run_all_baselines(config_path)
