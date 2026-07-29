# N2LN-QEM Makefile - Reproducibility Bundle (TDD §10.3)
# Usage: make reproduce  # Full end-to-end run

.PHONY: help setup data train evaluate clean reproduce

help:
    @echo "Available targets:"
    @echo "  make setup      - Install dependencies and setup environment"
    @echo "  make data       - Generate all datasets (small config for testing)"
    @echo "  make train      - Train all models (SN-D, HN-E, Unified)"
    @echo "  make evaluate   - Run all evaluations"
    @echo "  make reproduce  - Full end-to-end pipeline"
    @echo "  make clean      - Clean all generated files"

setup:
    pip install -r requirements.txt
    pip install -e .
    python -c "from src.utils.seeding import set_seed; set_seed(42); print('Setup complete')"

data:
    python experiments/exp1_snd/generate_data.py
    python experiments/exp2_hne/generate_data.py
    python experiments/exp4_scale/generate_data.py

train:
    python experiments/exp1_snd/train.py
    python experiments/exp2_hne/train.py
    python experiments/exp3_unified/train.py

evaluate:
    python experiments/exp1_snd/evaluate.py
    python experiments/exp2_hne/baselines/zne.py
    python experiments/exp3_unified/evaluate.py
    python experiments/exp7_ablation/run_ablation.py
    python experiments/exp8_baselines/run_baselines.py

reproduce: setup data train evaluate
    @echo "========================================="
    @echo "✅ Reproducibility bundle complete!"
    @echo "========================================="

clean:
    rm -rf data/raw/*
    rm -rf checkpoints/*
    rm -rf experiments/*/plots/*
    rm -rf experiments/*/REPORT.md
    rm -rf __pycache__ */__pycache__
    @echo "Cleaned all generated files"

# Quick test (small config for CI)
test:
    python experiments/exp1_snd/generate_data.py
    python experiments/exp1_snd/train.py
    python experiments/exp1_snd/evaluate.py
    @echo "✅ Quick test complete"
