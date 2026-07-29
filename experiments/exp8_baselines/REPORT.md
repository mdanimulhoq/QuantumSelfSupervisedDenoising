# Experiment 8: ML-QEM Baselines Report

## Head-to-Head Comparison

### TVD Comparison
| Method | TVD (n=4) | TVD (n=6) | TVD (n=8) | Average |
|--------|-----------|-----------|-----------|---------|
| Raw | - | - | - | - |
| CDR | - | - | - | - |
| ML-QEM RF | - | - | - | - |
| ML-QEM MLP | - | - | - | - |
| ML-QEM GNN | - | - | - | - |
| **N2LN (Ours)** | - | - | - | - |

### Success Criterion (TDD §6.4 Exp 7)
- **Goal**: N2LN wins on the metric it was designed for (TVD)
- **Result**: Pending

## Files
- config.yaml: Baseline configurations
- cdr.py: CDR implementation
- ml_qem.py: IBM ML-QEM implementation
- un_baselines.py: Run all baselines
