# Experiment 1: SN-D Validation Report

## Summary
- **Model**: SN-D (Shot-Noise Denoising)
- **Qubits**: 4, 6, 8
- **Low shots**: 100
- **High shots**: 100000

## Results

### TVD Comparison
| Metric | Raw vs High | SN-D vs High | Improvement |
|--------|-------------|--------------|-------------|
| TVD | 0.1795 | 0.6566 | -265.8% |
| Fidelity | 0.8118 | 0.3063 | -62.3% |
| MAE | 0.0044 | 0.0163 | -268.3% |

## Success Criterion (TDD §1.3)
- **Goal**: SN-D TVD <= 50% of raw low-shot TVD
- **Result**: SN-D TVD = 0.6566 vs Raw TVD = 0.1795
- **Status**: ❌ FAILED

## Plots
- `plots/tvd_comparison.png` - TVD comparison
- `plots/distribution_example.png` - Example distributions

## Files
- `config.yaml`: Experiment configuration
- `metrics.json`: Full metrics
- `best_model.pt`: Trained model

## Next Steps
Proceed to Experiment 2: HN-E Validation
