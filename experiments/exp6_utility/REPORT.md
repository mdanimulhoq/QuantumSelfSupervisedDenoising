# Experiment 6: Utility-Scale Hardware Report (127-qubit)

## Summary
- **Hardware**: IBMQ Brisbane/Kyiv (127-qubit)
- **Circuits**: 50 test circuits
- **Mode**: SN-D only (shot-noise reduction)
- **Low shots**: 100
- **High shots**: 1000

## Results

### Shot-Noise Reduction
| Circuit | Raw TVD | SN-D TVD | Improvement |
|---------|---------|----------|-------------|
| Circuit 1 | - | - | -% |
| Circuit 2 | - | - | -% |
| Circuit 3 | - | - | -% |
| **Average** | - | - | -% |

### Success Criterion (TDD §7.4)
- **Goal**: TVD reduction on ≥3 utility circuits
- **Result**: Pending

### Distribution Visualization
- plots/utility_tvd_comparison.png - TVD comparison
- plots/utility_distribution_example.png - Example distributions

## Files
- config.yaml: Experiment configuration
- collect.py: Data collection script
- un_snd_only.py: SN-D demo script

## Next Steps
Proceed to Phase 10: Ablations, Baselines, Paper
