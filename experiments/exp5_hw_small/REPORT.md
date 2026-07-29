# Experiment 5: Real Hardware Small Scale Report

## Summary
- **Hardware**: IBMQ Nairobi (7-qubit)
- **Circuits**: 50 test circuits
- **Shots**: 1000 per circuit
- **Model**: Unified N2LN fine-tuned on hardware

## Results

### TVD Comparison
| Method | TVD (Hardware) | Fidelity | QPU Time (s) |
|--------|----------------|----------|--------------|
| Raw (No mitigation) | - | - | - |
| SN-D Only | - | - | - |
| HN-E Only | - | - | - |
| Unified N2LN (Sim) | - | - | - |
| **Unified N2LN (Fine-tuned)** | - | - | - |

### Sim-to-Hardware Transfer Gap
- **Before fine-tuning**: TVD = -
- **After fine-tuning**: TVD = -
- **Improvement**: -

### Cost Analysis
| Item | Value |
|------|-------|
| Total QPU time used | - seconds |
| Circuits submitted | 50 |
| Shots per circuit | 1000 |
| Total shots | 50,000 |

### Distribution Visualization
- plots/hardware_distribution.png - Example distribution comparison
- plots/tvd_comparison.png - TVD comparison chart

## Success Criterion (TDD §6.4 Exp 5)
- **Goal**: Fine-tuned model beats non-fine-tuned model (p < 0.05)
- **Result**: Pending

## Files
- config.yaml: Experiment configuration
- inetune.py: Fine-tuning script
- est_model.pt: Fine-tuned model

## Next Steps
Proceed to Experiment 6: Utility Scale (127-qubit)
