# Experiment 3: Unified N2LN Report

## Summary
- **Model**: Unified N2LN (SN-D + HN-E with consistency loss)
- **Qubits**: 4, 6, 8
- **Training**: Joint fine-tuning with consistency loss

## Results

### TVD Comparison
| Method | TVD | Fidelity | MAE |
|--------|-----|----------|-----|
| Raw | - | - | - |
| SN-D Only | - | - | - |
| HN-E Only | - | - | - |
| ZNE | - | - | - |
| **Unified N2LN** | - | - | - |
| Ideal-Supervised | - | - | - |

### Success Criterion (TDD §1.3)
- **Goal**: Unified N2LN approaches Ideal-Supervised within 1.5x
- **Result**: Pending

## Files
- config.yaml: Experiment configuration
- est_model.pt: Trained unified model
- evaluate.py: Evaluation script

## Next Steps
Proceed to Experiment 4: Qubit Scaling
