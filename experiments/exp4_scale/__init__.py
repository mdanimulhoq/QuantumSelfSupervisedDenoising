# Experiment 4: Qubit Scaling (10-30 qubits)

## Purpose
Test qubit-count generalization using MPS simulator.
Train on n=4,6 → test on n=10,15,20,25,30.

## Prerequisites
- Step 7.1: MPS simulator integration

## Usage
`ash
python experiments/exp4_scale/generate_data.py
New-Item -ItemType Directory -Path experiments\exp4_scale -Force
@"
"""Experiment 4: Qubit Scaling (10-30 qubits with MPS)."""
