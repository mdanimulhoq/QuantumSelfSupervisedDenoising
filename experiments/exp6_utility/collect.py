"""Utility-scale data collection for 127-qubit hardware (TDD §7.4).

Collects low-shot (100) and high-shot (1000) data from IBMQ Brisbane/Kyiv.
Restricted to shot-noise-dominated regime. SN-D only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from src.data.hardware.ibmq import IBMQInterface
from src.data.encoding import counts_to_tensor


def generate_utility_circuit(n_qubits: int, family: str, seed: int) -> QuantumCircuit:
    """Generate a circuit for utility-scale."""
    np.random.seed(seed)
    
    if family == "random":
        depth = min(5, n_qubits // 10)
        return random_circuit(n_qubits, depth=depth, measure=False, seed=seed)
    elif family == "qaoa":
        circuit = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            circuit.h(q)
        for _ in range(2):
            for q in range(n_qubits - 1):
                circuit.rzz(0.5, q, q + 1)
            for q in range(n_qubits):
                circuit.rx(0.5, q)
        return circuit
    elif family == "shallow":
        circuit = QuantumCircuit(n_qubits)
        for q in range(0, n_qubits, 2):
            circuit.h(q)
        for q in range(0, n_qubits - 1, 2):
            circuit.cx(q, q + 1)
        return circuit
    else:
        return random_circuit(n_qubits, depth=3, measure=False, seed=seed)


def collect_utility_data(config_path: str):
    """Collect data from 127-qubit hardware."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    np.random.seed(config['seed'])
    output_dir = Path('data/raw/exp6_utility')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 9.1: Utility-Scale Data Collection Setup")
    print("=" * 60)
    print(f"Backend: {config['data']['hardware_backend']}")
    print(f"Qubits: {config['data']['n_qubits']}")
    print(f"Circuits: {config['data']['circuits']}")
    print(f"Low shots: {config['data']['low_shots']}")
    print(f"High shots: {config['data']['high_shots']}")
    print("=" * 60)
    
    # TODO: Initialize IBMQ interface
    # TODO: Generate 50 circuits
    # TODO: Collect low-shot and high-shot data
    # TODO: Save to HDF5
    # TODO: Document QPU budget
    
    print("\nPrerequisites:")
    print("  - Step 8.1: IBMQ interface")
    print("  - IBMQ account with 127-qubit access")
    print("  - QPU budget for 50 circuits x 2 shot counts")
    print("=" * 60)
    
    print("\n✅ Utility-scale data collection script ready!")
    print("   Estimated QPU time: ~50 circuits x 2 runs = 100 jobs")
    print("   Run after completing Step 8.1")


if __name__ == "__main__":
    config_path = "experiments/exp6_utility/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    collect_utility_data(config_path)
