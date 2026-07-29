"""VQE ansatz circuit generator."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
import numpy as np

def generate_vqe(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Generate a hardware-efficient VQE ansatz."""
    np.random.seed(seed)
    # Use EfficientSU2 from Qiskit
    reps = depth
    circuit = EfficientSU2(n_qubits, reps=reps, entanglement='linear')
    
    # Random parameters
    params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)
    return circuit.assign_parameters(params)
