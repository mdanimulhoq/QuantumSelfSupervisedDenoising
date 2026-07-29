"""Clifford circuit generator."""

from qiskit import QuantumCircuit
import numpy as np

def generate_clifford(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Generate a random Clifford circuit."""
    np.random.seed(seed)
    circuit = QuantumCircuit(n_qubits)
    
    for _ in range(depth):
        # Random single-qubit Clifford gates
        for q in range(n_qubits):
            gate = np.random.choice(['h', 's', 'sdg'])
            if gate == 'h':
                circuit.h(q)
            elif gate == 's':
                circuit.s(q)
            else:
                circuit.sdg(q)
        
        # Random CNOTs
        for _ in range(n_qubits // 2):
            control = np.random.randint(0, n_qubits)
            target = np.random.randint(0, n_qubits)
            if control != target:
                circuit.cx(control, target)
    
    return circuit
