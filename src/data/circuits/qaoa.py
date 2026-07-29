"""QAOA circuit generator."""

from qiskit import QuantumCircuit
import numpy as np

def generate_qaoa(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Generate a QAOA circuit for MaxCut."""
    np.random.seed(seed)
    circuit = QuantumCircuit(n_qubits)
    
    # Initial state: Hadamard on all qubits
    for q in range(n_qubits):
        circuit.h(q)
    
    # QAOA layers
    for layer in range(depth):
        # Problem Hamiltonian (ZZ interactions on random edges)
        for _ in range(n_qubits):
            q1 = np.random.randint(0, n_qubits)
            q2 = np.random.randint(0, n_qubits)
            if q1 != q2:
                gamma = np.random.uniform(0, 2 * np.pi)
                circuit.rzz(gamma, q1, q2)
        
        # Mixer Hamiltonian (X rotations)
        for q in range(n_qubits):
            beta = np.random.uniform(0, np.pi)
            circuit.rx(beta, q)
    
    return circuit
