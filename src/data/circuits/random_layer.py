"""Random non-Clifford circuit generator."""

from qiskit import QuantumCircuit
import numpy as np

def generate_random_non_clifford(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Generate a random circuit with non-Clifford gates."""
    np.random.seed(seed)
    circuit = QuantumCircuit(n_qubits)
    
    for _ in range(depth):
        for q in range(n_qubits):
            gate = np.random.choice(['h', 't', 'tdg', 'x', 'y', 'z'])
            if gate == 'h':
                circuit.h(q)
            elif gate == 't':
                circuit.t(q)
            elif gate == 'tdg':
                circuit.tdg(q)
            elif gate == 'x':
                circuit.x(q)
            elif gate == 'y':
                circuit.y(q)
            else:
                circuit.z(q)
        
        # Random CNOTs
        for _ in range(n_qubits):
            control = np.random.randint(0, n_qubits)
            target = np.random.randint(0, n_qubits)
            if control != target:
                circuit.cx(control, target)
    
    return circuit

# Alias for backward compatibility
generate_random = generate_random_non_clifford
