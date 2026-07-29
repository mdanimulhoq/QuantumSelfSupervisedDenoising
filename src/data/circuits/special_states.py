"""Special state circuit generators (TDD §5.1)."""

from qiskit import QuantumCircuit


def generate_ghz(n_qubits: int, depth: int = 1, seed: int = None) -> QuantumCircuit:
    """Generate a GHZ state preparation circuit. Depth ignored."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for q in range(1, n_qubits):
        qc.cx(0, q)
    qc.measure_all()
    return qc


def generate_w_state(n_qubits: int, depth: int = 1, seed: int = None) -> QuantumCircuit:
    """Generate a W-state preparation circuit. Depth ignored."""
    qc = QuantumCircuit(n_qubits)
    qc.x(0)
    for q in range(1, n_qubits):
        qc.cx(0, q)
        qc.x(0)
    qc.h(0)
    qc.measure_all()
    return qc
