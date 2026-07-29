"""Tests for circuit generators (TDD §5.1)."""

from qiskit import QuantumCircuit

from src.data.circuits import (
    generate_clifford,
    generate_random,
    generate_vqe,
    generate_qaoa,
    generate_ghz,
    generate_w_state,
)


def _check_circuit(qc: QuantumCircuit, n_qubits: int):
    """Helper: verify circuit is valid."""
    assert qc.num_qubits == n_qubits
    assert len(qc.data) > 0


def test_clifford_4_qubits():
    qc = generate_clifford(4, 10, seed=42)
    _check_circuit(qc, 4)


def test_clifford_8_qubits():
    qc = generate_clifford(8, 15, seed=123)
    _check_circuit(qc, 8)


def test_clifford_12_qubits():
    qc = generate_clifford(12, 20, seed=7)
    _check_circuit(qc, 12)


def test_random_4_qubits():
    qc = generate_random(4, 10, seed=42)
    _check_circuit(qc, 4)


def test_random_8_qubits():
    qc = generate_random(8, 15, seed=123)
    _check_circuit(qc, 8)


def test_random_12_qubits():
    qc = generate_random(12, 20, seed=7)
    _check_circuit(qc, 12)


def test_vqe_4_qubits():
    qc = generate_vqe(4, 3, seed=42)
    _check_circuit(qc, 4)


def test_vqe_8_qubits():
    qc = generate_vqe(8, 4, seed=99)
    _check_circuit(qc, 8)


def test_qaoa_4_qubits():
    qc = generate_qaoa(4, 2, seed=42)
    _check_circuit(qc, 4)


def test_qaoa_8_qubits():
    qc = generate_qaoa(8, 3, seed=55)
    _check_circuit(qc, 8)


def test_ghz_4_qubits():
    qc = generate_ghz(4)
    _check_circuit(qc, 4)


def test_w_state_4_qubits():
    qc = generate_w_state(4)
    _check_circuit(qc, 4)


def test_reproducibility():
    qc1 = generate_random(4, 10, seed=42)
    qc2 = generate_random(4, 10, seed=42)
    assert qc1.num_qubits == qc2.num_qubits
