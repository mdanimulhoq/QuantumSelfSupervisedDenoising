"""Tests for data encoding (TDD §5.4)."""

import torch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from src.data.encoding import (
    counts_to_bitstrings_and_probs,
    counts_to_tensor,
    tensor_to_counts,
    batch_counts_to_tensors,
)


def test_counts_to_bitstrings_and_probs_shape():
    counts = {"00": 30, "01": 20, "10": 40, "11": 10}
    bs, pr = counts_to_bitstrings_and_probs(counts, n_qubits=2)
    assert bs.shape == (4, 2)
    assert pr.shape == (4,)
    assert torch.allclose(pr.sum(), torch.tensor(1.0))


def test_counts_to_bitstrings_sorted_by_count():
    counts = {"11": 10, "00": 70, "01": 15, "10": 5}
    bs, pr = counts_to_bitstrings_and_probs(counts, n_qubits=2)
    assert pr[0] == 0.70


def test_counts_to_tensor_full_vector():
    counts = {"00": 25, "01": 25, "10": 25, "11": 25}
    prob_vec = counts_to_tensor(counts, n_qubits=2)
    assert prob_vec.shape == (4,)
    assert torch.allclose(prob_vec, torch.tensor([0.25, 0.25, 0.25, 0.25]))


def test_roundtrip_counts_tensor_counts():
    counts = {"00": 30, "01": 20, "10": 40, "11": 10}
    bs, pr = counts_to_bitstrings_and_probs(counts, n_qubits=2)
    recovered = tensor_to_counts(bs, pr, n_qubits=2, total_shots=100)
    assert recovered["00"] >= 28 and recovered["00"] <= 32


def test_batch_counts_to_tensors():
    c1 = {"00": 50, "11": 50}
    c2 = {"01": 30, "10": 70}
    bs_batch, pr_batch = batch_counts_to_tensors([c1, c2], n_qubits=2)
    assert bs_batch.shape[0] == 2
    assert bs_batch.shape[2] == 2
    assert pr_batch.shape[0] == 2


def test_real_qiskit_counts():
    """End-to-end: Qiskit circuit -> Counts -> tensor."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    backend = AerSimulator()
    job = backend.run(qc, shots=1000)
    counts = job.result().get_counts()
    bs, pr = counts_to_bitstrings_and_probs(counts, n_qubits=2)
    assert bs.shape[1] == 2
    assert torch.allclose(pr.sum(), torch.tensor(1.0))
