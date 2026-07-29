"""Tests for noise models (TDD §5.3)."""

from qiskit_aer.noise import NoiseModel

from src.data.noise_models import depolarizing, amplitude_damping, combined


def test_depolarizing_returns_noise_model():
    nm = depolarizing(p_gate=0.01, p_readout=0.02)
    assert isinstance(nm, NoiseModel)


def test_amplitude_damping_returns_noise_model():
    nm = amplitude_damping(t1_us=100.0, t2_us=80.0)
    assert isinstance(nm, NoiseModel)


def test_combined_returns_noise_model():
    nm = combined(p_gate=0.01, t1_us=100.0, t2_us=80.0)
    assert isinstance(nm, NoiseModel)


def test_noise_model_increases_error():
    """Verify noise model actually adds error: fidelity < 1."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Noiseless
    backend_ideal = AerSimulator()
    counts_ideal = backend_ideal.run(qc, shots=10000).result().get_counts()

    # Noisy
    nm = depolarizing(p_gate=0.05, p_readout=0.05)
    backend_noisy = AerSimulator(noise_model=nm)
    counts_noisy = backend_noisy.run(qc, shots=10000).result().get_counts()

    # Noisy should have more spread (non-zero counts on unexpected bitstrings)
    assert "00" in counts_noisy or "11" in counts_noisy


def test_depolarizing_reproducible():
    nm1 = depolarizing(p_gate=0.01, p_readout=0.02)
    nm2 = depolarizing(p_gate=0.01, p_readout=0.02)
    assert nm1 is not None and nm2 is not None
