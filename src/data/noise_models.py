"""Noise models for simulation (TDD §5.3)."""

from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, amplitude_damping_error, phase_damping_error

def depolarizing_noise(p: float = 0.01) -> NoiseModel:
    """Depolarizing noise model."""
    noise_model = NoiseModel()
    error = depolarizing_error(p, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'])
    return noise_model

def readout_error(p0: float = 0.02, p1: float = 0.02) -> NoiseModel:
    """Readout error model."""
    noise_model = NoiseModel()
    error = ReadoutError([[1 - p0, p0], [p1, 1 - p1]])
    noise_model.add_all_qubit_readout_error(error)
    return noise_model

def amplitude_damping(gamma: float = 0.01) -> NoiseModel:
    """Amplitude damping noise model."""
    noise_model = NoiseModel()
    error = amplitude_damping_error(gamma)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z', 's', 't'])
    return noise_model

def phase_damping(gamma: float = 0.01) -> NoiseModel:
    """Phase damping noise model."""
    noise_model = NoiseModel()
    error = phase_damping_error(gamma)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z', 's', 't'])
    return noise_model

def combined_noise(p_dep: float = 0.01, p0: float = 0.02, p1: float = 0.02) -> NoiseModel:
    """Combined depolarizing + readout noise."""
    noise_model = NoiseModel()
    
    # Depolarizing
    dep_error = depolarizing_error(p_dep, 1)
    noise_model.add_all_qubit_quantum_error(dep_error, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'])
    
    # Readout
    read_error = ReadoutError([[1 - p0, p0], [p1, 1 - p1]])
    noise_model.add_all_qubit_readout_error(read_error)
    
    return noise_model

def ibmq_realistic(backend_name: str = "nairobi") -> NoiseModel:
    """IBMQ realistic noise from calibration data."""
    # This uses Qiskit's FakeBackend for realistic noise
    try:
        from qiskit.providers.fake_provider import FakeNairobi, FakeBrisbane, FakeKyiv
        fake_backends = {
            "nairobi": FakeNairobi,
            "brisbane": FakeBrisbane,
            "kyiv": FakeKyiv,
        }
        backend = fake_backends.get(backend_name, FakeNairobi)()
        return NoiseModel.from_backend(backend)
    except ImportError:
        # Fallback to combined noise
        print(f"Warning: FakeBackend not available, using combined noise")
        return combined_noise(0.01, 0.02, 0.02)
