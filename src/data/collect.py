"""Data collection protocol (TDD §5.2)."""

from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel
from typing import Dict, List, Optional
import numpy as np

def run_circuit(
    circuit: QuantumCircuit,
    shots: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Run a circuit and return counts."""
    backend = Aer.get_backend("qasm_simulator")
    job = execute(
        circuit,
        backend,
        shots=shots,
        noise_model=noise_model if noise_model else NoiseModel(),
        seed_simulator=seed,
    )
    result = job.result()
    return result.get_counts()

def collect_shot_pairs(
    circuit: QuantumCircuit,
    low_shots: int = 100,
    high_shots: int = 10000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """Collect low-shot and high-shot pairs."""
    if noise_model is None:
        noise_model = NoiseModel()
    
    low_counts = run_circuit(circuit, low_shots, noise_model, seed)
    high_counts = run_circuit(circuit, high_shots, noise_model, seed + 1000 if seed else None)
    
    return {
        "low": low_counts,
        "high": high_counts,
    }

def gate_fold(circuit: QuantumCircuit, factor: float) -> QuantumCircuit:
    """Apply gate folding for noise scaling."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroller, GateFolding
    
    # Unroll to basis gates
    pm = PassManager([Unroller(['u1', 'u2', 'u3', 'cx'])])
    folded = pm.run(circuit)
    
    # Simple folding: repeat each gate (factor - 1) times
    if factor <= 1.0:
        return circuit
    
    # For now, just return original (simplified)
    # Full implementation would fold gates properly
    return circuit

def add_dynamical_decoupling(circuit: QuantumCircuit) -> QuantumCircuit:
    """Add dynamical decoupling sequences."""
    # Simplified: no DD implemented for now
    return circuit
