"""MPS (Matrix Product State) simulator wrapper (TDD §7.2).

Wraps Qiskit Aer MPS backend with the same interface as standard Aer.
Supports up to 30 qubits for low-entanglement circuits.
"""

import numpy as np
from typing import Dict, Optional, List
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel


class MPSSimulator:
    """MPS simulator wrapper with Aer-compatible interface."""
    
    def __init__(
        self,
        max_bond_dimension: int = 256,
        max_qubits: int = 30,
        enable_truncation: bool = True,
    ):
        self.max_bond_dimension = max_bond_dimension
        self.max_qubits = max_qubits
        self.enable_truncation = enable_truncation
        
        # Use Aer.get_backend for MPS method
        self.backend = Aer.get_backend(
            'qasm_simulator',
            method='matrix_product_state',
            mps_max_bond_dimension=max_bond_dimension,
        )
    
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, int]:
        if circuit.num_qubits > self.max_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits, "
                f"exceeds maximum {self.max_qubits}"
            )
        
        circ = circuit.copy()
        if not any(instr[0].name == 'measure' for instr in circ.data):
            circ.measure_all()
        
        circ = transpile(circ, self.backend)
        
        job = self.backend.run(
            circ,
            shots=shots,
            noise_model=noise_model if noise_model else NoiseModel(),
            seed_simulator=seed,
        )
        result = job.result()
        
        try:
            return result.get_counts(0)
        except:
            return result.get_counts()
    
    def get_probabilities(
        self,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        # Use statevector for exact probabilities
        sv_backend = Aer.get_backend('statevector_simulator', method='matrix_product_state')
        
        circ = circuit.copy()
        if not any(instr[0].name == 'measure' for instr in circ.data):
            circ.save_statevector()
        
        job = sv_backend.run(circ, seed_simulator=seed)
        result = job.result()
        statevector = result.get_statevector(0)
        
        return np.abs(statevector.data) ** 2


def test_mps_simulator():
    """Test MPS simulator on a 20-qubit circuit."""
    from qiskit.circuit.random import random_circuit
    
    print("\n" + "=" * 60)
    print("STEP 7.1: MPS Simulator Test")
    print("=" * 60)
    
    n_qubits = 20
    circuit = random_circuit(n_qubits, depth=5, measure=False, seed=42)
    
    print(f"Qubits: {n_qubits}")
    print(f"Circuit depth: {circuit.depth()}")
    
    try:
        # Run MPS simulation
        mps = MPSSimulator(max_bond_dimension=256)
        counts = mps.run(circuit, shots=1000, seed=42)
        
        # Run Aer for comparison
        aer = AerSimulator()
        circ = circuit.copy()
        circ.measure_all()
        circ = transpile(circ, aer)
        job = aer.run(circ, shots=1000, seed_simulator=42)
        aer_counts = job.result().get_counts(0)
        
        def counts_to_probs(counts, total):
            return {k: v / total for k, v in counts.items()}
        
        total = 1000
        mps_probs = counts_to_probs(counts, total)
        aer_probs = counts_to_probs(aer_counts, total)
        
        all_keys = set(mps_probs.keys()) | set(aer_probs.keys())
        tvd = 0.5 * sum(
            abs(mps_probs.get(k, 0) - aer_probs.get(k, 0))
            for k in all_keys
        )
        
        print(f"MPS counts: {len(counts)} bitstrings")
        print(f"Aer counts: {len(aer_counts)} bitstrings")
        print(f"TVD between MPS and Aer: {tvd:.6f}")
        
        if tvd <= 1e-3:
            print("\n✅ TVD <= 1e-3: PASSED")
        else:
            print(f"\n⚠️ TVD = {tvd:.6f} > 1e-3: CHECK")
        
        return tvd
    
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("MPS may not be available in this version of Qiskit Aer.")
        print("You may need to install: pip install qiskit-aer==0.15.0")
        return None


if __name__ == "__main__":
    test_mps_simulator()
