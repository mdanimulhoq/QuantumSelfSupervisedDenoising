"""Generate dataset for Experiment 1: SN-D validation.

Creates low-shot (100) → high-shot (100,000) pairs for n ∈ {4, 6, 8}.
Circuits: Clifford + random non-Clifford.
Noise: depolarizing (p=0.01) + readout error (p=0.02).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Import project modules
from src.data.circuits.clifford import generate_clifford
from src.data.circuits.random_layer import generate_random_non_clifford
from src.data.circuits.vqe import generate_vqe
from src.data.circuits.qaoa import generate_qaoa
from src.data.noise_models import combined_noise
from src.data.encoding import counts_to_tensor


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "n_qubits": [4, 6, 8],
    "circuits_per_n": {4: 5, 6: 5, 8: 5},
    "circuit_families": {
        "clifford": 0.3,
        "random_non_clifford": 0.4,
        "vqe": 0.2,
        "qaoa": 0.1,
    },
    "low_shots": 10,
    "high_shots": 1000,
    "noise": {
        "depolarizing_p": 0.01,
        "readout_p0": 0.02,
        "readout_p1": 0.02,
    },
    "seed": 42,
    "output_dir": "data/raw/exp1_snd",
}


# =============================================================================
# Circuit Generation
# =============================================================================

def generate_circuit(
    n_qubits: int,
    family: str,
    depth: int,
    seed: int,
) -> QuantumCircuit:
    """Generate a circuit of the specified family."""
    np.random.seed(seed)
    if family == "clifford":
        return generate_clifford(n_qubits, depth, seed)
    elif family == "random_non_clifford":
        return generate_random_non_clifford(n_qubits, depth, seed)
    elif family == "vqe":
        return generate_vqe(n_qubits, depth, seed)
    elif family == "qaoa":
        return generate_qaoa(n_qubits, depth, seed)
    else:
        raise ValueError(f"Unknown circuit family: {family}")


def get_depth(n_qubits: int, family: str) -> int:
    """Get circuit depth based on qubit count and family."""
    if family in ["clifford", "random_non_clifford"]:
        return min(10, 3 * n_qubits)
    elif family in ["vqe", "qaoa"]:
        return 2
    else:
        return 5


# =============================================================================
# Noise Model
# =============================================================================

def get_noise_model() -> NoiseModel:
    """Get combined noise model for Experiment 1."""
    p_dep = CONFIG["noise"]["depolarizing_p"]
    p0 = CONFIG["noise"]["readout_p0"]
    p1 = CONFIG["noise"]["readout_p1"]
    return combined_noise(p_dep, p0, p1)


# =============================================================================
# Data Collection
# =============================================================================

def run_circuit(
    circuit: QuantumCircuit,
    shots: int,
    noise_model: NoiseModel,
    seed: int,
) -> Dict[str, int]:
    """Run a circuit and return counts."""
    from qiskit import transpile
    
    # Make a copy and add measurements if missing
    circ = circuit.copy()
    if not any(instr[0].name == 'measure' for instr in circ.data):
        circ.measure_all()
    
    backend = AerSimulator()
    
    # Transpile for better compatibility
    circ = transpile(circ, backend)
    
    job = backend.run(
        circ,
        shots=shots,
        noise_model=noise_model,
        seed_simulator=seed,
    )
    result = job.result()
    
    # Try multiple ways to get counts
    try:
        # Method 1: Direct
        counts = result.get_counts()
        if counts:
            return counts
    except:
        pass
    
    try:
        # Method 2: With index
        counts = result.get_counts(0)
        if counts:
            return counts
    except:
        pass
    
    try:
        # Method 3: From results
        counts = result.results[0].data.to_dict()
        if counts:
            return counts
    except:
        pass
    
    # Last resort: if empty, return empty dict
    return {}



def collect_pair(
    circuit: QuantumCircuit,
    low_shots: int,
    high_shots: int,
    noise_model: NoiseModel,
    seed: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Collect low-shot and high-shot counts for a circuit."""
    low_counts = run_circuit(circuit, low_shots, noise_model, seed)
    high_counts = run_circuit(circuit, high_shots, noise_model, seed + 1000)
    return low_counts, high_counts


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_dataset() -> None:
    """Generate the full dataset for Experiment 1."""
    np.random.seed(CONFIG["seed"])
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    noise_model = get_noise_model()

    all_data = []
    total_circuits = sum(CONFIG["circuits_per_n"].values())

    print(f"Generating {total_circuits} circuits for Experiment 1...")

    for n_qubits in CONFIG["n_qubits"]:
        n_circuits = CONFIG["circuits_per_n"][n_qubits]
        print(f"\n  Qubits: {n_qubits} ({n_circuits} circuits)")

        # Determine family distribution
        families = []
        for family, ratio in CONFIG["circuit_families"].items():
            n_family = int(ratio * n_circuits)
            families.extend([family] * n_family)
        # Fill remaining with random
        while len(families) < n_circuits:
            families.append("random_non_clifford")
        np.random.shuffle(families)

        for i in tqdm(range(n_circuits), desc=f"n={n_qubits}"):
            family = families[i]
            depth = get_depth(n_qubits, family)
            seed = CONFIG["seed"] + i * 100 + n_qubits * 10000

            # Generate circuit
            circuit = generate_circuit(n_qubits, family, depth, seed)

            # Collect counts
            low_counts, high_counts = collect_pair(
                circuit,
                CONFIG["low_shots"],
                CONFIG["high_shots"],
                noise_model,
                seed,
            )

            # Get ideal distribution (noise-free) for evaluation
            ideal_counts = run_circuit(circuit, CONFIG["high_shots"], NoiseModel(), seed + 2000)

            # Convert to tensors
            low_bitstrings, low_probs = counts_to_tensor(low_counts, n_qubits)
            high_bitstrings, high_probs = counts_to_tensor(high_counts, n_qubits)
            ideal_bitstrings, ideal_probs = counts_to_tensor(ideal_counts, n_qubits)

            all_data.append({
                "n_qubits": n_qubits,
                "family": family,
                "depth": depth,
                "seed": seed,
                "low_shots": CONFIG["low_shots"],
                "high_shots": CONFIG["high_shots"],
                "low_bitstrings": low_bitstrings.numpy(),
                "low_probs": low_probs.numpy(),
                "high_bitstrings": high_bitstrings.numpy(),
                "high_probs": high_probs.numpy(),
                "ideal_bitstrings": ideal_bitstrings.numpy(),
                "ideal_probs": ideal_probs.numpy(),
            })

    # Save dataset
    save_dataset(all_data, output_dir)
    save_metadata(output_dir)

    print(f"\n✅ Dataset saved to {output_dir}")
    print(f"   Total circuits: {len(all_data)}")


def save_dataset(data: List[Dict], output_dir: str) -> None:
    """Save dataset to HDF5 file."""
    # Split into train/val/test
    np.random.seed(42)
    n = len(data)
    indices = np.random.permutation(n)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    for split_name, split_indices in splits.items():
        if len(split_indices) == 0:
            continue

        filepath = os.path.join(output_dir, f"exp1_snd_{split_name}.h5")
        with h5py.File(filepath, "w") as f:
            # Store each field
            for key in data[0].keys():
                if key in ["low_bitstrings", "high_bitstrings", "ideal_bitstrings"]:
                    # Variable-length arrays
                    dtype = h5py.special_dtype(vlen=np.int64)
                    f.create_dataset(key, (len(split_indices),), dtype=dtype)
                    for j, idx in enumerate(split_indices):
                        f[key][j] = data[idx][key]
                elif key in ["low_probs", "high_probs", "ideal_probs"]:
                    dtype = h5py.special_dtype(vlen=np.float32)
                    f.create_dataset(key, (len(split_indices),), dtype=dtype)
                    for j, idx in enumerate(split_indices):
                        f[key][j] = data[idx][key]
                else:
                    # Scalar values
                    values = [data[idx][key] for idx in split_indices]
                    if isinstance(values[0], str):
                        dt = h5py.special_dtype(vlen=str)
                        f.create_dataset(key, (len(split_indices),), dtype=dt, data=values)
                    else:
                        f.create_dataset(key, data=values)

        print(f"   Saved {split_name}: {len(split_indices)} circuits")


def save_metadata(output_dir: str) -> None:
    """Save dataset metadata."""
    metadata = {
        "experiment": "exp1_snd",
        "description": "SN-D validation: low-shot (100) → high-shot (100,000)",
        "config": CONFIG,
        "train_val_test_split": [0.7, 0.15, 0.15],
        "total_circuits": sum(CONFIG["circuits_per_n"].values()),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("N2LN-QEM: Experiment 1 Dataset Generation")
    print("=" * 60)
    print(f"Qubits: {CONFIG['n_qubits']}")
    print(f"Low shots: {CONFIG['low_shots']}")
    print(f"High shots: {CONFIG['high_shots']}")
    print(f"Noise: depolarizing(p={CONFIG['noise']['depolarizing_p']}) + readout")
    print("=" * 60)

    generate_dataset()



