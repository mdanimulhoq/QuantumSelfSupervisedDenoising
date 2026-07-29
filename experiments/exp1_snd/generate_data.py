"""Generate dataset for Experiment 1: SN-D validation with auto-save.

Auto-saves after each qubit block to prevent data loss on restart.
"""

import sys
import os
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
    "circuits_per_n": {4: 2000, 6: 1500, 8: 1500},
    "circuit_families": {
        "clifford": 0.3,
        "random_non_clifford": 0.4,
        "vqe": 0.2,
        "qaoa": 0.1,
    },
    "low_shots": 100,
    "high_shots": 100000,
    "noise": {
        "depolarizing_p": 0.01,
        "readout_p0": 0.02,
        "readout_p1": 0.02,
    },
    "seed": 42,
    "output_dir": "data/raw/exp1_snd",
    "checkpoint_file": "data/raw/exp1_snd/checkpoint.json",  # track progress
}

# =============================================================================
# Circuit Generation
# =============================================================================

def generate_circuit(n_qubits, family, depth, seed):
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
        raise ValueError(f"Unknown family: {family}")

def get_depth(n_qubits, family):
    if family in ["clifford", "random_non_clifford"]:
        return min(10, 3 * n_qubits)
    elif family in ["vqe", "qaoa"]:
        return 2
    else:
        return 5

def get_noise_model():
    p_dep = CONFIG["noise"]["depolarizing_p"]
    p0 = CONFIG["noise"]["readout_p0"]
    p1 = CONFIG["noise"]["readout_p1"]
    return combined_noise(p_dep, p0, p1)

def run_circuit(circuit, shots, noise_model, seed):
    from qiskit import transpile
    circ = circuit.copy()
    if not any(instr[0].name == 'measure' for instr in circ.data):
        circ.measure_all()
    backend = AerSimulator()
    circ = transpile(circ, backend)
    job = backend.run(circ, shots=shots, noise_model=noise_model, seed_simulator=seed)
    result = job.result()
    try:
        return result.get_counts(0)
    except:
        return result.get_counts()

# =============================================================================
# Auto-Save Functions
# =============================================================================

def load_checkpoint():
    """Load progress from checkpoint."""
    ckpt_path = CONFIG["checkpoint_file"]
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            return json.load(f)
    return {"completed_qubits": []}

def save_checkpoint(completed_qubits):
    """Save progress checkpoint."""
    ckpt_path = CONFIG["checkpoint_file"]
    with open(ckpt_path, 'w') as f:
        json.dump({"completed_qubits": completed_qubits}, f)

def save_block(data, n_qubits, output_dir):
    """Save a single qubit block to HDF5."""
    # Create temporary file for this block
    block_file = os.path.join(output_dir, f"block_{n_qubits}.h5")
    with h5py.File(block_file, 'w') as f:
        # Store each key as dataset
        for key in data[0].keys():
            values = [d[key] for d in data]
            if isinstance(values[0], (int, float, np.integer, np.floating)):
                f.create_dataset(key, data=np.array(values))
            elif isinstance(values[0], str):
                dt = h5py.special_dtype(vlen=str)
                ds = f.create_dataset(key, (len(values),), dtype=dt)
                for i, v in enumerate(values):
                    ds[i] = v
            else:
                # bitstrings or probs
                if 'bitstrings' in key:
                    dtype = h5py.special_dtype(vlen=np.int64)
                    ds = f.create_dataset(key, (len(values),), dtype=dtype)
                    shape_ds = f.create_dataset(f"{key}_shape", (len(values), 2), dtype=np.int64)
                    for i, arr in enumerate(values):
                        ds[i] = arr.flatten().astype(np.int64)
                        shape_ds[i] = arr.shape
                elif 'probs' in key:
                    dtype = h5py.special_dtype(vlen=np.float32)
                    ds = f.create_dataset(key, (len(values),), dtype=dtype)
                    shape_ds = f.create_dataset(f"{key}_shape", (len(values), 1), dtype=np.int64)
                    for i, arr in enumerate(values):
                        ds[i] = arr.astype(np.float32)
                        shape_ds[i] = arr.shape[0]
    print(f"   ✅ Block {n_qubits} saved to {block_file}")

def merge_blocks(output_dir, final_files):
    """Merge all block files into final train/val/test splits."""
    # Load all data from block files
    all_data = []
    for nq in CONFIG["n_qubits"]:
        block_file = os.path.join(output_dir, f"block_{nq}.h5")
        if not os.path.exists(block_file):
            continue
        with h5py.File(block_file, 'r') as f:
            # Read data
            n_samples = len(f['n_qubits'])
            for i in range(n_samples):
                entry = {}
                for key in f.keys():
                    if key.endswith('_shape'):
                        continue
                    val = f[key][i]
                    if isinstance(val, np.ndarray) and val.dtype == np.object_:
                        # variable length
                        shape_key = key + "_shape"
                        if shape_key in f:
                            shape = f[shape_key][i]
                            val = val.reshape(shape)
                    entry[key] = val
                all_data.append(entry)
    # Now split and save as final HDF5
    np.random.seed(42)
    n = len(all_data)
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
        with h5py.File(filepath, 'w') as f:
            for key in all_data[0].keys():
                values = [all_data[idx][key] for idx in split_indices]
                if isinstance(values[0], (int, float, np.integer, np.floating)):
                    f.create_dataset(key, data=np.array(values))
                elif isinstance(values[0], str):
                    dt = h5py.special_dtype(vlen=str)
                    ds = f.create_dataset(key, (len(values),), dtype=dt)
                    for i, v in enumerate(values):
                        ds[i] = v
                else:
                    if 'bitstrings' in key:
                        dtype = h5py.special_dtype(vlen=np.int64)
                        ds = f.create_dataset(key, (len(values),), dtype=dtype)
                        shape_ds = f.create_dataset(f"{key}_shape", (len(values), 2), dtype=np.int64)
                        for i, arr in enumerate(values):
                            ds[i] = arr.flatten().astype(np.int64)
                            shape_ds[i] = arr.shape
                    elif 'probs' in key:
                        dtype = h5py.special_dtype(vlen=np.float32)
                        ds = f.create_dataset(key, (len(values),), dtype=dtype)
                        shape_ds = f.create_dataset(f"{key}_shape", (len(values), 1), dtype=np.int64)
                        for i, arr in enumerate(values):
                            ds[i] = arr.astype(np.float32)
                            shape_ds[i] = arr.shape[0]
        print(f"   ✅ Saved {split_name}: {len(split_indices)} circuits")
    # Remove block files after merge
    for nq in CONFIG["n_qubits"]:
        block_file = os.path.join(output_dir, f"block_{nq}.h5")
        if os.path.exists(block_file):
            os.remove(block_file)

# =============================================================================
# Main Dataset Generation with Auto-Save
# =============================================================================

def generate_dataset():
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = load_checkpoint()
    completed = checkpoint.get("completed_qubits", [])
    
    noise_model = get_noise_model()
    all_data = []  # will collect all data for final merge
    
    for n_qubits in CONFIG["n_qubits"]:
        if n_qubits in completed:
            print(f"⏩ Skipping n={n_qubits} (already completed)")
            continue
        
        n_circuits = CONFIG["circuits_per_n"][n_qubits]
        print(f"\n  Qubits: {n_qubits} ({n_circuits} circuits)")
        
        # Family distribution
        families = []
        for family, ratio in CONFIG["circuit_families"].items():
            n_family = int(ratio * n_circuits)
            families.extend([family] * n_family)
        while len(families) < n_circuits:
            families.append("random_non_clifford")
        np.random.shuffle(families)
        
        block_data = []
        for i in tqdm(range(n_circuits), desc=f"n={n_qubits}"):
            family = families[i]
            depth = get_depth(n_qubits, family)
            seed = CONFIG["seed"] + i * 100 + n_qubits * 10000
            circuit = generate_circuit(n_qubits, family, depth, seed)
            low_counts, high_counts = collect_pair(circuit, CONFIG["low_shots"], CONFIG["high_shots"], noise_model, seed)
            ideal_counts = run_circuit(circuit, CONFIG["high_shots"], NoiseModel(), seed + 2000)
            low_bitstrings, low_probs = counts_to_tensor(low_counts, n_qubits)
            high_bitstrings, high_probs = counts_to_tensor(high_counts, n_qubits)
            ideal_bitstrings, ideal_probs = counts_to_tensor(ideal_counts, n_qubits)
            entry = {
                'n_qubits': n_qubits,
                'family': family,
                'depth': depth,
                'seed': seed,
                'low_shots': CONFIG["low_shots"],
                'high_shots': CONFIG["high_shots"],
                'low_bitstrings': low_bitstrings.numpy(),
                'low_probs': low_probs.numpy(),
                'high_bitstrings': high_bitstrings.numpy(),
                'high_probs': high_probs.numpy(),
                'ideal_bitstrings': ideal_bitstrings.numpy(),
                'ideal_probs': ideal_probs.numpy(),
            }
            block_data.append(entry)
            all_data.append(entry)
        
        # Save block
        save_block(block_data, n_qubits, output_dir)
        # Update checkpoint
        completed.append(n_qubits)
        save_checkpoint(completed)
        print(f"   ✅ Block {n_qubits} completed and checkpoint updated.")
    
    # After all blocks, merge into final HDF5
    print("\n🔄 Merging blocks into final dataset...")
    merge_blocks(output_dir, CONFIG["n_qubits"])
    # Save metadata
    metadata = {
        "experiment": "exp1_snd",
        "description": "SN-D validation: low-shot (100) → high-shot (100,000)",
        "config": CONFIG,
        "train_val_test_split": [0.7, 0.15, 0.15],
        "total_circuits": sum(CONFIG["circuits_per_n"].values()),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    # Remove checkpoint file
    if os.path.exists(CONFIG["checkpoint_file"]):
        os.remove(CONFIG["checkpoint_file"])
    print(f"\n✅ Dataset saved to {output_dir}")
    print(f"   Total circuits: {len(all_data)}")

def collect_pair(circuit, low_shots, high_shots, noise_model, seed):
    low_counts = run_circuit(circuit, low_shots, noise_model, seed)
    high_counts = run_circuit(circuit, high_shots, noise_model, seed + 1000)
    return low_counts, high_counts

if __name__ == "__main__":
    print("="*60)
    print("N2LN-QEM: Experiment 1 Dataset Generation (Auto-Save)")
    print("="*60)
    print(f"Qubits: {CONFIG['n_qubits']}")
    print(f"Low shots: {CONFIG['low_shots']}")
    print(f"High shots: {CONFIG['high_shots']}")
    print("Auto-save ENABLED: each qubit block saved separately.")
    print("="*60)
    generate_dataset()