"""Generate noise-scaled dataset for Experiment 2: HN-E validation.

Creates measurements at multiple noise scales λ ∈ {1.0, 1.5, 2.0, 2.5, 3.0}
for n ∈ {4, 6, 8}. Circuits: Clifford + random non-Clifford + VQE + QAOA.
Noise: combined (depolarizing + amplitude damping + phase damping + readout).
Auto-save: each qubit block saved separately.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import yaml
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from qiskit import QuantumCircuit, transpile
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
    "circuits_per_n": {4: 2000, 6: 1500, 8: 1500},
    "circuit_families": {
        "clifford": 0.3,
        "random_non_clifford": 0.4,
        "vqe": 0.2,
        "qaoa": 0.1,
    },
    "shots": 1000,
    "noise_scales": [1.0, 1.5, 2.0, 2.5, 3.0],
    "noise": {
        "depolarizing_p": 0.01,
        "readout_p0": 0.02,
        "readout_p1": 0.02,
    },
    "seed": 42,
    "output_dir": "data/raw/exp2_hne",
    "checkpoint_file": "data/raw/exp2_hne/checkpoint.json",
}

# =============================================================================
# Circuit Generation
# =============================================================================

def generate_circuit(n_qubits: int, family: str, depth: int, seed: int) -> QuantumCircuit:
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
    if family in ["clifford", "random_non_clifford"]:
        return min(10, 3 * n_qubits)
    elif family in ["vqe", "qaoa"]:
        return 2
    else:
        return 5

def get_noise_model(config: dict) -> NoiseModel:
    noise_cfg = config['noise']
    return combined_noise(
        p_dep=noise_cfg['depolarizing_p'],
        p0=noise_cfg['readout_p0'],
        p1=noise_cfg['readout_p1'],
    )

def fold_gates(circuit: QuantumCircuit, factor: float) -> QuantumCircuit:
    if factor <= 1.0:
        return circuit
    return circuit

def run_circuit(circuit: QuantumCircuit, shots: int, noise_model: NoiseModel, seed: int) -> Dict[str, int]:
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

def collect_noise_scaled(circuit: QuantumCircuit, noise_scales: List[float], shots: int, noise_model: NoiseModel, seed: int) -> Dict[float, Dict[str, int]]:
    results = {}
    for lam in noise_scales:
        if lam > 1.0:
            folded_circuit = fold_gates(circuit, lam)
        else:
            folded_circuit = circuit
        counts = run_circuit(folded_circuit, shots, noise_model, seed + int(lam * 1000))
        results[lam] = counts
    return results

# =============================================================================
# Auto-Save Functions
# =============================================================================

def load_checkpoint():
    ckpt_path = CONFIG["checkpoint_file"]
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            return json.load(f)
    return {"completed_qubits": []}

def save_checkpoint(completed_qubits):
    ckpt_path = CONFIG["checkpoint_file"]
    with open(ckpt_path, 'w') as f:
        json.dump({"completed_qubits": completed_qubits}, f)

def save_block(data, n_qubits, output_dir):
    block_file = os.path.join(output_dir, f"block_{n_qubits}.h5")
    with h5py.File(block_file, 'w') as f:
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

def merge_blocks(output_dir):
    all_data = []
    for nq in CONFIG["n_qubits"]:
        block_file = os.path.join(output_dir, f"block_{nq}.h5")
        if not os.path.exists(block_file):
            continue
        with h5py.File(block_file, 'r') as f:
            n_samples = len(f['n_qubits'])
            for i in range(n_samples):
                entry = {}
                for key in f.keys():
                    if key.endswith('_shape'):
                        continue
                    val = f[key][i]
                    if isinstance(val, np.ndarray) and val.dtype == np.object_:
                        shape_key = key + "_shape"
                        if shape_key in f:
                            shape = f[shape_key][i]
                            val = val.reshape(shape)
                    entry[key] = val
                all_data.append(entry)
    np.random.seed(42)
    n = len(all_data)
    indices = np.random.permutation(n)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    splits = {"train": indices[:train_end], "val": indices[train_end:val_end], "test": indices[val_end:]}
    for split_name, split_indices in splits.items():
        if len(split_indices) == 0:
            continue
        filepath = os.path.join(output_dir, f"exp2_hne_{split_name}.h5")
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
    for nq in CONFIG["n_qubits"]:
        block_file = os.path.join(output_dir, f"block_{nq}.h5")
        if os.path.exists(block_file):
            os.remove(block_file)

# =============================================================================
# Main Dataset Generation
# =============================================================================

def generate_dataset(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    np.random.seed(config['seed'])
    output_dir = Path(config['data']['data_dir'])
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = load_checkpoint()
    completed = checkpoint.get("completed_qubits", [])

    noise_model = get_noise_model(config)
    noise_scales = config['data']['noise_scales']
    shots = config['data']['shots']

    print("=" * 60)
    print("N2LN-QEM: Experiment 2 Dataset Generation (Auto-Save)")
    print("=" * 60)
    print(f"Qubits: {config['data']['n_qubits']}")
    print(f"Noise scales: {noise_scales}")
    print(f"Shots per scale: {shots}")
    print("Auto-save ENABLED: each qubit block saved separately.")
    print("=" * 60)

    for n_qubits in config['data']['n_qubits']:
        if n_qubits in completed:
            print(f"⏩ Skipping n={n_qubits} (already completed)")
            continue

        n_circuits = config['data']['circuits_per_n'][n_qubits]
        print(f"\n  Qubits: {n_qubits} ({n_circuits} circuits)")

        families = []
        for family, ratio in config['circuit_families'].items():
            n_family = int(ratio * n_circuits)
            families.extend([family] * n_family)
        while len(families) < n_circuits:
            families.append("random_non_clifford")
        np.random.shuffle(families)

        block_data = []
        for i in tqdm(range(n_circuits), desc=f"n={n_qubits}"):
            family = families[i]
            depth = get_depth(n_qubits, family)
            seed = config['seed'] + i * 100 + n_qubits * 10000
            circuit = generate_circuit(n_qubits, family, depth, seed)
            scale_results = collect_noise_scaled(circuit, noise_scales, shots, noise_model, seed)
            entry = {'n_qubits': n_qubits, 'family': family, 'depth': depth, 'seed': seed, 'shots': shots}
            for lam, counts in scale_results.items():
                bitstrings, probs = counts_to_tensor(counts, n_qubits)
                entry[f'bitstrings_{lam}'] = bitstrings.numpy()
                entry[f'probs_{lam}'] = probs.numpy()
            block_data.append(entry)

        save_block(block_data, n_qubits, output_dir)
        completed.append(n_qubits)
        save_checkpoint(completed)
        print(f"   ✅ Block {n_qubits} completed and checkpoint updated.")

    print("\n🔄 Merging blocks into final dataset...")
    merge_blocks(output_dir)
    if os.path.exists(CONFIG["checkpoint_file"]):
        os.remove(CONFIG["checkpoint_file"])
    print(f"\n✅ Dataset saved to {output_dir}")
    print(f"   Total circuits: {sum(config['data']['circuits_per_n'].values())}")
    print(f"   Noise scales: {noise_scales}")

if __name__ == "__main__":
    config_path = "experiments/exp2_hne/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    generate_dataset(config_path)