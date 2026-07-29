"""Generate large-scale dataset for Experiment 4 (10-30 qubits).

Uses MPS simulator for circuits up to 30 qubits.
Low-entanglement circuits only (shallow depth, 1D connectivity).
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
from typing import Dict, List
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.data.circuits.clifford import generate_clifford
from src.data.circuits.random_layer import generate_random_non_clifford
from src.data.circuits.vqe import generate_vqe
from src.data.circuits.qaoa import generate_qaoa
from src.data.noise_models import combined_noise
from src.data.encoding import counts_to_tensor
from src.data.simulators.mps import MPSSimulator


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
        return min(5, 2 * n_qubits)
    elif family in ["vqe", "qaoa"]:
        return 2
    else:
        return 3


def generate_dataset(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    np.random.seed(config['seed'])
    output_dir = Path(config['data']['data_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    mps = MPSSimulator(
        max_bond_dimension=config['mps']['max_bond_dimension'],
        enable_truncation=config['mps']['enable_truncation'],
    )
    
    all_data = []
    total_circuits = sum(config['data']['circuits_per_n'].values())
    
    print("=" * 60)
    print("N2LN-QEM: Experiment 4 Dataset Generation (MPS)")
    print("=" * 60)
    print(f"Qubits: {config['data']['n_qubits']}")
    print(f"Total circuits: {total_circuits}")
    print("=" * 60)
    
    for n_qubits in config['data']['n_qubits']:
        n_circuits = config['data']['circuits_per_n'][n_qubits]
        print(f"\n  Qubits: {n_qubits} ({n_circuits} circuits)")
        
        families = []
        for family, ratio in config['circuit_families'].items():
            n_family = int(ratio * n_circuits)
            families.extend([family] * n_family)
        while len(families) < n_circuits:
            families.append("random_non_clifford")
        np.random.shuffle(families)
        
        for i in tqdm(range(n_circuits), desc=f"n={n_qubits}"):
            family = families[i]
            depth = get_depth(n_qubits, family)
            seed = config['seed'] + i * 100 + n_qubits * 10000
            
            circuit = generate_circuit(n_qubits, family, depth, seed)
            counts = mps.run(circuit, shots=config['data']['shots'], seed=seed)
            bitstrings, probs = counts_to_tensor(counts, n_qubits)
            
            all_data.append({
                'n_qubits': n_qubits,
                'family': family,
                'depth': depth,
                'seed': seed,
                'bitstrings': bitstrings.numpy(),
                'probs': probs.numpy(),
            })
    
    save_dataset(all_data, output_dir, config)
    print(f"\n✅ Dataset saved to {output_dir}")
    print(f"   Total circuits: {len(all_data)}")


def save_dataset(data: List[Dict], output_dir: Path, config: dict):
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
        
        filepath = output_dir / f'exp4_scale_{split_name}.h5'
        with h5py.File(filepath, 'w') as f:
            for key in data[0].keys():
                values = [data[idx][key] for idx in split_indices]
                
                if isinstance(values[0], (int, float, np.integer, np.floating)):
                    f.create_dataset(key, data=np.array(values))
                elif isinstance(values[0], str):
                    dt = h5py.special_dtype(vlen=str)
                    ds = f.create_dataset(key, (len(split_indices),), dtype=dt)
                    for j, v in enumerate(values):
                        ds[j] = v
                else:
                    if 'bitstrings' in key:
                        dtype = h5py.special_dtype(vlen=np.int64)
                        ds = f.create_dataset(key, (len(split_indices),), dtype=dtype)
                        shape_ds = f.create_dataset(f"{key}_shape", (len(split_indices), 2), dtype=np.int64)
                        for j, idx in enumerate(split_indices):
                            arr = data[idx][key]
                            ds[j] = arr.flatten().astype(np.int64)
                            shape_ds[j] = arr.shape
                    elif 'probs' in key:
                        dtype = h5py.special_dtype(vlen=np.float32)
                        ds = f.create_dataset(key, (len(split_indices),), dtype=dtype)
                        shape_ds = f.create_dataset(f"{key}_shape", (len(split_indices), 1), dtype=np.int64)
                        for j, idx in enumerate(split_indices):
                            arr = data[idx][key]
                            ds[j] = arr.astype(np.float32)
                            shape_ds[j] = arr.shape[0]
        
        print(f"   Saved {split_name}: {len(split_indices)} circuits")


if __name__ == "__main__":
    config_path = "experiments/exp4_scale/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    generate_dataset(config_path)
