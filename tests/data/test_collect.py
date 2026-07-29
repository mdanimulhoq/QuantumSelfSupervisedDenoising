"""Tests for data collection protocol (TDD §5.2)."""

import os
import tempfile

import numpy as np

from src.data.circuits import generate_random
from src.data.collect import (
    collect_shot_pairs,
    collect_noise_scaled,
    save_to_hdf5,
    load_from_hdf5,
    generate_dataset,
)
from src.data.noise_models import depolarizing


def test_collect_shot_pairs():
    circuit = generate_random(4, 10, seed=42)
    data = collect_shot_pairs(
        circuit, "test_001", low_shots=100, high_shots=1000, seed=42
    )
    assert data["circuit_id"] == "test_001"
    assert data["n_qubits"] == 4
    assert data["low_shot"]["shots"] == 100
    assert data["high_shot"]["shots"] == 1000
    assert sum(data["low_shot"]["counts"].values()) == 100


def test_collect_noise_scaled():
    circuit = generate_random(4, 10, seed=42)
    data = collect_noise_scaled(
        circuit, "test_002", shots=500, scale_factors=[1.0, 2.0], seed=42
    )
    assert len(data["noise_scaled"]) == 2
    assert 1.0 in data["noise_scaled"]
    assert 2.0 in data["noise_scaled"]


def test_save_and_load_hdf5():
    circuit = generate_random(4, 10, seed=42)
    data = collect_shot_pairs(
        circuit, "test_003", low_shots=100, high_shots=1000, seed=42
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.h5")
        save_to_hdf5(data, filepath)
        assert os.path.exists(filepath)

        loaded = load_from_hdf5(filepath)
        assert loaded["circuit_id"] == "test_003"
        assert loaded["n_qubits"] == 4
        assert sum(loaded["low_shot"]["counts"].values()) == 100


def test_generate_dataset():
    nm = depolarizing(p_gate=0.01, p_readout=0.02)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepaths = generate_dataset(
            generate_random,
            n_circuits=5,
            n_qubits=2,
            depth=5,
            noise_model=nm,
            output_dir=tmpdir,
            prefix="test_ds",
            seed=42,
        )
        assert len(filepaths) == 5
        for fp in filepaths:
            assert os.path.exists(fp)
            loaded = load_from_hdf5(fp)
            assert loaded["n_qubits"] == 2
