"""ZNE (Zero-Noise Extrapolation) baseline for Experiment 2.

Implements Richardson extrapolation using measurements at multiple noise scales.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
import yaml
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit


class ZNE:
    """Zero-Noise Extrapolation baseline."""
    
    def __init__(self, noise_scales: List[float], extrapolation_order: int = 1):
        self.noise_scales = noise_scales
        self.extrapolation_order = extrapolation_order
    
    def _align_probs(self, probs_list: List[np.ndarray]) -> List[np.ndarray]:
        """Align all probability vectors to the same length."""
        min_len = min(len(p) for p in probs_list)
        return [p[:min_len] for p in probs_list]
    
    def fit_and_extrapolate(self, measurements: Dict[float, np.ndarray]) -> np.ndarray:
        """Fit measurements at different noise scales and extrapolate to λ=0."""
        scales = sorted(self.noise_scales)
        fit_scales = [s for s in scales if s <= max(scales)]
        
        # Get measurements for each scale
        probs_list = [measurements[s] for s in fit_scales]
        
        # Align all to same length
        probs_list = self._align_probs(probs_list)
        
        if self.extrapolation_order == 1:
            return self._linear_extrapolation(fit_scales, probs_list)
        elif self.extrapolation_order == 2:
            return self._quadratic_extrapolation(fit_scales, probs_list)
        else:
            return self._richardson_extrapolation(fit_scales, probs_list)
    
    def _linear_extrapolation(self, scales: List[float], probs_list: List[np.ndarray]) -> np.ndarray:
        """Linear extrapolation using first and last points."""
        if len(probs_list) < 2:
            return probs_list[0]
        
        p1 = probs_list[0]
        p2 = probs_list[-1]
        λ1 = scales[0]
        λ2 = scales[-1]
        
        if λ2 == λ1:
            return p1
        
        slope = (p2 - p1) / (λ2 - λ1)
        p0 = p1 - slope * λ1
        return np.clip(p0, 0, 1)
    
    def _quadratic_extrapolation(self, scales: List[float], probs_list: List[np.ndarray]) -> np.ndarray:
        """Quadratic extrapolation using three points."""
        if len(probs_list) < 3:
            return self._linear_extrapolation(scales, probs_list)
        
        λ1, λ2, λ3 = scales[0], scales[1], scales[2]
        p1, p2, p3 = probs_list[0], probs_list[1], probs_list[2]
        
        A = np.array([
            [λ1**2, λ1, 1],
            [λ2**2, λ2, 1],
            [λ3**2, λ3, 1]
        ])
        
        try:
            # Solve for each element
            p0 = []
            for i in range(len(p1)):
                b = np.array([p1[i], p2[i], p3[i]])
                coeffs = np.linalg.solve(A, b)
                p0.append(coeffs[2])  # Constant term
            return np.clip(np.array(p0), 0, 1)
        except np.linalg.LinAlgError:
            return self._linear_extrapolation(scales, probs_list)
    
    def _richardson_extrapolation(self, scales: List[float], probs_list: List[np.ndarray]) -> np.ndarray:
        """Richardson extrapolation using all points."""
        if len(probs_list) < 2:
            return probs_list[0]
        
        scales = np.array(scales)
        probs = np.array(probs_list)
        degree = min(self.extrapolation_order, len(scales) - 1)
        
        if degree == 0:
            return probs[0]
        
        # Fit polynomial for each index
        p0 = []
        for i in range(probs.shape[1]):
            coeffs = np.polyfit(scales, probs[:, i], degree)
            p0.append(coeffs[-1])
        return np.clip(np.array(p0), 0, 1)


def compute_tvd(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute TVD between two distributions."""
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
    return 0.5 * np.abs(pred - target).sum()


def compute_zne_baseline(config_path: str):
    """Compute ZNE baseline on test set."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Data paths
    data_dir = Path(config['data']['data_dir'])
    test_path = data_dir / 'exp2_hne_test.h5'
    
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        return None
    
    # Load test data
    with h5py.File(test_path, 'r') as f:
        n_samples = len(f['n_qubits'])
        noise_scales = json.loads(f.attrs['noise_scales'])
        
        all_raw_results = []
        all_zne_results = []
        
        for i in range(n_samples):
            # Load measurements at all noise scales
            measurements = {}
            for lam in noise_scales:
                probs = f[f'probs_{lam}'][i]
                probs = probs.astype(np.float32)
                measurements[lam] = probs
            
            # Load low-noise target (λ=1.0) as reference
            target_probs = measurements[1.0]
            
            # Raw high-noise (λ=3.0)
            high_noise_probs = measurements[3.0]
            
            # ZNE extrapolation
            zne = ZNE(noise_scales, extrapolation_order=1)
            zne_probs = zne.fit_and_extrapolate(measurements)
            
            # Compute TVD
            raw_tvd = compute_tvd(high_noise_probs, target_probs)
            zne_tvd = compute_tvd(zne_probs, target_probs)
            
            all_raw_results.append(raw_tvd)
            all_zne_results.append(zne_tvd)
    
    # Average results
    avg_raw_tvd = np.mean(all_raw_results)
    avg_zne_tvd = np.mean(all_zne_results)
    
    print("\n📊 ZNE Baseline Results:")
    print("-" * 50)
    print(f"Raw (λ=3.0) TVD: {avg_raw_tvd:.4f}")
    print(f"ZNE TVD:          {avg_zne_tvd:.4f}")
    improvement = ((avg_raw_tvd - avg_zne_tvd) / avg_raw_tvd * 100) if avg_raw_tvd > 0 else 0
    print(f"Improvement:      {improvement:.1f}%")
    print("-" * 50)
    
    return {
        'raw_tvd': avg_raw_tvd,
        'zne_tvd': avg_zne_tvd,
        'improvement': improvement,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    config_path = "experiments/exp2_hne/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    results = compute_zne_baseline(config_path)
    
    if results:
        print(f"\n✅ ZNE baseline computed!")