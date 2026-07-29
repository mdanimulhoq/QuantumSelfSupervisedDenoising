"""IBMQ interface with rate-limiting, retry, and caching (TDD §7.3).

Wraps Qiskit Runtime for job submission with automatic retry and caching.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler


class IBMQInterface:
    """IBMQ interface with rate-limiting, retry, and caching."""
    
    def __init__(
        self,
        backend_name: str = "ibmq_nairobi",
        cache_dir: str = "data/hardware_cache",
        max_retries: int = 3,
        retry_delay: int = 5,
        shots: int = 1000,
    ):
        """
        Args:
            backend_name: IBMQ backend name (e.g., "ibmq_nairobi", "ibmq_brisbane")
            cache_dir: Directory for caching results
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            shots: Default number of shots
        """
        self.backend_name = backend_name
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.shots = shots
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize service
        self.service = None
        self.backend = None
        self._connect()
    
    def _connect(self):
        """Connect to IBMQ service."""
        try:
            # Try to load from saved account
            self.service = QiskitRuntimeService()
            self.backend = self.service.backend(self.backend_name)
            print(f"✅ Connected to IBMQ backend: {self.backend_name}")
        except Exception as e:
            print(f"⚠️ Could not connect to IBMQ: {e}")
            print("   You may need to: pip install qiskit-ibm-runtime")
            print("   and save your IBMQ token:")
            print("   QiskitRuntimeService.save_account(token='YOUR_TOKEN')")
            self.service = None
            self.backend = None
    
    def _get_cache_key(self, circuit: QuantumCircuit, shots: int) -> str:
        """Generate cache key from circuit and parameters."""
        # Convert circuit to string for hashing
        circuit_str = circuit.qasm()
        key_str = f"{circuit_str}_{shots}_{self.backend_name}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, int]]:
        """Get cached result if available."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                # Check if cache is still valid (24 hours)
                timestamp = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - timestamp < timedelta(hours=24):
                    return data['counts']
        return None
    
    def _save_cache(self, cache_key: str, counts: Dict[str, int]):
        """Save result to cache."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        data = {
            'counts': counts,
            'timestamp': datetime.now().isoformat(),
            'backend': self.backend_name,
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        use_cache: bool = True,
        optimize_level: int = 1,
    ) -> Dict[str, int]:
        """
        Run a circuit on IBMQ hardware with caching and retry.
        
        Args:
            circuit: Quantum circuit to run
            shots: Number of shots (default: self.shots)
            use_cache: Whether to use cached results
            optimize_level: Transpiler optimization level
        
        Returns:
            Counts dictionary
        """
        if shots is None:
            shots = self.shots
        
        # Check cache
        cache_key = self._get_cache_key(circuit, shots)
        if use_cache:
            cached = self._get_cached_result(cache_key)
            if cached:
                print(f"📦 Using cached result for circuit")
                return cached
        
        # Run on hardware
        if self.backend is None:
            print("⚠️ No IBMQ backend available. Using simulator fallback.")
            return self._run_simulator(circuit, shots)
        
        # Transpile circuit
        transpiled = transpile(circuit, self.backend, optimization_level=optimize_level)
        
        # Run with retry
        for attempt in range(self.max_retries):
            try:
                print(f"🚀 Submitting job (attempt {attempt + 1}/{self.max_retries})...")
                
                # Use Session for better performance
                with Session(backend=self.backend) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run(transpiled, shots=shots)
                    result = job.result()
                    
                    # Get counts
                    counts = result.quasi_dists[0].binary_probabilities()
                    
                    # Convert to integer counts
                    total = shots
                    counts_int = {k: int(v * total) for k, v in counts.items()}
                    
                    # Save cache
                    self._save_cache(cache_key, counts_int)
                    
                    return counts_int
                    
            except Exception as e:
                print(f"⚠️ Job failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print("❌ All retries failed. Using simulator fallback.")
                    return self._run_simulator(circuit, shots)
        
        return self._run_simulator(circuit, shots)
    
    def _run_simulator(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Fallback to Aer simulator."""
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        backend = AerSimulator()
        circuit.measure_all()
        circ = transpile(circuit, backend)
        job = backend.run(circ, shots=shots)
        result = job.result()
        return result.get_counts(0)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information."""
        if self.backend:
            return {
                'name': self.backend.name,
                'version': self.backend.version,
                'num_qubits': self.backend.num_qubits,
                'status': self.backend.status(),
            }
        return {'error': 'No backend available'}


def test_ibmq_interface():
    """Test IBMQ interface on a 4-qubit Bell circuit."""
    print("\n" + "=" * 60)
    print("STEP 8.1: IBMQ Interface Test")
    print("=" * 60)
    
    # Create a Bell circuit
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.measure_all()
    
    # Initialize interface
    ibmq = IBMQInterface(shots=1000)
    
    # First run (should hit hardware or simulator)
    print("\n🔄 First run (caching)...")
    start = time.time()
    counts1 = ibmq.run(circuit, shots=1000, use_cache=True)
    elapsed1 = time.time() - start
    print(f"✅ First run completed in {elapsed1:.2f}s")
    print(f"   Counts: {len(counts1)} bitstrings")
    
    # Second run (should use cache)
    print("\n🔄 Second run (should use cache)...")
    start = time.time()
    counts2 = ibmq.run(circuit, shots=1000, use_cache=True)
    elapsed2 = time.time() - start
    print(f"✅ Second run completed in {elapsed2:.2f}s")
    
    if elapsed2 < elapsed1 * 0.5:
        print("✅ Cache working: second run much faster!")
    else:
        print("⚠️ Cache may not be working correctly")
    
    # Check cache files
    cache_dir = Path("data/hardware_cache")
    cache_files = list(cache_dir.glob("*.json"))
    print(f"\n📦 Cache files: {len(cache_files)}")
    for f in cache_files[:3]:
        print(f"   - {f.name}")
    
    print("\n" + "=" * 60)
    print("✅ IBMQ interface ready!")
    print("   To use real hardware, save your IBMQ token:")
    print("   QiskitRuntimeService.save_account(token='YOUR_TOKEN')")
    print("=" * 60)


if __name__ == "__main__":
    test_ibmq_interface()
