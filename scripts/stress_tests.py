# ruff: noqa: I001
import gc
import os
import time

import numpy as np
import psutil

from memory_tracker import track_memory
from python import hadronis
from safetensors.numpy import save


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


@track_memory
def run_engine(
    engine: hadronis.HadronisEngine, atomic_numbers, positions, mol_ptrs, features
):
    return engine.run_batch_inference(
        atomic_numbers,
        positions,
        mol_ptrs,
        features,
        cutoff=1.2,
        k=8,
    )


def run_stress_test():
    print(f"--- Starting Stress Test [PID: {os.getpid()}] ---")
    initial_mem = get_memory_mb()
    print(f"Initial Memory: {initial_mem:.2f} MB")

    # 1. Initialize Engine
    # Save random weights to a temporary safetensors file
    weights = np.random.rand(64, 64).astype(np.float32)
    weights_path = "./tmp_stress_weights.safetensors"
    save({"weights": weights}, weights_path)
    engine = hadronis.HadronisEngine(weights_path)

    # 2. Generate a large batch of molecules
    batch_size = 100
    atoms_per_mol = 1000
    print(f"Creating batch: {batch_size} molecules, {atoms_per_mol} atoms each...")

    # Concatenate all molecules into batch arrays
    atomic_numbers = np.repeat(6, batch_size * atoms_per_mol).astype(np.int32)
    positions = np.random.rand(batch_size * atoms_per_mol, 3).astype(np.float32)
    features = np.random.rand(batch_size * atoms_per_mol, 64).astype(np.float32)
    mol_ptrs = np.arange(
        0, batch_size * atoms_per_mol + 1, atoms_per_mol, dtype=np.int32
    )

    # 4. Execution Loop
    print("Running inference...")
    start_time = time.perf_counter()

    # We run this multiple times to see if memory keeps climbing (a leak)
    for i in range(5):
        results = run_engine(engine, atomic_numbers, positions, mol_ptrs, features)
        gc.collect()  # Force Python to clean up finished result objects
        current_mem = get_memory_mb()
        print(
            f"  Iteration {i + 1}: Mem = {current_mem:.2f} MB | Results shape = {results.shape}"
        )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print("-" * 30)
    print(f"Total Throughput: {batch_size * 5 / total_time:.2f} molecules/sec")

    # 4. Cleanup & Final Leak Check
    del results
    time.sleep(1)  # Give OS a second to reclaim

    final_mem = get_memory_mb()
    leak = final_mem - initial_mem
    print(f"Final Memory: {final_mem:.2f} MB")
    print(f"Net Growth: {leak:.2f} MB {'[OK]' if leak < 50 else '[POTENTIAL LEAK]'}")

    # 5. Cleanup temporary file
    if os.path.exists(weights_path):
        os.remove(weights_path)


if __name__ == "__main__":
    run_stress_test()
