import os
import time

import numpy as np
from valence.core import Molecule, ValenceEngine


def generate_mock_data(n_molecules, atoms_per_mol, feature_dim=64):
    mols = []
    feats = []
    for _ in range(n_molecules):
        # Using a dense sphere-like distribution for more realistic neighbor counts
        pos = np.random.normal(0, 10, (atoms_per_mol, 3)).astype(np.float32)
        m = Molecule(atomic_numbers=[6] * atoms_per_mol, positions=pos.tolist())
        f = np.random.rand(atoms_per_mol, feature_dim).astype(np.float32)
        mols.append(m)
        feats.append(f)
    return mols, feats


def run_benchmarks():
    # Setup
    W_DIM = 64
    weights = np.random.rand(W_DIM, W_DIM).astype(np.float32)
    np.save("temp_weights.npy", weights)

    engine = ValenceEngine("temp_weights.npy")

    # --- Warm-up ---
    # Essential for Rayon thread pool initialization and CPU frequency scaling
    print("Warming up engine...")
    warm_mols, warm_feats = generate_mock_data(5, 500, W_DIM)
    for m, f in zip(warm_mols, warm_feats):
        _ = engine.run(m, f)

    # --- Phase 1: Latency (10k Atoms) ---
    print("\n--- Phase 1: Latency (Single 10k Atom Molecule) ---")
    m_lat, f_lat = generate_mock_data(1, 10000, W_DIM)

    # Measure multiple times for a stable average
    iters = 5
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = m_lat[0].predict(f_lat[0])
        latencies.append((time.perf_counter() - t0) * 1000)

    print(f"Mean Latency: {np.mean(latencies):.2f} ms (±{np.std(latencies):.2f} ms)")

    # --- Phase 2: Throughput (Batching) ---
    print("\n--- Phase 2: Throughput (100 Molecules x 500 Atoms) ---")
    m_batch, f_batch = generate_mock_data(100, 500, W_DIM)

    t0 = time.perf_counter()
    engine.predict_batch(m_batch, f_batch)
    t_total = (time.perf_counter() - t0) * 1000

    total_atoms = 100 * 500
    print(f"Total Time: {t_total:.2f} ms")
    print(f"Throughput: {total_atoms / (t_total / 1000):,.0f} atoms/sec")
    print(f"Molecules per second: {100 / (t_total / 1000):.1f}")

    # Cleanup
    if os.path.exists("temp_weights.npy"):
        os.remove("temp_weights.npy")


if __name__ == "__main__":
    run_benchmarks()
