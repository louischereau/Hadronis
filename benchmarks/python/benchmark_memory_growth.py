import argparse
import gc
import os
import time

import hadronis
import numpy as np
import psutil


def _get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _make_large_system(n_atoms: int = 128):
    atomic_numbers = np.full(n_atoms, 6, dtype=np.int32)
    positions = np.random.rand(n_atoms, 3).astype(np.float32)
    return atomic_numbers, positions


def run_memory_growth_benchmark(n_atoms: int = 128, n_iters: int = 5) -> None:
    atomic_numbers, positions = _make_large_system(n_atoms=n_atoms)

    print(f"[memory-growth] n_atoms={n_atoms} n_iters={n_iters}")
    engine = hadronis.compile("benchmarks/python/known-weights.bin")

    # Optional warmup to trigger one-time allocations before measuring.
    _ = engine.predict(atomic_numbers, positions)
    gc.collect()

    initial_mem = _get_memory_mb()
    start_time = time.perf_counter()

    for i in range(n_iters):
        out = engine.predict(atomic_numbers, positions)
        print(f"[memory-growth] output: {out} (type: {type(out)})")

        gc.collect()
        current_mem = _get_memory_mb()
        print(
            f"[memory-growth] iteration={i + 1} mem={current_mem:.2f} MB "
            f"delta={current_mem - initial_mem:+.2f} MB"
        )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    final_mem = _get_memory_mb()
    leak = final_mem - initial_mem

    print("-" * 40)
    print(f"[memory-growth] total_time={total_time:.3f}s")
    print(f"[memory-growth] net_growth={leak:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hadronis memory growth benchmark")
    parser.add_argument(
        "--n-atoms", type=int, default=128, help="Number of atoms in the test system"
    )
    parser.add_argument(
        "--n-iters", type=int, default=5, help="Number of repeated inference iterations"
    )
    args = parser.parse_args()

    run_memory_growth_benchmark(n_atoms=args.n_atoms, n_iters=args.n_iters)
