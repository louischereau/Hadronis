import gc
import os
import time

import hadronis
import numpy as np
import psutil
import pytest

pytestmark = pytest.mark.no_codspeed


def _get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _make_large_system(batch_size: int = 64, atoms_per_mol: int = 512):
    n_atoms = batch_size * atoms_per_mol

    atomic_numbers = np.full(n_atoms, 6, dtype=np.int32)
    positions = np.random.rand(n_atoms, 3).astype(np.float32)

    return atomic_numbers, positions


def test_memory_growth_under_repeated_inference():
    """Smoke test for memory leaks under a heavy, repeated workload.

    This is a slimmed-down version of the old stress test:
    - Builds a large system of atoms.
    - Runs Engine.predict multiple times.
    - Tracks RSS before/after and flags large growth as a potential leak.
    """

    atomic_numbers, positions = _make_large_system(batch_size=64, atoms_per_mol=512)

    engine = hadronis.compile("dummy-weights.bin")

    initial_mem = _get_memory_mb()
    start_time = time.perf_counter()

    n_iters = 5
    for i in range(n_iters):
        out = engine.predict(atomic_numbers, positions)
        # Ensure result is used so it's not trivially optimized away
        assert out.shape[0] == atomic_numbers.shape[0]

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
    label = "[OK]" if leak < 50.0 else "[POTENTIAL LEAK]"
    print(f"[memory-growth] net_growth={leak:.2f} MB {label}")

    # Turn very large growth into a test failure so CI catches regressions.
    # Threshold is intentionally generous to avoid noise from allocator/OS.
    assert leak < 200.0, (
        f"[POTENTIAL LEAK] RSS grew by {leak:.2f} MB during repeated inference"
    )
