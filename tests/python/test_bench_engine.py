import numpy as np
import pytest

import hadronis


# Only enable these benchmarks when pytest-codspeed is installed (e.g. in CI
# performance runs). Local "uv run pytest tests/" will then skip them rather
# than failing due to the missing "benchmark" fixture.
pytest.importorskip("pytest_codspeed")


# These tests are intended for CodSpeed: they should represent
# realistic high-throughput workloads but stay fast enough for CI.
# They rely on the public Python API (hadronis.compile / Engine.predict)
# and avoid heavy assertions so timing is dominated by the core path.


def _make_random_batch(n_atoms: int, n_molecules: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Assign each atom to a molecule index in [0, n_molecules)
    batch = rng.integers(0, n_molecules, size=n_atoms, dtype=np.int32)

    # Atomic numbers roughly in the range of common organic elements
    atomic_numbers = rng.integers(1, 18, size=n_atoms, dtype=np.int32)

    # Random positions in a cube (Angstroms)
    positions = rng.normal(loc=0.0, scale=5.0, size=(n_atoms, 3)).astype(np.float32)

    return atomic_numbers, positions, batch


@pytest.mark.parametrize("n_atoms,n_molecules", [(10_000, 128)])
def test_large_batch_single_pass(benchmark, n_atoms: int, n_molecules: int):
    """Benchmark a single large batch inference call.

    This measures end-to-end latency for a realistic-sized batch
    (10k atoms across many molecules).
    """

    engine = hadronis.compile("dummy-weights.bin")
    atomic_numbers, positions, batch = _make_random_batch(n_atoms, n_molecules)

    def run():
        out = engine.predict(atomic_numbers, positions, batch)
        # Light sanity check so the benchmark still validates behavior.
        assert out.shape == (n_atoms,)
        return out

    benchmark(run)


@pytest.mark.parametrize("n_atoms,n_molecules", [(2_000, 32)])
def test_medium_batch_repeated_calls(benchmark, n_atoms: int, n_molecules: int):
    """Benchmark many medium-sized calls to capture throughput.

    This approximates serving multiple smaller batches in a production
    environment (e.g. batched RPCs).
    """

    engine = hadronis.compile("dummy-weights.bin")
    atomic_numbers, positions, batch = _make_random_batch(n_atoms, n_molecules, seed=1)

    def run():
        out = engine.predict(atomic_numbers, positions, batch)
        assert out.shape == (n_atoms,)
        return out

    benchmark(run)


@pytest.mark.parametrize("n_atoms,n_molecules", [(20_000, 256)])
def test_scaling_with_batch_size(benchmark, n_atoms: int, n_molecules: int):
    """Coarser benchmark at a larger scale to study scaling.

    Useful for checking that runtime grows roughly linearly with the
    number of atoms and to catch regressions in asymptotic behavior.
    """

    engine = hadronis.compile("dummy-weights.bin")
    atomic_numbers, positions, batch = _make_random_batch(n_atoms, n_molecules, seed=2)

    def run():
        out = engine.predict(atomic_numbers, positions, batch)
        assert out.shape == (n_atoms,)
        return out

    benchmark(run)
