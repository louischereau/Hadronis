import hadronis
import numpy as np
import pytest

# Only enable these benchmarks when pytest-codspeed is installed (e.g. in CI
# performance runs). Local "uv run pytest tests/" will then skip them rather
# than failing due to the missing "benchmark" fixture.
pytest.importorskip("pytest_codspeed")


# These tests are intended for CodSpeed: they should represent
# realistic workloads on large single systems but stay fast enough for CI.
# They rely on the public Python API (hadronis.compile / Engine.predict)
# and avoid heavy assertions so timing is dominated by the core path.


def _make_random_system(n_atoms: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Atomic numbers roughly in the range of common organic elements
    atomic_numbers = rng.integers(1, 18, size=n_atoms, dtype=np.int32)

    # Random positions in a cube (Angstroms)
    positions = rng.normal(loc=0.0, scale=5.0, size=(n_atoms, 3)).astype(np.float32)

    return atomic_numbers, positions


@pytest.mark.parametrize("n_atoms", [128])
def test_small_system_single_pass(benchmark, n_atoms: int):
    """Benchmark a single small-system inference call.

    Captures low-latency behavior for a modest-size molecule.
    """

    engine = hadronis.compile("dummy-weights.bin")
    atomic_numbers, positions = _make_random_system(n_atoms, seed=42)

    def run():
        out = engine.predict(atomic_numbers, positions)
        assert out.shape == (n_atoms,)
        return out

    benchmark(run)


@pytest.mark.parametrize("n_atoms", [10_000])
def test_large_system_single_pass(benchmark, n_atoms: int):
    """Benchmark a single large-system inference call.

    This measures end-to-end latency for a realistic-sized system
    (10k atoms in one configuration).
    """

    engine = hadronis.compile("dummy-weights.bin")
    atomic_numbers, positions = _make_random_system(n_atoms)

    def run():
        out = engine.predict(atomic_numbers, positions)
        # Light sanity check so the benchmark still validates behavior.
        assert out.shape == (n_atoms,)
        return out

    benchmark(run)


@pytest.mark.parametrize("n_atoms", [2_000])
def test_medium_system_repeated_calls(benchmark, n_atoms: int):
    """Benchmark many medium-sized calls to capture throughput.

    This approximates repeatedly evaluating a medium-sized system.
    """

    engine = hadronis.compile("dummy-weights.bin")
    atomic_numbers, positions = _make_random_system(n_atoms, seed=1)

    def run():
        out = engine.predict(atomic_numbers, positions)
        assert out.shape == (n_atoms,)
        return out

    benchmark(run)


@pytest.mark.parametrize("n_atoms", [20_000])
def test_scaling_with_system_size(benchmark, n_atoms: int):
    """Coarser benchmark at a larger scale to study scaling.

    Useful for checking that runtime grows roughly linearly with the
    number of atoms and to catch regressions in asymptotic behavior.
    """

    engine = hadronis.compile("dummy-weights.bin")
    atomic_numbers, positions = _make_random_system(n_atoms, seed=2)

    def run():
        out = engine.predict(atomic_numbers, positions)
        assert out.shape == (n_atoms,)
        return out

    benchmark(run)
