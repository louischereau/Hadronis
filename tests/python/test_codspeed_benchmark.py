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

    # Random positions in a cube (Angstroms) with approximately
    # constant physical density as n_atoms grows. A spacing of
    # ~3.5 Å per atom side length gives a reasonable density
    # and keeps the average neighbor count O(1) so that the
    # graph builder does not run into quadratic memory usage.
    spacing = 3.5
    box = float(n_atoms) ** (1.0 / 3.0) * spacing
    positions = rng.uniform(0.0, box, size=(n_atoms, 3)).astype(np.float32)

    return atomic_numbers, positions


@pytest.mark.parametrize("n_atoms", [128])
def test_small_system(benchmark, n_atoms: int, known_weights_file):
    """Benchmark a single small-system inference call.

    Captures low-latency behavior for a modest-size molecule.
    """

    engine = hadronis.compile(known_weights_file)
    atomic_numbers, positions = _make_random_system(n_atoms, seed=42)

    def run():
        out = engine.predict(atomic_numbers, positions)
        assert isinstance(out, float)
        return out

    benchmark(run)


@pytest.mark.parametrize("n_atoms", [256])
def test_medium_system(benchmark, n_atoms: int, known_weights_file):
    """Benchmark many medium-sized calls to capture throughput.

    This approximates repeatedly evaluating a medium-sized system.
    """

    engine = hadronis.compile(known_weights_file)
    atomic_numbers, positions = _make_random_system(n_atoms, seed=1)

    def run():
        out = engine.predict(atomic_numbers, positions)
        assert isinstance(out, float)
        return out

    benchmark(run)
