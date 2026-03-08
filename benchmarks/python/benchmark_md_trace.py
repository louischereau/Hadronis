import argparse
import platform
import sys
import time
from typing import List, Sequence

from benchmark_single_molecule_latency import (
    LatencyStats,
    compute_latency_stats,
    _generate_random_system,
)


def simple_integrator_step(r):
    """Placeholder for an MD integrator step.

    This applies a tiny random displacement to each atom so that
    positions change over time but remain bounded.
    """
    import numpy as np

    noise = np.random.normal(loc=0.0, scale=1e-3, size=r.shape).astype(r.dtype)
    return r + noise


def benchmark_md_trace(
    weight_path: str,
    n_atoms: int,
    n_steps: int,
    n_warmup: int,
    cutoff: float,
    max_neighbors: int,
    n_threads: int,
) -> LatencyStats:
    try:
        import hadronis  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(f"hadronis is not importable: {exc}") from exc

    engine = hadronis.compile(
        weight_path,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        n_threads=n_threads,
    )

    system = _generate_random_system(n_atoms, seed=0)
    z = system["Z"]
    r = system["R"]

    # Warmup: short trace to let caches / JITs settle.
    for _ in range(n_warmup):
        r = simple_integrator_step(r)
        _ = engine.predict(z, r)

    durations: List[float] = []
    for _ in range(n_steps):
        r = simple_integrator_step(r)
        t0 = time.perf_counter()
        _ = engine.predict(z, r)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    return compute_latency_stats("hadronis_md", n_atoms, durations)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Hadronis latency in an MD-style inner loop, "
            "tracking per-step latency over a long trace."
        )
    )

    parser.add_argument(
        "--n-atoms",
        type=int,
        default=256,
        help="Number of atoms in the simulated system.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10_000,
        help="Number of MD-like steps to run.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=100,
        help="Number of warmup steps before timing.",
    )
    parser.add_argument(
        "--painn-weights",
        type=str,
        default="painn.bin",
        help="Path to the PaiNN weight file for Hadronis.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Distance cutoff used for neighbor construction (Angstrom).",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=64,
        help="Maximum number of neighbors per atom for Hadronis.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=16,
        help="Number of CPU threads for Hadronis.",
    )

    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    # Print hardware / configuration header for reproducibility
    cpu_desc = platform.processor() or platform.uname().processor or "unknown"
    print(
        f"# CPU: {cpu_desc} | n_atoms: {args.n_atoms} | n_steps: {args.n_steps} | "
        f"n_warmup: {args.n_warmup} | cutoff: {args.cutoff} | "
        f"max_neighbors: {args.max_neighbors} | n_threads: {args.n_threads}",
        flush=True,
    )

    stats = benchmark_md_trace(
        weight_path=args.painn_weights,
        n_atoms=args.n_atoms,
        n_steps=args.n_steps,
        n_warmup=args.n_warmup,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        n_threads=args.n_threads,
    )

    header = (
        f"{'backend':>10}  {'n_atoms':>7}  {'mean_ms':>8}  "
        f"{'p50_ms':>8}  {'p95_ms':>8}  {'p99_ms':>8}  {'max_ms':>8}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{stats.backend:>10}  {stats.n_atoms:7d}  "
        f"{stats.mean_ms:8.3f}  {stats.p50_ms:8.3f}  "
        f"{stats.p95_ms:8.3f}  {stats.p99_ms:8.3f}  {stats.max_ms:8.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
