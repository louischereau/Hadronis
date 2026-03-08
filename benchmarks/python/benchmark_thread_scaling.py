import argparse
import sys
import time
from typing import List, Sequence

from benchmark_single_molecule_latency import (
    LatencyStats,
    compute_latency_stats,
    _generate_random_system,
)


def benchmark_thread_scaling(
    weight_path: str,
    sizes: Sequence[int],
    thread_counts: Sequence[int],
    n_warmup: int,
    n_iters: int,
    cutoff: float,
    max_neighbors: int,
) -> List[LatencyStats]:
    try:
        import hadronis  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(f"hadronis is not importable: {exc}") from exc

    results: List[LatencyStats] = []

    for n_atoms in sizes:
        system = _generate_random_system(n_atoms, seed=n_atoms)
        z = system["Z"]
        r = system["R"]

        for n_threads in thread_counts:
            engine = hadronis.compile(
                weight_path,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                n_threads=n_threads,
            )

            # Warmup
            for _ in range(n_warmup):
                _ = engine.predict(z, r)

            durations: List[float] = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _ = engine.predict(z, r)
                t1 = time.perf_counter()
                durations.append(t1 - t0)

            backend_label = f"hadronis_t{n_threads}"
            stats = compute_latency_stats(backend_label, n_atoms, durations)
            results.append(stats)

    return results


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Hadronis steady-state latency as a function of "
            "number of threads and system size."
        )
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default="64,256,1024",
        help="Comma-separated list of atom counts to benchmark.",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated list of thread counts to benchmark.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per configuration.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=1000,
        help="Number of timed iterations per configuration.",
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

    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    thread_counts = [int(x) for x in args.threads.split(",") if x.strip()]

    results = benchmark_thread_scaling(
        weight_path=args.painn_weights,
        sizes=sizes,
        thread_counts=thread_counts,
        n_warmup=args.n_warmup,
        n_iters=args.n_iters,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
    )

    header = (
        f"{'backend':>12}  {'n_atoms':>7}  {'mean_ms':>8}  "
        f"{'p50_ms':>8}  {'p95_ms':>8}  {'p99_ms':>8}  {'max_ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(results, key=lambda r: (r.n_atoms, r.backend)):
        print(
            f"{row.backend:>12}  {row.n_atoms:7d}  "
            f"{row.mean_ms:8.3f}  {row.p50_ms:8.3f}  "
            f"{row.p95_ms:8.3f}  {row.p99_ms:8.3f}  "
            f"{row.max_ms:8.3f}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
