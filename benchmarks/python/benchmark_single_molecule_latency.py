import argparse
import csv
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class LatencyStats:
    backend: str
    n_atoms: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    n_iters: int


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def compute_latency_stats(backend: str, n_atoms: int, durations_s: Sequence[float]) -> LatencyStats:
    arr = np.asarray(durations_s, dtype=np.float64)
    arr_ms = arr * 1e3
    return LatencyStats(
        backend=backend,
        n_atoms=n_atoms,
        mean_ms=float(arr_ms.mean()),
        std_ms=float(arr_ms.std(ddof=0)),
        min_ms=float(arr_ms.min()),
        p50_ms=_percentile(arr_ms, 50.0),
        p90_ms=_percentile(arr_ms, 90.0),
        p95_ms=_percentile(arr_ms, 95.0),
        p99_ms=_percentile(arr_ms, 99.0),
        max_ms=float(arr_ms.max()),
        n_iters=int(arr_ms.shape[0]),
    )


def _generate_random_system(n_atoms: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Atomic numbers roughly in the range of common organic elements
    z = rng.integers(1, 18, size=n_atoms, dtype=np.int32)

    # Random positions in a cube (Angstroms)
    r = rng.normal(loc=0.0, scale=5.0, size=(n_atoms, 3)).astype(np.float32)

    return {"Z": z, "R": r}


def benchmark_hadronis(
    weight_path: str,
    sizes: Sequence[int],
    n_warmup: int,
    n_iters: int,
    cutoff: float,
    max_neighbors: int,
    n_threads: int,
    include_cold: bool = True,
) -> List[LatencyStats]:
    try:
        import hadronis  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(f"hadronis is not importable: {exc}") from exc

    results: List[LatencyStats] = []

    # Optional cold-start measurement: compile + first predict for the
    # smallest system size. This approximates "time to first useful
    # prediction" for a typical configuration.
    if include_cold and sizes:
        cold_n_atoms = int(sizes[0])
        cold_system = _generate_random_system(cold_n_atoms, seed=0)
        cold_z = cold_system["Z"]
        cold_r = cold_system["R"]

        t0 = time.perf_counter()
        cold_engine = hadronis.compile(
            weight_path,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            n_threads=n_threads,
        )
        _ = cold_engine.predict(cold_z, cold_r)
        t1 = time.perf_counter()

        cold_stats = compute_latency_stats("hadronis_cold", cold_n_atoms, [t1 - t0])
        results.append(cold_stats)

    # Steady-state benchmark: reuse a single engine across sizes.
    engine = hadronis.compile(
        weight_path,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        n_threads=n_threads,
    )

    for i, n_atoms in enumerate(sizes):
        system = _generate_random_system(n_atoms, seed=i)
        z = system["Z"]
        r = system["R"]

        # Warmup
        for _ in range(n_warmup):
            _ = engine.predict(z, r)

        durations: List[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = engine.predict(z, r)
            t1 = time.perf_counter()
            durations.append(t1 - t0)

        stats = compute_latency_stats("hadronis", n_atoms, durations)
        results.append(stats)

    return results


def benchmark_pytorch_painn(
    sizes: Sequence[int],
    n_warmup: int,
    n_iters: int,
    cutoff: float,
    device_str: str,
    include_cold: bool = True,
) -> List[LatencyStats]:
    """Benchmark a minimal PaiNN-style PyTorch model on the same systems.

    This uses the reference implementation in `painn_pyg.PaiNNModel` and
    runs a steady-state latency benchmark comparable to `benchmark_hadronis`.
    """

    try:
        import torch  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "PyTorch is required for the PyTorch PaiNN baseline. "
            "Install torch/torch_geometric/torch_scatter/torch_cluster first."
        ) from exc

    try:
        from painn_pyg import (  # type: ignore[import]
            PaiNNConfig,
            PaiNNModel,
            build_single_molecule_data,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "painn_pyg module is not importable. Make sure benchmarks/python "
            "is on PYTHONPATH and optional PyG dependencies are installed."
        ) from exc

    device = torch.device(device_str)
    model = PaiNNModel(PaiNNConfig(cutoff=cutoff)).to(device)
    model.eval()

    results: List[LatencyStats] = []

    # Optional cold-start measurement: model construction + first forward
    # pass for the smallest system size.
    if include_cold and sizes:
        cold_n_atoms = int(sizes[0])
        cold_system = _generate_random_system(cold_n_atoms, seed=10_000)
        cold_z_np = cold_system["Z"]
        cold_r_np = cold_system["R"]

        cold_z = torch.from_numpy(cold_z_np.astype(np.int64)).to(device)
        cold_pos = torch.from_numpy(cold_r_np.astype(np.float32)).to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            cold_data = build_single_molecule_data(
                cold_z, cold_pos, cutoff=cutoff, device=device
            )
            _ = model(cold_data)
            t1 = time.perf_counter()

        cold_stats = compute_latency_stats("painn_pytorch_cold", cold_n_atoms, [t1 - t0])
        results.append(cold_stats)

    for i, n_atoms in enumerate(sizes):
        system = _generate_random_system(n_atoms, seed=10_000 + i)
        z_np = system["Z"]
        r_np = system["R"]

        z = torch.from_numpy(z_np.astype(np.int64)).to(device)
        pos = torch.from_numpy(r_np.astype(np.float32)).to(device)

        # Warmup: rebuild graph + forward each time to mirror Hadronis
        with torch.no_grad():
            for _ in range(n_warmup):
                data = build_single_molecule_data(z, pos, cutoff=cutoff, device=device)
                _ = model(data)

        durations: List[float] = []
        with torch.no_grad():
            for _ in range(n_iters):
                t0 = time.perf_counter()
                data = build_single_molecule_data(z, pos, cutoff=cutoff, device=device)
                _ = model(data)
                t1 = time.perf_counter()
                durations.append(t1 - t0)

        stats = compute_latency_stats("painn_pytorch", n_atoms, durations)
        results.append(stats)

    return results


def write_csv(path: str, rows: Sequence[LatencyStats]) -> None:
    fieldnames = [
        "backend",
        "n_atoms",
        "mean_ms",
        "std_ms",
        "min_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "n_iters",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def maybe_plot(path: str, rows: Sequence[LatencyStats]) -> None:
    if not path:
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return

    backends = sorted({row.backend for row in rows})

    plt.figure(figsize=(6, 4))

    for backend in backends:
        xs = [row.n_atoms for row in rows if row.backend == backend]
        ys = [row.p50_ms for row in rows if row.backend == backend]
        plt.plot(xs, ys, marker="o", label=backend)

    plt.xlabel("Number of atoms")
    plt.ylabel("p50 latency (ms)")
    plt.title("Single-molecule inference latency vs system size")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark low-latency single-molecule inference for Hadronis and "
            "a minimal PyTorch+PyG PaiNN baseline."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["hadronis", "pytorch_painn", "both"],
        default="both",
        help=(
            "Which backend(s) to benchmark: Hadronis, PyTorch PaiNN, or both. "
            "The PyTorch model is a minimal PaiNN-style reference implemented "
            "in benchmarks/python/painn_pyg.py."
        ),
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="16,32,64,128,256,512,1024",
        help="Comma-separated list of atom counts to benchmark.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per size.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=1000,
        help=(
            "Number of timed iterations per size. For stable p99 estimates, "
            "values >= 1000 are recommended."
        ),
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
    parser.add_argument(
        "--pytorch-device",
        type=str,
        default="cpu",
        help=(
            "Device for the PyTorch PaiNN baseline, e.g. 'cpu' or 'cuda:0'. "
            "Hadronis currently runs on CPU."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional path to write a CSV file with latency statistics.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="",
        help="Optional path to save a latency vs. system size plot (PNG).",
    )

    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    # Print hardware / configuration header for reproducibility
    cpu_desc = platform.processor() or platform.uname().processor or "unknown"
    print(
        f"# CPU: {cpu_desc} | backend: {args.backend} | sizes: {sizes} | "
        f"n_iters: {args.n_iters} | n_warmup: {args.n_warmup} | cutoff: {args.cutoff} | "
        f"max_neighbors: {args.max_neighbors} | n_threads: {args.n_threads} | "
        f"pytorch_device: {args.pytorch_device}",
        flush=True,
    )

    all_results: List[LatencyStats] = []
    if args.backend in {"hadronis", "both"}:
        hadronis_results = benchmark_hadronis(
            weight_path=args.painn_weights,
            sizes=sizes,
            n_warmup=args.n_warmup,
            n_iters=args.n_iters,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            n_threads=args.n_threads,
        )
        all_results.extend(hadronis_results)

    if args.backend in {"pytorch_painn", "both"}:
        painn_results = benchmark_pytorch_painn(
            sizes=sizes,
            n_warmup=args.n_warmup,
            n_iters=args.n_iters,
            cutoff=args.cutoff,
            device_str=args.pytorch_device,
        )
        all_results.extend(painn_results)

    # Print a compact table to stdout
    header = (
        f"{'backend':>8}  {'n_atoms':>7}  {'mean_ms':>8}  {'p50_ms':>8}  "
        f"{'p95_ms':>8}  {'p99_ms':>8}  {'max_ms':>8}  {'n_iters':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(all_results, key=lambda r: (r.backend, r.n_atoms)):
        print(
            f"{row.backend:>8}  {row.n_atoms:7d}  "
            f"{row.mean_ms:8.3f}  {row.p50_ms:8.3f}  "
            f"{row.p95_ms:8.3f}  {row.p99_ms:8.3f}  "
            f"{row.max_ms:8.3f}  {row.n_iters:7d}"
        )

    if args.output_csv:
        write_csv(args.output_csv, all_results)

    if args.output_plot:
        maybe_plot(args.output_plot, all_results)


if __name__ == "__main__":  # pragma: no cover
    main()
