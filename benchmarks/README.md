# Hadronis Benchmarks

This directory contains **local performance benchmarks** for low-latency, single-molecule inference.

The main entry points today are:

- `python/benchmark_single_molecule_latency.py`
  - Steady-state single-molecule latency vs system size.
  - Supports Hadronis and a minimal **PyTorch+PyG PaiNN** baseline implemented in `python/painn_pyg.py`.
  - Also reports a Hadronis **cold-start** measurement (compile + first predict) for the smallest system.
- `python/benchmark_thread_scaling.py`
  - Steady-state Hadronis latency vs number of threads for a few representative system sizes.
- `python/benchmark_md_trace.py`
  - MD-style inner-loop benchmark: runs a long trace for a single system and reports per-step latency statistics.

For the first script, you can choose to run Hadronis only, the PyTorch PaiNN baseline only, or both, using a `--backend` flag. This keeps the sampling, percentile computation, and plotting logic in one place while providing an architecture-aligned PyTorch reference.

Both Hadronis and the PyTorch PaiNN baseline include **neighbor list construction** in their timed calls: Hadronis performs neighbor search and radial basis expansion internally on each `predict`, while the PyTorch baseline rebuilds the radius-graph inside each iteration via `torch_cluster.radius_graph`. This matches the intended MD-style use case where positions change every step and the graph cannot be reused.

## Requirements

- Hadronis installed and importable (built from this repo or from a wheel)
- For the PyTorch PaiNN baseline:
  - `torch`
  - `torch_geometric`
  - `torch_scatter`
  - `torch_cluster`
- Optional (for plotting):
  - `matplotlib`

## Usage

From the project root, with your virtual environment active:

### Single-molecule latency (Hadronis vs PyTorch PaiNN)

Run both backends and plot them together:

```bash
python benchmarks/python/benchmark_single_molecule_latency.py \
  --backend both \
  --painn-weights painn.bin \
  --pytorch-device cpu \
  --output-csv latency_compare.csv \
  --output-plot latency_compare.png
```

Hadronis only:

```bash
python benchmarks/python/benchmark_single_molecule_latency.py \
  --backend hadronis \
  --painn-weights painn.bin \
  --output-csv hadronis_latency.csv \
  --output-plot hadronis_latency.png
```

PyTorch PaiNN only:

```bash
python benchmarks/python/benchmark_single_molecule_latency.py \
  --backend pytorch_painn \
  --pytorch-device cpu \
  --output-csv painn_latency.csv \
  --output-plot painn_latency.png
```

### Key options

- `--backend`: `hadronis`, `pytorch_painn`, or `both`.
- `--sizes`: comma-separated list of atom counts to benchmark (default `16,32,64,128,256,512,1024`).
- `--n-warmup`: warmup iterations per size (default `10`).
- `--n-iters`: timed iterations per size (default `1000` recommended for stable p99 estimates).
- `--pytorch-device`: device for the PyTorch baseline (e.g. `cpu`, `cuda:0`).
- `--cutoff`, `--max-neighbors`, `--n-threads`: control Hadronis neighbor construction and parallelism.
- `--output-csv`: optional CSV with per-size latency stats.
- `--output-plot`: optional PNG of p50 latency vs system size.

### Thread scaling (Hadronis only)

```bash
python benchmarks/python/benchmark_thread_scaling.py \
  --sizes 64,256,1024 \
  --threads 1,2,4,8,16 \
  --painn-weights painn.bin
```

This prints p50/p95/p99 latency for each `(n_atoms, n_threads)` configuration.

### MD-style inner loop (Hadronis only)

```bash
python benchmarks/python/benchmark_md_trace.py \
  --n-atoms 256 \
  --n-steps 10000 \
  --painn-weights painn.bin
```

This approximates embedding Hadronis in an MD integrator, reporting per-step latency statistics over a long trace.

You can add additional benchmark scripts under this folder (for example, GPU-specific runs or alternative baselines) while keeping this README as an overview of the available tools.