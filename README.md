
# Valence

**High-performance Geometric GNN Engine for Chemistry and Physics**

## Overview

Valence is a hybrid Rust/Python library for molecular graph neural networks (GNNs), designed for scientific computing at scale. It combines the speed of Rust with the flexibility of Python, targeting real-world chemistry and physics applications.

## Chemistry Domain Knowledge

Valence builds molecular graphs from atomic coordinates and atomic numbers, representing each atom as a node and chemical bonds or spatial proximity as edges. The graph construction leverages domain knowledge:

 - **Nodes**: Atoms, defined by atomic number and 3D position.
 - **Edges**: Created using a distance-based cutoff, reflecting chemical bonding and physical interactions.
 - **RBF Expansion**:
	 - Edge features are expanded using Radial Basis Functions (RBFs), a standard technique in molecular machine learning.
	 - RBF expansion transforms raw interatomic distances into a smooth, differentiable feature space, improving the GNN’s ability to learn complex spatial relationships.
	 - This is critical for capturing both short-range (covalent) and long-range (non-covalent) interactions.
 - **Cutoff Choice**: The cutoff parameter (e.g., 1.2 Å for methane, 5.0 Å for large systems) is chosen to balance physical realism and computational efficiency. It captures both covalent bonds and relevant non-covalent interactions, ensuring the GNN sees all chemically meaningful neighbors without excessive noise.

## Why This Cutoff?

- **Chemistry**: Typical covalent bond lengths are 1–2 Å; non-covalent interactions (e.g., van der Waals) extend to 3–5 Å.
- **PhysicsX Use Case**: The default cutoff is tuned to include all atoms that can influence local electronic structure or molecular properties, maximizing predictive power for quantum chemistry, drug design, and materials science.

## Features

- **Rust core**: SIMD-optimized graph, model, and batch computation modules.
- **Python API**: Pydantic validation, easy integration, and rapid prototyping.
- **PyO3 bridge**: Zero-copy handoff for high-throughput inference.
- **Benchmarking**: Criterion-based Rust benchmarks for engine profiling.
- **Testing**: Pytest-based unit tests for scientific reliability.


## Systems Optimizations & Performance

Valence is engineered for high-throughput scientific workloads:

- **Graph Construction Time Complexity**:
	Building the molecular graph from atomic positions is $O(N^2)$ in the naive case (where $N$ is the number of atoms), as all pairwise distances are checked for edge creation. For small molecules, this is negligible; for large systems, optimizations (e.g., spatial partitioning) can be added.

- **Parallelism with Rayon**:
	Rust’s [Rayon](https://github.com/rayon-rs/rayon) library is used for data-parallelism in core modules (`batch.rs`, `graph.rs`).
	- **Why Rayon?**: Rayon provides ergonomic, zero-cost abstractions for parallel iteration, allowing us to scale across all CPU cores with minimal code changes.
	- **Where?**:
		- In `batch.rs`, batch inference over multiple graphs is parallelized.
		- In `graph.rs`, neighbor search and feature aggregation are parallelized for large graphs.

- **Batching**:
	Valence supports batch inference, allowing multiple molecular graphs to be processed simultaneously.
	- **How?**:
		- The `MolecularBatch` struct collects multiple graphs and runs inference in parallel, maximizing hardware utilization.
		- This is critical for real-world workloads (e.g., screening thousands of molecules) and is fully integrated with the Python API.

- **SIMD & Native Optimizations**:
	The engine is built with `RUSTFLAGS="-C target-cpu=native"` for maximum SIMD (AVX/SSE) utilization, further accelerating matrix and vector operations.

## For PhysicsX: Scientific Reliability & Extensibility

- **Physical Realism**:
	- All graph construction and inference steps are grounded in chemical and physical principles (e.g., cutoff selection, neighbor search).
	- Supports both covalent and non-covalent interactions, enabling accurate modeling of molecular and materials systems.

- **Extensibility**:
	- Modular Rust and Python codebase allows rapid prototyping and extension to new GNN architectures, featurizations, or physical models.
	- Easy to integrate with existing simulation pipelines or data sources.

- **Validation & Testing**:
	- Comprehensive unit tests (Python/Pytest and Rust) ensure correctness and reproducibility.
	- Example test cases (e.g., methane) demonstrate validation of both input data and output shapes.

- **Performance Benchmarks**:
	- Criterion-based Rust benchmarks and profiling scripts provide transparent performance metrics.
	- SIMD and multi-core scaling demonstrated for large molecular systems.

- **Batch Processing & Throughput**:
	- Designed for high-throughput screening and large-scale inference (e.g., drug discovery, materials informatics).
	- Batch API and parallelization maximize resource utilization.

- **Integration**:
	- Python API is compatible with NumPy, Pydantic, and standard scientific Python tools.
	- Rust core can be called directly for custom workflows or embedded in larger systems.

- **Reproducibility**:
	- All dependencies are pinned and installation is automated via `Makefile` and `pyproject.toml`.
	- Example weights and test data included for out-of-the-box validation.

## Directory Structure

- `src/`: Rust source code (`lib.rs`, `graph.rs`, `model.rs`, `batch.rs`)
- `python/valence/`: Python API (`core.py`, `__init__.py`)
- `benches/`: Rust benchmarks (`molecular_bench.rs`)
- `examples/`: Rust profiling example (`profile_engine.rs`)
- `scripts/`: Python utilities (`memory_tracker.py`, `stress_tests.py`)
- `tests/`: Python tests (`test_core.py`)
- `test_weights.npy`: Example weights for testing

## Installation

**Python:**
```bash
pip install maturin
maturin develop
```

**Rust:**
```bash
cargo build --release
```

## Usage

**Python Example:**
```python
import valence
mol = valence.Molecule(
	atomic_numbers=[6, 1, 1, 1, 1],
	positions=[[0,0,0],[0.63,0.63,0.63],[-0.63,-0.63,0.63],[-0.63,0.63,-0.63],[0.63,-0.63,-0.63]]
)
engine = valence.ValenceEngine("test_weights.npy")
output = engine.run(mol, feats, cutoff=1.2, k=8)
```
- **Graph Construction**: The engine automatically builds a graph using atomic positions and applies the cutoff to define edges.
- **Inference**: The GNN processes the graph, aggregating neighbor information for each atom.

**Rust Example:**
See `examples/profile_engine.rs` and `benches/molecular_bench.rs`.

## Testing

```bash
pytest tests/
cargo bench
```

## License

MIT OR Apache-2.0
