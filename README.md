# Hadronis

[![PyPI version](https://img.shields.io/pypi/v/hadronis?cacheSeconds=3600)](https://pypi.org/project/hadronis/) [![Python versions](https://img.shields.io/pypi/pyversions/hadronis?cacheSeconds=3600)](https://pypi.org/project/hadronis/) [![PyPI downloads](https://img.shields.io/pypi/dm/hadronis)](https://pypi.org/project/hadronis/) [![CodSpeed](https://img.shields.io/badge/CodSpeed-Performance%20Tracking-blue?logo=github&style=flat-square)](https://codspeed.io/louischereau/Hadronis?utm_source=badge)


**A minimal, CPU-Optimized PaiNN Inference Pipeline for Molecular Graph Neural Networks**

## Overview

Hadronis is a low-latency, single-molecule all-in-one inference pipeline for molecular graph neural networks (GNNs), designed for CPU-bound scientific computing where per-configuration evaluation time matters. It currently targets a PaiNN-style equivariant architecture rather than arbitrary GNNs, letting the implementation focus on a single, well-motivated model family instead of reimplementing a generic GNN framework. It combines the speed of C++ with the flexibility of Python, targeting real-world chemistry and physics applications.

## Why Hadronis?

Many molecular ML applications now require fast, per-configuration evaluations rather than just large-batch screening. Examples include molecular dynamics (MD) and Monte Carlo simulations with learned potentials, real-time exploration of potential energy surfaces, and tight control loops where a single molecule (or a small system) must be evaluated at every step. In these regimes, the primary constraint is latency per inference rather than total throughput over large libraries.

Hadronis focuses on this setting: a compact, CPU-optimized structure → properties engine that can sit inside inner simulation loops or interactive workflows, providing geometry-aware predictions (for example, energies, forces, or other observables) for a single molecular configuration at a time. It is intended to be embedded in MD or other simulation codes as a surrogate for more expensive electronic-structure calculations when appropriate, or as a fast pre-screening layer to decide when higher-level methods should be called.

At the same time, Hadronis is explicitly **not** a drop-in replacement for first-principles methods or experimental data. Using AI to evaluate molecular configurations carries risks: systematic biases in the training data, failure modes on out-of-distribution chemistry, and feedback loops where a simulator or generator over-optimizes for the surrogate model instead of real physics. The goal is therefore to provide a transparent, well-engineered inference pipeline that is easy to benchmark, stress-test, and validate against trusted reference methods, not to claim ground truth on its own.

### Core Model: PaiNN

Hadronis is optimized around PaiNN-like message passing for molecular systems. PaiNN provides a strong balance between physical inductive bias and engineering practicality:

- **Equivariance built-in**: PaiNN operates on scalar and vector features in a way that is invariant to global rotations and translations of the molecular geometry. This is a natural fit for 3D chemistry, where predictions should not depend on how a molecule is oriented in space.
- **Compact and efficient**: Compared to very large graph transformers or attention-based 3D models, PaiNN-style networks are relatively lightweight in parameter count and memory footprint. This makes them well-suited to high-throughput, CPU-oriented screening where throughput and latency matter.
- **Targeted, not “framework-y”**: By committing to a PaiNN-style architecture, Hadronis can specialize data layouts, neighbor list construction, and kernel implementations for this one family of models instead of trying to be a general-purpose GNN engine (which frameworks like PyTorch Geometric already cover). The goal is a small, focused runtime for fast, robust inference—not a full training ecosystem.

Compared to models such as MACE, which use higher-order equivariant features and richer angular bases, this PaiNN-style focus trades some architectural complexity for lower per-step cost—especially on CPUs. That makes it easier to hand-optimise kernels, control memory use, and port weights between reference PyTorch implementations and the C++ runtime, while still retaining the key geometric equivariances needed for molecular modeling.

## Architecture

**Architecture (single configuration → prediction)**

## Architecture

| Stage | Transformation | Description |
|-------|----------------|-------------|
| 1. Input            | (Z, R) → neighbor list      | Atomic numbers Z and positions R for a single configuration. |
| 2. Neighbor graph   | neighbor list → distances dᵢⱼ | Build a fixed-size neighbor list (cutoff, max_neighbors) and compute interatomic distances. |
| 3. RBF features     | dᵢⱼ → RBFᵢ(dᵢⱼ)            | Expand distances into radial basis function features. |
| 4. PaiNN blocks     | RBF features → learned features | Apply PaiNN-style equivariant message passing and feature updates. |
| 5. Per-atom outputs | features → per-atom scalars | Map final features to per-atom predictions (e.g. energy contributions). |
| 6. Aggregation      | per-atom → global           | Optionally aggregate (e.g. sum) to obtain global quantities. |

- **Inputs (Z, R)**: A single frame of atomic numbers and 3D coordinates for one molecule or configuration.
- **Neighbor list**: For each atom, Hadronis builds a fixed-size list of nearby atoms based on a distance cutoff and `max_neighbors`, which drives both accuracy and performance.
- **Distances and RBFs**: Interatomic distances along edges are expanded into a bank of radial basis functions RBFᵢ(d), giving a smooth, expressive representation of local geometry.
- **PaiNN interaction blocks**: Stacked PaiNN-style layers update scalar and vector features using equivariant message passing over the neighbor graph, encoding chemistry-aware local environments.
- **Readout and aggregation**: Final features are mapped to per-atom scalars (e.g. energy contributions) and optionally aggregated (e.g. summed) to produce global quantities.

## Molecular Graph Representation

Hadronis builds molecular graphs from atomic coordinates and atomic numbers, representing each atom as a node and chemical bonds or spatial proximity as edges. The graph construction leverages domain knowledge:

 - **Nodes**: Atoms, defined by atomic number and 3D position.
 - **Edges**: Created using a distance-based cutoff, reflecting chemical bonding and physical interactions.
 - **RBF Expansion**:
	 - Edge features are expanded using Radial Basis Functions (RBFs), a standard technique in molecular machine learning.
	 - The typical RBF expansion formula is:

		 RBFᵢ(d) = exp(-γ (d − μᵢ)²)

		 where d is the interatomic distance, μᵢ is the center of the i-th basis function, and γ controls the width.

	 - RBF expansion transforms raw interatomic distances into a smooth, differentiable feature space, improving the GNN’s ability to learn complex spatial relationships.
	 - This is critical for capturing both short-range (covalent) and long-range (non-covalent) interactions.
 - **Symmetries and invariances**: The representation is compatible with PaiNN's equivariant architecture, which respects global translations and rotations of the molecule.
 - **Cutoff Choice**: The cutoff parameter (e.g., 1.2 Å for methane, 5.0 Å for large systems) is chosen to balance physical realism and computational efficiency. It captures both covalent bonds and relevant non-covalent interactions, ensuring the GNN sees all chemically meaningful neighbors without excessive noise.


## Usage

**Python Example (single molecule, low latency):**
```python
import hadronis
import numpy as np

engine = hadronis.compile(
	"painn.bin",
	cutoff=5.0,
    max_neighbors=64,
    n_threads=16
)

Z = np.array([6, 1, 1, 1, 1], dtype=np.int32)

R = np.array([
    [0.0, 0.0, 0.0],
    [0.6, 0.6, 0.6],
    [-0.6, -0.6, 0.6],
    [-0.6, 0.6, -0.6],
    [0.6, -0.6, -0.6],
], dtype=np.float32)

predictions = engine.predict(
    atomic_numbers=Z,
    positions=R,
)
```
**MD-style inner loop (conceptual):**
```python
engine = hadronis.compile("painn.bin", cutoff=5.0)

Z = ...  # (n_atoms,) atomic numbers
R = R0   # (n_atoms, 3) initial positions

for step in range(n_steps):
    # advance positions using your integrator
    R = integrator_step(R)

    # low-latency single-configuration inference
    per_atom_pred = engine.predict(Z, R)
    total_pred = per_atom_pred.sum()

    # use `total_pred` (e.g. as an energy-like scalar)
    control_simulation(total_pred)
```

## Limitations

The current graph construction and cutoff design are primarily targeted at neutral organic and small-molecule chemistry in gas-phase or simple solvent-like environments. Systems with strong ionic character, highly delocalised electronic structure, extended periodic materials, or exotic bonding patterns may require adapted featurisation, longer-range interaction models, or specialised training data before Hadronis can be used reliably.

## License

MIT OR Apache-2.0
