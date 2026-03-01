# Hadronis

[![CodSpeed](https://img.shields.io/badge/CodSpeed-Performance%20Tracking-blue?logo=github&style=flat-square)](https://codspeed.io/louischereau/Hadronis?utm_source=badge)

**High-performance Geometric GNN Engine for Chemistry and Physics**

## Overview

Hadronis is a high-throughput batch inference engine library for molecular graph neural networks (GNNs), designed for CPU-bound scientific computing at scale. It currently targets a PaiNN-style equivariant architecture rather than arbitrary GNNs, letting the implementation focus on optimizing a single, well-motivated model family instead of reimplementing a generic GNN framework. It combines the speed of C++ with the flexibility of Python, targeting real-world chemistry and physics applications.

## Why Hadronis?

Modern molecular ML has shifted from *structure → properties* (classical QSAR, hand-crafted descriptors) to *properties → structure* with large generative graph transformers. These models can propose vast numbers of candidate molecules, but validating each one with high-level quantum methods like DFT is still orders of magnitude too slow for practical screening. Hadronis focuses on the complementary direction: a high-throughput, batched structure → properties engine that can rapidly filter out physically implausible or uninteresting candidates before expensive calculations. In other words, it is designed to sit between generative models and ab initio verification, providing a fast, geometry-aware signal that helps triage large libraries.

At the same time, Hadronis is explicitly **not** a drop-in replacement for first-principles methods or experimental data. Using AI to evaluate AI-generated structures carries risks: systematic biases in the training data, failure modes on out-of-distribution chemistry, and feedback loops where the generator over-optimizes for the surrogate model instead of real physics. The goal is therefore to provide a transparent, well-engineered inference engine that is easy to benchmark, stress-test, and validate against trusted reference methods, not to claim ground truth on its own.

### Core Model: PaiNN

Hadronis is optimized around PaiNN-like message passing for molecular systems. PaiNN provides a strong balance between physical inductive bias and engineering practicality:

- **Equivariance built-in**: PaiNN operates on scalar and vector features in a way that is invariant to global rotations and translations of the molecular geometry. This is a natural fit for 3D chemistry, where predictions should not depend on how a molecule is oriented in space.
- **Compact and efficient**: Compared to very large graph transformers or attention-based 3D models, PaiNN-style networks are relatively lightweight in parameter count and memory footprint. This makes them well-suited to high-throughput, CPU-oriented screening where throughput and latency matter.
- **Targeted, not “framework-y”**: By committing to a PaiNN-style architecture, Hadronis can specialize data layouts, neighbor list construction, and kernel implementations for this one family of models instead of trying to be a general-purpose GNN engine (which frameworks like PyTorch Geometric already cover). The goal is a small, focused runtime for fast, robust inference—not a full training ecosystem.

## Chemistry Domain Knowledge

Hadronis builds molecular graphs from atomic coordinates and atomic numbers, representing each atom as a node and chemical bonds or spatial proximity as edges. The graph construction leverages domain knowledge:

 - **Nodes**: Atoms, defined by atomic number and 3D position.
 - **Edges**: Created using a distance-based cutoff, reflecting chemical bonding and physical interactions.
 - **RBF Expansion**:
	 - Edge features are expanded using Radial Basis Functions (RBFs), a standard technique in molecular machine learning.
	 - The typical RBF expansion formula is:

		 $RBF_i(d) = \exp(-\gamma (d - \mu_i)^2)$

		 where $d$ is the interatomic distance, $\mu_i$ is the center of the $i$-th basis function, and $\gamma$ controls the width.

	 - RBF expansion transforms raw interatomic distances into a smooth, differentiable feature space, improving the GNN’s ability to learn complex spatial relationships.
	 - This is critical for capturing both short-range (covalent) and long-range (non-covalent) interactions.
 - **Symmetries and invariances**: The underlying PaiNN-style architecture is designed to respect the fundamental symmetries of molecular systems: predictions are invariant to global translations and rotations of the molecule, and (ideally) to permutations of atoms within a molecule that leave the physical system unchanged. In practice, this means the model learns on relative geometry and composition rather than arbitrary coordinate frames or atom orderings, which is essential for chemically meaningful generalisation.
 - **Cutoff Choice**: The cutoff parameter (e.g., 1.2 Å for methane, 5.0 Å for large systems) is chosen to balance physical realism and computational efficiency. It captures both covalent bonds and relevant non-covalent interactions, ensuring the GNN sees all chemically meaningful neighbors without excessive noise.

## Why This Cutoff?

- **Chemistry**: Typical covalent bond lengths are 1–2 Å; non-covalent interactions (e.g., van der Waals) extend to 3–5 Å.
- **Use Case**: The default cutoff is tuned to include all atoms that can influence local electronic structure or molecular properties, maximizing predictive power for quantum chemistry, drug design, and materials science.


## Limitations

The current graph construction and cutoff design are primarily targeted at neutral organic and small-molecule chemistry in gas-phase or simple solvent-like environments. Systems with strong ionic character, highly delocalised electronic structure, extended periodic materials, or exotic bonding patterns may require adapted featurisation, longer-range interaction models, or specialised training data before Hadronis can be used reliably.


## Usage

**Python Example:**
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

# Map each atom to its molecule
batch = np.zeros(len(Z), dtype=np.int32)

predictions = engine.predict(
    atomic_numbers=Z,
    positions=R,
    batch=batch,
)
```
- **Graph Construction**: The engine automatically builds a graph using atomic positions and applies the cutoff to define edges.
- **Inference**: The GNN processes the graph, aggregating neighbor information for each atom.

## Future Work

- **GPU backend for high-throughput inference**: Add an optional CUDA backend for neighbor list construction and message passing, while keeping the current CPU path as a portable reference. This would let Hadronis saturate modern accelerators for massive screening campaigns, without forcing GPU as a hard dependency for users who only need lightweight CPU inference.

## License

MIT OR Apache-2.0
