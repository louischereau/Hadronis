# Valence

[![CodSpeed](https://img.shields.io/badge/CodSpeed-Performance%20Tracking-blue?logo=github&style=flat-square)](https://codspeed.io/louischereau/Valence?utm_source=badge)

**High-performance Geometric GNN Engine for Chemistry and Physics**

## Overview

Valence is a hybrid Rust/Python library for molecular graph neural networks (GNNs), designed for scientific computing at scale. It combines the speed of Rust with the flexibility of Python, targeting real-world chemistry and physics applications.

## Chemistry Domain Knowledge

Valence builds molecular graphs from atomic coordinates and atomic numbers, representing each atom as a node and chemical bonds or spatial proximity as edges. The graph construction leverages domain knowledge:

 - **Nodes**: Atoms, defined by atomic number and 3D position.
 - **Edges**: Created using a distance-based cutoff, reflecting chemical bonding and physical interactions.
 - **RBF Expansion**:
	 - Edge features are expanded using Radial Basis Functions (RBFs), a standard technique in molecular machine learning.
	 - The typical RBF expansion formula is:

		 $RBF_i(d) = \exp(-\gamma (d - \mu_i)^2)$

		 where $d$ is the interatomic distance, $\mu_i$ is the center of the $i$-th basis function, and $\gamma$ controls the width.

	 - RBF expansion transforms raw interatomic distances into a smooth, differentiable feature space, improving the GNN’s ability to learn complex spatial relationships.
	 - This is critical for capturing both short-range (covalent) and long-range (non-covalent) interactions.
 - **Cutoff Choice**: The cutoff parameter (e.g., 1.2 Å for methane, 5.0 Å for large systems) is chosen to balance physical realism and computational efficiency. It captures both covalent bonds and relevant non-covalent interactions, ensuring the GNN sees all chemically meaningful neighbors without excessive noise.

## Why This Cutoff?

- **Chemistry**: Typical covalent bond lengths are 1–2 Å; non-covalent interactions (e.g., van der Waals) extend to 3–5 Å.
- **Use Case**: The default cutoff is tuned to include all atoms that can influence local electronic structure or molecular properties, maximizing predictive power for quantum chemistry, drug design, and materials science.


## Usage

**Python Example:**
```python
import valence
mol = valence.Molecule(
	atomic_numbers=[6, 1, 1, 1, 1],
	positions=[[0,0,0],[0.63,0.63,0.63],[-0.63,-0.63,0.63],[-0.63,0.63,-0.63],[0.63,-0.63,-0.63]]
)
engine = valence.ValenceEngine("gnn_model_weights.npy")
output = engine.run(mol, feats, cutoff=1.2, k=8)
```
- **Graph Construction**: The engine automatically builds a graph using atomic positions and applies the cutoff to define edges.
- **Inference**: The GNN processes the graph, aggregating neighbor information for each atom.


## License

MIT OR Apache-2.0
