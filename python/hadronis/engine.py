# ruff: noqa: I001

import numpy as np
from numpy.typing import NDArray
from . import _lowlevel


class HadronisEngine:
    def __init__(self, weight_path: str):
        """
        Load GNN model weights once at engine initialization (Rust HadronisEngine).
        """
        self.engine = _lowlevel.HadronisEngine.from_file(weight_path)
        if self.engine is None:
            raise ValueError("Failed to load HadronisEngine weights.")

    def predict_batch(
        self,
        atomic_numbers: NDArray[np.int32],
        positions: NDArray[np.float32],
        features: NDArray[np.float32],
        mol_ptrs: NDArray[np.int32],
        cutoff: float = 5.0,
        k: int = 16,
        num_rbf: int = 32,
    ) -> np.ndarray:
        """
        Run fully vectorized batch inference using new Rust API.

        Parameters
        ----------
        atomic_numbers : np.ndarray[int32]
            Flattened atomic numbers for all molecules.
        positions : np.ndarray[float32]
            Flattened 3D positions for all atoms, shape [total_atoms, 3].
        features : np.ndarray[float32]
            Flattened per-atom feature arrays, shape [total_atoms, feature_dim]
        mol_ptrs : np.ndarray[int32]
            Start/end indices per molecule: cumulative sum of atom counts.
            Length = num_molecules + 1
        cutoff : float
            Neighbor cutoff distance.
        k : int
            Max number of neighbors per atom.
        num_rbf : int
            Number of RBF features for edge expansion.

        Returns
        -------
        np.ndarray
            GNN outputs for each molecule in the batch.
        """
        # --- Input validation ---
        if (
            not isinstance(atomic_numbers, np.ndarray)
            or atomic_numbers.dtype != np.int32
        ):
            raise ValueError("atomic_numbers must be a np.ndarray of dtype int32")
        if not isinstance(positions, np.ndarray) or positions.dtype != np.float32:
            raise ValueError("positions must be a np.ndarray of dtype float32")
        if not isinstance(mol_ptrs, np.ndarray) or mol_ptrs.dtype != np.int32:
            raise ValueError("mol_ptrs must be a np.ndarray of dtype int32")
        if not isinstance(features, np.ndarray) or features.dtype != np.float32:
            raise ValueError("features must be a np.ndarray of dtype float32")
        if self.engine is None:
            raise ValueError("Engine weights are not loaded.")

        # --- Build neighbor list (Rust) ---
        edge_src, edge_dst, edge_relpos = (
            _lowlevel.HadronisEngine.build_batched_neighbors(
                positions, mol_ptrs.tolist(), cutoff, k
            )
        )
        # Ensure outputs are numpy arrays for downstream Rust extension compatibility
        edge_src = (
            np.array(edge_src, dtype=np.int32)
            if not isinstance(edge_src, np.ndarray)
            else edge_src
        )
        edge_dst = (
            np.array(edge_dst, dtype=np.int32)
            if not isinstance(edge_dst, np.ndarray)
            else edge_dst
        )
        edge_relpos = (
            np.array(edge_relpos, dtype=np.float32)
            if not isinstance(edge_relpos, np.ndarray)
            else edge_relpos
        )

        # --- Run batched inference ---
        return self.engine.run_batched(
            atomic_numbers.tolist(),
            positions,
            features,
            edge_relpos,
            edge_src,
            edge_dst,
            mol_ptrs.tolist(),
            cutoff,
            num_rbf,
        )

    @staticmethod
    def build_batched_neighbors(
        positions: NDArray[np.float32],
        mol_ptrs: NDArray[np.int32],
        cutoff: float = 5.0,
        k: int = 16,
    ):
        """
        Build batched neighbor lists for molecules using the Rust backend.

        Parameters
        ----------
        positions : np.ndarray[float32]
            Atom positions, shape [total_atoms, 3].
        mol_ptrs : np.ndarray[int32]
            Start/end indices per molecule: cumulative sum of atom counts.
        cutoff : float
            Neighbor cutoff distance.
        k : int
            Max number of neighbors per atom.

        Returns
        -------
        edge_src : list[int]
            Source atom indices for edges.
        edge_dst : list[int]
            Destination atom indices for edges.
        edge_relpos : np.ndarray[float32]
            Relative position vectors for each edge, shape [n_edges, 3].
        """
        if not isinstance(positions, np.ndarray) or positions.dtype != np.float32:
            raise ValueError("positions must be a np.ndarray of dtype float32")
        if not isinstance(mol_ptrs, np.ndarray) or mol_ptrs.dtype != np.int32:
            raise ValueError("mol_ptrs must be a np.ndarray of dtype int32")
        edge_src, edge_dst, edge_relpos = (
            _lowlevel.HadronisEngine.build_batched_neighbors(
                positions, mol_ptrs.tolist(), cutoff, k
            )
        )
        edge_src = (
            np.array(edge_src, dtype=np.int32)
            if not isinstance(edge_src, np.ndarray)
            else edge_src
        )
        edge_dst = (
            np.array(edge_dst, dtype=np.int32)
            if not isinstance(edge_dst, np.ndarray)
            else edge_dst
        )
        edge_relpos = (
            np.array(edge_relpos, dtype=np.float32)
            if not isinstance(edge_relpos, np.ndarray)
            else edge_relpos
        )
        return edge_src, edge_dst, edge_relpos
