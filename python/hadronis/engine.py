# ruff: noqa: I001
import numpy as np
from numpy.typing import NDArray
from . import _lowlevel


class HadronisEngine:
    def __init__(self, weight_path: str):
        """
        Load GNN model weights once at engine initialization.
        """
        self.model = _lowlevel.GNNModel.from_file(weight_path)
        if self.model is None:
            raise ValueError("Failed to load GNN model weights.")

    def predict_batch(
        self,
        atomic_numbers_batch: NDArray[np.int32],
        positions_batch: NDArray[np.float32],
        mol_ptrs: NDArray[np.int32],
        features_batch: NDArray[np.float32],
        cutoff: float = 5.0,
        k: int = 16,
    ) -> list[NDArray[np.float32]]:
        """
        Fully vectorized batch inference with zero Python overhead.

        Parameters
        ----------
        atomic_numbers_batch : np.ndarray[int32]
            Flattened atomic numbers for all molecules.
        positions_batch : np.ndarray[float32]
            Flattened 3D positions for all atoms, shape [total_atoms, 3].
        mol_ptrs : np.ndarray[int32]
            Start/end indices per molecule: cumulative sum of atom counts.
            Length = num_molecules + 1
        features_batch : np.ndarray[float32]
            Flattened per-atom feature arrays, shape [total_atoms, feature_dim]
        cutoff : float
            Neighbor cutoff distance.
        k : int
            Max number of neighbors per atom.

        Returns
        -------
        results : list[NDArray[np.float32]]
            GNN outputs for each molecule in the batch.
        """

        # --- Input validation (minimal overhead) ---
        if (
            not isinstance(atomic_numbers_batch, np.ndarray)
            or atomic_numbers_batch.dtype != np.int32
        ):
            raise ValueError("atomic_numbers_batch must be a np.ndarray of dtype int32")
        if (
            not isinstance(positions_batch, np.ndarray)
            or positions_batch.dtype != np.float32
        ):
            raise ValueError("positions_batch must be a np.ndarray of dtype float32")
        if not isinstance(mol_ptrs, np.ndarray) or mol_ptrs.dtype != np.int32:
            raise ValueError("mol_ptrs must be a np.ndarray of dtype int32")
        if (
            not isinstance(features_batch, np.ndarray)
            or features_batch.dtype != np.float32
        ):
            raise ValueError("features_batch must be a np.ndarray of dtype float32")
        if self.model is None:
            raise ValueError("Model weights are not loaded.")

        # # --- Build Rust batch (all molecules in one call) ---
        # rust_batch = _lowlevel.MolecularBatch.from_arrays(
        #     atomic_numbers_batch, positions_batch, mol_ptrs
        # )

        # # --- Run fully parallel GNN inference ---
        # results = rust_batch.run_batch_inference(
        #     self.model, features_batch, cutoff, k
        # )

        # return results
        return _lowlevel.run_batch_inference(
            self.model,
            atomic_numbers_batch,
            positions_batch,
            mol_ptrs,
            features_batch,
            cutoff,
            k,
        )
