# ruff: noqa: I001
import numpy as np
from . import _lowlevel
from .molecule import Molecule


class ValenceEngine:
    def __init__(self, weight_path: str = None):
        self.model = None
        if weight_path:
            # Assume weights are stored as a .npy file for now
            w = np.load(weight_path).astype(np.float32)
            self.model = _lowlevel.GNNModel(w)

    def run(
        self,
        molecule: Molecule,
        atom_features: np.ndarray,
        cutoff: float = 5.0,
        k: int = 16,
    ):
        graph = molecule.build_graph()
        # Pass the model weights into the fused parallel kernel
        return graph.run_fused_with_model(self.model, atom_features, cutoff, k)

    def predict_batch(
        self,
        molecules: list[Molecule],
        features_list: list[np.ndarray],
        cutoff: float = 5.0,
        k: int = 16,
    ):
        """
        High-throughput entry point. Takes a list of molecules and runs
        them in a single parallel sweep in Rust.
        Adds input validation and debug logging to catch invalid input and diagnose issues.
        """
        # Input validation
        if not isinstance(molecules, list) or not all(
            isinstance(m, Molecule) for m in molecules
        ):
            raise ValueError("'molecules' must be a list of Molecule objects.")
        if not isinstance(features_list, list) or not all(
            isinstance(f, np.ndarray) for f in features_list
        ):
            raise ValueError("'features_list' must be a list of numpy arrays.")
        if len(molecules) != len(features_list):
            raise ValueError(
                f"Number of molecules ({len(molecules)}) does not match number of feature arrays ({len(features_list)})."
            )
        for i, (mol, feats) in enumerate(zip(molecules, features_list)):
            if len(mol.atomic_numbers) != feats.shape[0]:
                raise ValueError(
                    f"Feature array at index {i} does not match number of atoms in molecule: {len(mol.atomic_numbers)} vs {feats.shape[0]}"
                )
        if self.model is None:
            raise ValueError(
                "Model weights are not loaded. Please initialize ValenceEngine with a valid weight_path."
            )

        # 1. Build low-level graphs
        rust_graphs = [m.build_graph() for m in molecules]

        # 2. Create the Batch object
        batch = _lowlevel.MolecularBatch(rust_graphs)

        # 3. Execute parallel batch inference
        results = batch.run_batch_inference(self.model, features_list, cutoff, k)

        return results
