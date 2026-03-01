import _lowlevel
import numpy as np
from numpy.typing import NDArray


class Engine:
    """High-level Hadronis engine wrapper.

    Constructed via :func:`compile`, then used through :meth:`predict`.
    """

    def __init__(
        self,
        weight_path: str,
        cutoff: float = 5.0,
        max_neighbors: int = 64,
        n_threads: int = 16,
    ) -> None:
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.n_threads = int(n_threads)
        self._engine = _lowlevel.HadronisEngine(
            weight_path, self.cutoff, self.max_neighbors, self.n_threads
        )

    def predict(
        self,
        atomic_numbers: NDArray[np.int32],
        positions: NDArray[np.float32],
        batch: NDArray[np.int32],
    ) -> np.ndarray:
        """Run inference for a batch of atoms.

        Parameters
        ----------
        atomic_numbers:
            1D array of shape ``[n_atoms]`` with atomic numbers (int32).
        positions:
            2D array of shape ``[n_atoms, 3]`` with 3D coordinates (float32).
        batch:
            1D array of shape ``[n_atoms]`` mapping each atom to a molecule index.
        """

        z = np.asarray(atomic_numbers, dtype=np.int32)
        r = np.asarray(positions, dtype=np.float32)
        b = np.asarray(batch, dtype=np.int32)

        if z.ndim != 1:
            raise ValueError("atomic_numbers must be 1D [n_atoms]")
        if r.shape != (z.shape[0], 3):
            raise ValueError("positions must have shape (n_atoms, 3)")
        if b.shape != z.shape:
            raise ValueError("batch must have shape (n_atoms,)")

        return self._engine.predict(z, r, b)


def compile(
    weight_path: str,
    cutoff: float = 5.0,
    max_neighbors: int = 64,
    n_threads: int = 16,
) -> Engine:
    """Compile a Hadronis model from a weight file.

    This is the main user-facing entrypoint:

    .. code-block:: python

        import hadronis
        engine = hadronis.compile("painn.bin", cutoff=5.0, max_neighbors=64)
    """

    return Engine(
        weight_path, cutoff=cutoff, max_neighbors=max_neighbors, n_threads=n_threads
    )
