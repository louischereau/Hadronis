# ruff: noqa: I001
from pydantic import BaseModel, Field, field_validator
import numpy as np
from . import _lowlevel  # ruff: noqa


class Molecule(BaseModel):
    atomic_numbers: list[int] = Field(..., min_length=1)
    positions: list[list[float]]

    @field_validator("atomic_numbers")
    @classmethod
    def check_positive_elements(cls, v):
        if any(z <= 0 for z in v):
            raise ValueError("Atomic numbers must be positive integers")
        return v

    @field_validator("positions")
    @classmethod
    def check_dimensions(cls, v, info):
        num_atoms = len(info.data.get("atomic_numbers", []))
        if len(v) != num_atoms:
            raise ValueError(f"Expected {num_atoms} positions, got {len(v)}")
        if any(len(coord) != 3 for coord in v):
            raise ValueError("Each coordinate must be exactly 3D (x, y, z)")
        return v

    def build_graph(self) -> _lowlevel.MolecularGraph:
        """
        Initializes the Rust-side graph object.
        Note: Passing a list of lists will be handled by our optimized
        Rust constructor using PyReadonlyArray2.
        """
        # Convert to numpy array for zero-copy handoff in Rust
        pos_array = np.array(self.positions, dtype=np.float32)
        return _lowlevel.MolecularGraph(self.atomic_numbers, pos_array)
