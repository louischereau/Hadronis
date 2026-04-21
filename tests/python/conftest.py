"""Session fixtures shared across all Python tests."""

import json
import struct
from pathlib import Path

import pytest

# The tests reference "dummy-weights.bin" as a relative path, which resolves
# against the CWD at pytest invocation time (normally the repo root).
# We write it next to the repo root CMakeLists.txt so the path is stable
# regardless of where pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DUMMY_WEIGHTS = _REPO_ROOT / "dummy-weights.bin"

# PaINN hyper-parameters (must match kHiddenDim / kNumRbf in hadronis.cpp)
_H = 128
_N_RBF = 20
_N_INT = 3
_MAX_Z = 100


def _build_weight_shapes() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {
        "embedding.weight": [_MAX_Z, _H],
    }
    for i in range(_N_INT):
        b = f"interactions.{i}"
        shapes[f"{b}.message.mlp_linear1.weight"] = [_H, _H]
        shapes[f"{b}.message.mlp_linear1.bias"] = [_H]
        shapes[f"{b}.message.mlp_linear2.weight"] = [3 * _H, _H]
        shapes[f"{b}.message.mlp_linear2.bias"] = [3 * _H]
        shapes[f"{b}.message.mlp_linear3.weight"] = [3 * _H, _N_RBF]
        shapes[f"{b}.message.mlp_linear3.bias"] = [3 * _H]
        shapes[f"{b}.update.U.weight"] = [_H, _H]
        shapes[f"{b}.update.U.bias"] = [_H]
        shapes[f"{b}.update.V.weight"] = [_H, _H]
        shapes[f"{b}.update.V.bias"] = [_H]
        shapes[f"{b}.update.linear1.weight"] = [_H, 2 * _H]
        shapes[f"{b}.update.linear1.bias"] = [_H]
        shapes[f"{b}.update.linear2.weight"] = [3 * _H, _H]
        shapes[f"{b}.update.linear2.bias"] = [3 * _H]
    shapes["readout.linear1.weight"] = [_H, _H]
    shapes["readout.linear1.bias"] = [_H]
    shapes["readout.linear2.weight"] = [1, _H]
    shapes["readout.linear2.bias"] = [1]
    return shapes


def _write_dummy_safetensors(path: Path) -> None:
    """Write a minimal valid safetensors file with zero-valued F32 tensors."""
    shapes = _build_weight_shapes()

    header: dict = {}
    offset = 0
    data_parts: list[bytes] = []

    for name, shape in shapes.items():
        n_elems = 1
        for d in shape:
            n_elems *= d
        n_bytes = n_elems * 4  # float32 = 4 bytes per element
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + n_bytes],
        }
        data_parts.append(b"\x00" * n_bytes)
        offset += n_bytes

    # Pad header to a multiple of 8 bytes (required by safetensors spec)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padded_len = (len(header_bytes) + 7) & ~7
    header_bytes = header_bytes.ljust(padded_len)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", padded_len))
        f.write(header_bytes)
        for part in data_parts:
            f.write(part)


@pytest.fixture(scope="session", autouse=True)
def dummy_weights_file():
    """Create a minimal safetensors weight file once per test session."""
    _write_dummy_safetensors(_DUMMY_WEIGHTS)
    yield
    _DUMMY_WEIGHTS.unlink(missing_ok=True)


# Known-weights file: all weights/biases are zero except readout.linear2.bias
# which is set to _KNOWN_READOUT_BIAS.  This makes the expected total energy
# analytically computable: total = n_atoms * _KNOWN_READOUT_BIAS (the zero
# embedding produces zero scalar features; zero interaction weights leave them
# at zero through all message-passing layers; linear1 with zero weight and
# zero bias maps zero → zero; silu(0) = 0; linear2 with zero weight but
# non-zero bias maps zero → bias).
_KNOWN_WEIGHTS = _REPO_ROOT / "known-weights.bin"
KNOWN_READOUT_BIAS: float = 2.5


def _write_known_safetensors(path: Path, readout_linear2_bias: float) -> None:
    """Write a safetensors file identical to the dummy one except that
    readout.linear2.bias is set to *readout_linear2_bias* (a scalar broadcast
    to its single element)."""
    shapes = _build_weight_shapes()

    header: dict = {}
    offset = 0
    data_parts: list[bytes] = []

    for name, shape in shapes.items():
        n_elems = 1
        for d in shape:
            n_elems *= d
        n_bytes = n_elems * 4
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + n_bytes],
        }
        if name == "readout.linear2.bias":
            # Single float32 element
            data_parts.append(struct.pack("<f", readout_linear2_bias))
        else:
            data_parts.append(b"\x00" * n_bytes)
        offset += n_bytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padded_len = (len(header_bytes) + 7) & ~7
    header_bytes = header_bytes.ljust(padded_len)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", padded_len))
        f.write(header_bytes)
        for part in data_parts:
            f.write(part)


@pytest.fixture(scope="session")
def known_weights_file():
    """Safetensors file with analytically known output: E = n_atoms * KNOWN_READOUT_BIAS."""
    _write_known_safetensors(_KNOWN_WEIGHTS, KNOWN_READOUT_BIAS)
    yield str(_KNOWN_WEIGHTS)
    _KNOWN_WEIGHTS.unlink(missing_ok=True)
