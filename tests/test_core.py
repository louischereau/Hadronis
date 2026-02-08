# ruff: noqa: I001
import os
import numpy as np
import pytest
import hadronis


@pytest.fixture(scope="session", autouse=True)
def create_mock_safetensors():
    try:
        from safetensors.numpy import save
    except ImportError:
        pytest.skip("safetensors package not installed")
    path = "tests/mock_weights.safetensors"
    if not os.path.exists(path):
        weights = np.eye(8, dtype=np.float32)
        save({"weights": weights}, path)


@pytest.fixture
def methane_data():
    return {
        "atomic_numbers": [6, 1, 1, 1, 1],
        "positions": [
            [0.0, 0.0, 0.0],  # C
            [0.63, 0.63, 0.63],  # H1
            [-0.63, -0.63, 0.63],  # H2
            [-0.63, 0.63, -0.63],  # H3
            [0.63, -0.63, -0.63],  # H4
        ],
    }


def test_batch_input_shapes(methane_data):
    # Prepare batch arrays for a single molecule
    atomic_numbers = np.array(methane_data["atomic_numbers"], dtype=np.int32)
    positions = np.array(methane_data["positions"], dtype=np.float32)
    mol_ptrs = np.array([0, len(atomic_numbers)], dtype=np.int32)
    features = np.ones((len(atomic_numbers), 16), dtype=np.float32)
    assert atomic_numbers.shape[0] == positions.shape[0]
    assert features.shape[0] == positions.shape[0]
    assert mol_ptrs.shape[0] == 2


def test_batch_inference_single_molecule(methane_data):
    atomic_numbers = np.array(methane_data["atomic_numbers"], dtype=np.int32)
    positions = np.array(methane_data["positions"], dtype=np.float32)
    mol_ptrs = np.array([0, len(atomic_numbers)], dtype=np.int32)
    features = np.ones((len(atomic_numbers), 16), dtype=np.float32)
    engine = hadronis.HadronisEngine("tests/mock_weights.safetensors")
    output = engine.run_batch_inference(
        atomic_numbers,
        positions,
        mol_ptrs,
        features,
        cutoff=1.2,
        k=8,
    )
    assert output.shape == (len(atomic_numbers), 16)
    assert isinstance(output, np.ndarray)


def test_batch_inference_two_atoms():
    atomic_numbers = np.array([1, 1], dtype=np.int32)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    mol_ptrs = np.array([0, 2], dtype=np.int32)
    features = np.ones((2, 8), dtype=np.float32)
    engine = hadronis.HadronisEngine("tests/mock_weights.safetensors")
    output = engine.run_batch_inference(
        atomic_numbers,
        positions,
        mol_ptrs,
        features,
        cutoff=1.5,
        k=4,
    )
    assert np.all(output > 0)
    assert output.shape == (2, 8)


def test_batch_inference_multiple_molecules():
    # Two molecules, each with two atoms
    atomic_numbers = np.array([1, 1, 1, 1], dtype=np.int32)
    positions = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 2]], dtype=np.float32)
    mol_ptrs = np.array([0, 2, 4], dtype=np.int32)
    features = np.ones((4, 8), dtype=np.float32)
    engine = hadronis.HadronisEngine("tests/mock_weights.safetensors")
    output = engine.run_batch_inference(
        atomic_numbers,
        positions,
        mol_ptrs,
        features,
        cutoff=1.5,
        k=4,
    )
    assert output.shape == (4, 8)
    # First molecule should have non-zero results (within cutoff)
    assert np.sum(output[:2]) > 0
    # Second molecule should have zero results (2.0 > 1.5 cutoff)
    assert np.sum(output[2:]) == 0


def test_batch_inference_consistency():
    # Consistency test using batch arrays
    atomic_numbers = np.array([1, 1, 1, 1], dtype=np.int32)
    positions = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 2]], dtype=np.float32)
    mol_ptrs = np.array([0, 2, 4], dtype=np.int32)
    features = np.ones((4, 8), dtype=np.float32)
    engine = hadronis.HadronisEngine("tests/mock_weights.safetensors")
    output = engine.run_batch_inference(
        atomic_numbers,
        positions,
        mol_ptrs,
        features,
        cutoff=1.5,
        k=4,
    )
    assert output.shape == (4, 8)
    # First molecule should have non-zero results (within cutoff)
    assert np.sum(output[:2]) > 0
    # Second molecule should have zero results (2.0 > 1.5 cutoff)
    assert np.sum(output[2:]) == 0
