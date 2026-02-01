import numpy as np
import pytest
import valence


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


def test_pydantic_validation(methane_data):
    mol = valence.Molecule(**methane_data)
    assert len(mol.atomic_numbers) == 5

    # Test mismatch error
    with pytest.raises(ValueError):
        valence.Molecule(atomic_numbers=[6], positions=[[0, 0, 0], [1, 1, 1]])


def test_fused_inference_output_shape(methane_data):
    mol = valence.Molecule(**methane_data)
    # Feature dim = 16
    feats = np.ones((5, 16), dtype=np.float32)

    # run_fused_parallel returns a [NumAtoms, NumFeats] array
    output = mol.predict(feats, cutoff=1.2, k=8)

    assert output.shape == (5, 16)
    assert isinstance(output, np.ndarray)


def test_engine_weights_application():
    # Setup two atoms: if we have 1s as features and 1s as weights,
    # the output should reflect the RBF-weighted sum.
    mol = valence.Molecule(
        atomic_numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Identity-like weights (Diagonal 1s)
    weights = np.eye(8).astype(np.float32)
    np.save("test_weights.npy", weights)

    engine = valence.ValenceEngine("test_weights.npy")
    feats = np.ones((2, 8), dtype=np.float32)

    # Run inference
    output = engine.run(mol, feats, cutoff=1.5, k=4)

    # Because distance is 1.0 and cutoff is 1.5, there IS an interaction.
    # Output should be non-zero
    assert np.all(output > 0)
    assert output.shape == (2, 8)


def test_batch_inference_consistency():
    weights = np.eye(8).astype(np.float32)
    np.save("test_weights.npy", weights)
    engine = valence.ValenceEngine("test_weights.npy")

    mols = [
        valence.Molecule(atomic_numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]]),
        valence.Molecule(
            atomic_numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 2]]
        ),  # Out of cutoff
    ]
    feats = [np.ones((2, 8), dtype=np.float32) for _ in range(2)]

    results = engine.predict_batch(mols, feats, cutoff=1.5)

    assert len(results) == 2
    # First molecule should have non-zero results (within cutoff)
    assert np.sum(results[0]) > 0
    # Second molecule should have zero results (2.0 > 1.5 cutoff)
    assert np.sum(results[1]) == 0
