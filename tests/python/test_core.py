import hadronis
import numpy as np
import pytest

pytestmark = pytest.mark.no_codspeed


@pytest.fixture
def methane_system():
    atomic_numbers = np.array([6, 1, 1, 1, 1], dtype=np.int32)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # C
            [0.63, 0.63, 0.63],  # H1
            [-0.63, -0.63, 0.63],  # H2
            [-0.63, 0.63, -0.63],  # H3
            [0.63, -0.63, -0.63],  # H4
        ],
        dtype=np.float32,
    )
    return atomic_numbers, positions


def test_compile_returns_engine():
    engine = hadronis.compile("dummy-weights.bin")
    assert isinstance(engine, hadronis.Engine)


def test_predict_output_shape_and_dtype(methane_system):
    atomic_numbers, positions = methane_system
    engine = hadronis.compile("dummy-weights.bin")

    out = engine.predict(atomic_numbers, positions)

    assert out.shape == (atomic_numbers.shape[0],)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32


def test_predict_rejects_wrong_atomic_number_shape():
    engine = hadronis.compile("dummy-weights.bin")

    atomic_numbers = np.array([[1, 1]], dtype=np.int32)  # 2D instead of 1D
    positions = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        engine.predict(atomic_numbers, positions)


def test_predict_rejects_mismatched_positions_shape():
    engine = hadronis.compile("dummy-weights.bin")

    atomic_numbers = np.array([1, 1], dtype=np.int32)
    # Wrong last dimension (2 instead of 3)
    positions = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        engine.predict(atomic_numbers, positions)
