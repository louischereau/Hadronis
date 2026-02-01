use nalgebra::DMatrix;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

#[pyclass]
pub struct GNNModel {
    pub weights: DMatrix<f32>,
}

#[pymethods]
impl GNNModel {
    #[new]
    pub fn new(weights_raw: PyReadonlyArray2<f32>) -> Self {
        let view = weights_raw.as_array();
        let (rows, cols) = (view.shape()[0], view.shape()[1]);
        // Convert NumPy layout to nalgebra DMatrix
        let weights = DMatrix::from_iterator(rows, cols, view.iter().cloned());
        GNNModel { weights }
    }
}
