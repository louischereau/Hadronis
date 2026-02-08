#![feature(portable_simd)]
use pyo3::prelude::*;
// Declare the modules
pub mod batch;
pub mod model;

// Bring the structs into scope
use crate::batch::run_batch_inference;
use crate::model::GNNModel;

#[pymodule]
fn _lowlevel(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_batch_inference, m)?)?;
    m.add_class::<GNNModel>()?;
    Ok(())
}
