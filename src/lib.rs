use pyo3::prelude::*;
// Declare the modules
pub mod batch;
pub mod graph;
pub mod model;

// Bring the structs into scope
use crate::batch::MolecularBatch;
use crate::graph::MolecularGraph;
use crate::model::GNNModel;

#[pymodule]
fn _lowlevel(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MolecularGraph>()?;
    m.add_class::<GNNModel>()?;
    m.add_class::<MolecularBatch>()?;
    Ok(())
}
