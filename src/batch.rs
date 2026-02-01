use crate::{graph::MolecularGraph, model::GNNModel};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct MolecularBatch {
    pub graphs: Vec<MolecularGraph>,
}

#[pymethods]
impl MolecularBatch {
    #[new]
    pub fn new(graphs: Vec<MolecularGraph>) -> Self {
        MolecularBatch { graphs }
    }

    pub fn run_batch_inference(
        &self,
        model: &GNNModel,
        all_atom_features: Vec<PyReadonlyArray2<f32>>,
        cutoff: f32,
        num_offsets: usize,
    ) -> Vec<Py<PyArray2<f32>>> {
        // Step 1: Extract to owned arrays (sequential, safe)
        let owned_atom_features: Vec<_> = all_atom_features
            .iter()
            .map(|pyarr| pyarr.as_array().to_owned())
            .collect();

        // Step 2: Parallel processing on owned data
        self.graphs
            .par_iter()
            .zip(owned_atom_features.into_par_iter())
            .map(|(graph, feat_array)| {
                Python::attach(|py| {
                    let py_feat = PyArray2::from_array(py, &feat_array).readonly();
                    graph.run_fused_with_model(model, py_feat, cutoff, num_offsets)
                })
            })
            .collect()
    }
}
