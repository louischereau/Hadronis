use crate::{graph::MolecularGraph, model::GNNModel};
use numpy::{ndarray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct MolecularBatch {
    pub graphs: Vec<MolecularGraph>,
}

#[pymethods]
impl MolecularBatch {
    #[must_use]
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(graphs: Vec<MolecularGraph>) -> Self {
        MolecularBatch { graphs }
    }
    /// Runs batch inference for all graphs in the batch.
    ///
    /// # Panics
    /// Panics if the feature array row count does not match atom count.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn run_batch_inference(
        &self,
        model: &GNNModel,
        all_atom_features: Vec<PyReadonlyArray2<f32>>,
        cutoff: f32,
        num_offsets: usize,
    ) -> Vec<Py<PyArray2<f32>>> {
        #[allow(clippy::needless_pass_by_value)]
        // Step 1: Extract to owned arrays (sequential, safe)
        let owned_atom_features: Vec<_> = all_atom_features
            .iter()
            .map(|pyarr| pyarr.as_array().to_owned())
            .collect();

        // Step 2: Pure Rust batch computation
        let batch_results: Vec<ndarray::Array2<f32>> = self
            .graphs
            .par_iter()
            .zip(owned_atom_features.par_iter())
            .map(|(graph, feat_array)| {
                assert_eq!(
                    feat_array.shape()[0],
                    graph.atomic_numbers.len(),
                    "Feature array row count does not match atom count"
                );
                let fused_result = graph.run_fused_with_model_internal(
                    model,
                    &feat_array.view(),
                    cutoff,
                    num_offsets,
                );
                let n_atoms = graph.atomic_numbers.len();
                let n_feats = feat_array.shape()[1];
                let mut arr = ndarray::Array2::<f32>::zeros((n_atoms, n_feats));
                for (row_idx, dv) in fused_result.into_iter().enumerate() {
                    for (col_idx, val) in dv.iter().enumerate() {
                        arr[[row_idx, col_idx]] = *val;
                    }
                }
                arr
            })
            .collect();

        // Step 3: Convert results to Python objects inside a single GIL block
        Python::attach(|py| {
            batch_results
                .into_iter()
                .map(|arr| PyArray2::from_array(py, &arr).into())
                .collect::<Vec<_>>()
        })
    }
}
