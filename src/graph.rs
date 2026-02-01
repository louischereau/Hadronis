use crate::model::GNNModel;
use nalgebra::{DVector, Vector3};
use numpy::ndarray;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct MolecularGraph {
    #[pyo3(get)]
    pub atomic_numbers: Vec<i32>,
    pub positions: Vec<Vector3<f32>>,
}

#[pymethods]
impl MolecularGraph {
    #[new]
    pub fn new(atomic_numbers: Vec<i32>, positions: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let pos_view = positions.as_array();
        let pos: Vec<Vector3<f32>> = pos_view
            .axis_iter(ndarray::Axis(0))
            .map(|row| Vector3::new(row[0], row[1], row[2]))
            .collect();
        Ok(MolecularGraph {
            atomic_numbers,
            positions: pos,
        })
    }

    /// The flagship high-performance forward pass.
    /// Fuses: Neighbor Search -> RBF Expansion -> Aggregation -> Linear Transformation.
    pub fn run_fused_with_model(
        &self,
        model: &GNNModel,
        atom_features: PyReadonlyArray2<f32>,
        cutoff: f32,
        num_offsets: usize,
    ) -> Py<PyArray2<f32>> {
        let py = atom_features.py();
        let n = self.positions.len();
        let atom_view = atom_features.as_array();

        // 1. Core Computation: Search and Aggregate
        let aggregated_results = self.compute_core_fused(cutoff, num_offsets, atom_view);

        // 2. Linear Transformation and Output Formatting
        // Result = Weights * Aggregated_Features
        let results: Vec<Vec<f32>> = aggregated_results
            .into_par_iter()
            .map(|agg| {
                let updated_vec = &model.weights * agg;
                updated_vec.as_slice().to_vec()
            })
            .collect();

        // 3. Buffer Transfer to Python Memory
        let out_cols = results[0].len();
        let out_array = PyArray2::zeros(py, [n, out_cols], false);
        let mut out_view = unsafe { out_array.as_array_mut() };

        for (i, row) in results.into_iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                out_view[[i, j]] = val;
            }
        }
        out_array.into()
    }
}

impl MolecularGraph {
    /// Internal logic to handle the heavy O(N^2) math.
    /// This is the "Engine Room" of the project.
    fn compute_core_fused(
        &self,
        cutoff: f32,
        num_offsets: usize,
        atom_view: ndarray::ArrayView2<f32>,
    ) -> Vec<DVector<f32>> {
        let n = self.positions.len();
        let num_feats = atom_view.shape()[1];

        // Pre-calculate RBF constants to avoid repetitive math in the inner loop
        let centers: Vec<f32> = (0..num_offsets)
            .map(|i| (i as f32) * cutoff / (num_offsets as f32))
            .collect();
        let gamma = 0.5 / (cutoff / num_offsets as f32).powi(2);

        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut aggregated = DVector::zeros(num_feats);
                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    // Euclidean distance calculation
                    let dist = (self.positions[i] - self.positions[j]).norm();

                    if dist <= cutoff {
                        // Optimized RBF weight sum
                        let rbf_weight: f32 = centers
                            .iter()
                            .map(|&mu| (-(gamma * (dist - mu).powi(2))).exp())
                            .sum();

                        // Scatter-Add neighboring features into the local accumulator
                        for f in 0..num_feats {
                            aggregated[f] += rbf_weight * atom_view[[j, f]];
                        }
                    }
                }
                aggregated
            })
            .collect()
    }
}

// Inside src/graph.rs
impl MolecularGraph {
    pub fn run_fused_with_model_internal(
        &self,
        model: &GNNModel,
        atom_view: ndarray::ArrayView2<f32>,
        cutoff: f32,
        num_offsets: usize,
    ) -> Vec<DVector<f32>> {
        let aggregated_results = self.compute_core_fused(cutoff, num_offsets, atom_view);

        aggregated_results
            .into_iter()
            .map(|agg| &model.weights * agg)
            .collect()
    }
}
