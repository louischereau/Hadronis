use memmap2::Mmap;
use nalgebra::DMatrix;
use numpy::ndarray;
use pyo3::prelude::*;
use safetensors::SafeTensors;
use std::fs::File;
use std::simd::Simd;

#[pyclass]
pub struct GNNModel {
    pub weights: DMatrix<f32>,
}

#[pymethods]
impl GNNModel {
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        // Open file
        let file = File::open(path)
            .map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Memory-map it
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        // Load safetensors from bytes
        let safetensors =
            SafeTensors::deserialize(&mmap).map_err(|e: safetensors::SafeTensorError| {
                pyo3::exceptions::PyIOError::new_err(e.to_string())
            })?;

        // Extract weights
        let weights_data = safetensors
            .tensor("weights")
            .map_err(|_| pyo3::exceptions::PyKeyError::new_err("No tensor 'weights' found"))?;

        let nrows = weights_data.shape()[0];
        let ncols = weights_data.shape()[1];
        let weights = nalgebra::DMatrix::from_row_slice(
            nrows,
            ncols,
            bytemuck::cast_slice(weights_data.data()),
        );

        Ok(Self { weights })
    }
}

impl GNNModel {
    /// Batched GNN inference kernel, refactored into helper functions
    #[allow(clippy::too_many_arguments)]
    pub fn run_batched(
        &self,
        _atomic_numbers: &[i32],
        _positions: &ndarray::ArrayView2<f32>,
        features: &ndarray::ArrayView2<f32>,
        edge_src: &[usize],
        edge_dst: &[usize],
        edge_relpos: &[[f32; 3]],
        _mol_ptrs: &[i32],
        cutoff: f32,
        num_rbf: usize,
    ) -> ndarray::Array2<f32> {
        let (n_atoms, n_feats) = features.dim();

        // Precompute RBF centers & gamma
        let centers = compute_rbf_centers(num_rbf, cutoff);
        let gamma = 0.5 / (cutoff / num_rbf as f32).powi(2);

        // Aggregate neighbor features with RBF weights
        let aggregated = aggregate_features(
            n_atoms,
            n_feats,
            features,
            edge_src,
            edge_dst,
            edge_relpos,
            &centers,
            gamma,
        );

        // Apply linear layer (weights)
        apply_linear_layer(&self.weights, &aggregated)
    }
}

/// Compute RBF centers
#[inline(always)]
fn compute_rbf_centers(num_rbf: usize, cutoff: f32) -> Vec<f32> {
    let step = cutoff / num_rbf as f32;
    (0..num_rbf).map(|i| i as f32 * step).collect()
}

/// Aggregate neighbor features weighted by RBF
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn aggregate_features(
    n_atoms: usize,
    n_feats: usize,
    features: &ndarray::ArrayView2<f32>,
    edge_src: &[usize],
    edge_dst: &[usize],
    edge_relpos: &[[f32; 3]],
    centers: &[f32],
    gamma: f32,
) -> ndarray::Array2<f32> {
    let mut aggregated = ndarray::Array2::<f32>::zeros((n_atoms, n_feats));

    for ((&src, &dst), rel) in edge_src.iter().zip(edge_dst.iter()).zip(edge_relpos.iter()) {
        let dist = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let rbf_sum = compute_rbf_sum(dist, centers, gamma);
        for f in 0..n_feats {
            aggregated[[src, f]] += rbf_sum * features[[dst, f]];
        }
    }

    aggregated
}

/// Compute the RBF expansion per edge using SIMD
#[inline(always)]
fn compute_rbf_sum(dist: f32, centers: &[f32], gamma: f32) -> f32 {
    const W: usize = 8;
    let mut sum = 0.0f32;

    let dist_simd = Simd::<f32, W>::splat(dist);
    let gamma_simd = Simd::<f32, W>::splat(gamma);

    let chunks = centers.len() / W;

    for c in 0..chunks {
        let start = c * W;
        let mu = Simd::<f32, W>::from_slice(&centers[start..start + W]);
        let diff = dist_simd - mu;
        let val = -(diff * diff * gamma_simd);
        sum += val.to_array().iter().map(|&x| x.exp()).sum::<f32>();
    }

    // handle remainder
    let start_rem = chunks * W;
    for &mu in centers.iter().skip(start_rem) {
        sum += (-(gamma * (dist - mu).powi(2))).exp();
    }

    sum
}

/// Apply linear layer (weights * aggregated features)
#[inline(always)]
fn apply_linear_layer(
    weights: &DMatrix<f32>,
    aggregated: &ndarray::Array2<f32>,
) -> ndarray::Array2<f32> {
    let weights_nd =
        ndarray::ArrayView2::from_shape((weights.nrows(), weights.ncols()), weights.as_slice())
            .unwrap();

    weights_nd.dot(aggregated)
}
