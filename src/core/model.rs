use nalgebra::DMatrix;
use numpy::ndarray::{Array2, ArrayView2};
use std::simd::num::SimdFloat;
use std::simd::Simd;

/// SIMD-optimized GNN model with fast RBF
pub struct GNNModel {
    pub weights: DMatrix<f32>,
}

impl GNNModel {
    #[allow(clippy::too_many_arguments)]
    pub fn run_batched(
        &self,
        _atomic_numbers: &[i32],
        _positions: &ArrayView2<f32>,
        features: &ArrayView2<f32>,
        edge_src: &[usize],
        edge_dst: &[usize],
        edge_relpos: &Array2<f32>, // shape [n_edges,3]
        _mol_ptrs: &[i32],
        cutoff: f32,
        num_rbf: usize,
    ) -> Array2<f32> {
        let (n_atoms, n_feats) = features.dim();

        let centers = compute_rbf_centers(num_rbf, cutoff);
        let gamma = 0.5 / (cutoff / num_rbf as f32).powi(2);

        // Aggregate neighbor features with SIMD + fast exp
        let aggregated = aggregate_features_simd(
            n_atoms,
            n_feats,
            features,
            edge_src,
            edge_dst,
            &edge_relpos.view(),
            &centers,
            gamma,
        );

        // Linear layer
        apply_linear_layer(&self.weights, &aggregated)
    }
}

/// Compute RBF centers
#[inline(always)]
fn compute_rbf_centers(num_rbf: usize, cutoff: f32) -> Vec<f32> {
    let step = cutoff / num_rbf as f32;
    (0..num_rbf).map(|i| i as f32 * step).collect()
}

/// SIMD neighbor aggregation with fast exp approximation
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn aggregate_features_simd(
    n_atoms: usize,
    n_feats: usize,
    features: &ArrayView2<f32>,
    edge_src: &[usize],
    edge_dst: &[usize],
    edge_relpos: &ArrayView2<f32>,
    centers: &[f32],
    gamma: f32,
) -> Array2<f32> {
    let mut aggregated = Array2::<f32>::zeros((n_atoms, n_feats));

    for (i, (&src, &dst)) in edge_src.iter().zip(edge_dst.iter()).enumerate() {
        let rel = edge_relpos.row(i);
        let dist = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let rbf_sum = compute_rbf_sum(dist, centers, gamma);
        for f in 0..n_feats {
            aggregated[[src, f]] += rbf_sum * features[[dst, f]];
        }
    }

    aggregated
}

/// Compute RBF sum using SIMD + fast approximate exp
#[inline(always)]
fn compute_rbf_sum(dist: f32, centers: &[f32], gamma: f32) -> f32 {
    use std::simd::Simd;
    const W: usize = 8;
    let dist_simd = Simd::<f32, W>::splat(dist);
    let gamma_simd = Simd::<f32, W>::splat(gamma);

    let mut sum = 0.0;
    let chunks = centers.len() / W;

    for c in 0..chunks {
        let start = c * W;
        let mu = Simd::<f32, W>::from_slice(&centers[start..start + W]);
        let diff = dist_simd - mu;
        let val = -(diff * diff * gamma_simd);
        sum += fast_exp_simd(val).reduce_sum();
    }

    for &mu in centers.iter().skip(chunks * W) {
        sum += fast_exp_scalar(-(gamma * (dist - mu).powi(2)));
    }

    sum
}

/// Fast approximate exp for SIMD vectors
#[inline(always)]
fn fast_exp_simd<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    // 5th order polynomial approximation: exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let x5 = x4 * x;
    Simd::splat(1.0)
        + x
        + x2 * Simd::splat(0.5)
        + x3 * Simd::splat(1.0 / 6.0)
        + x4 * Simd::splat(1.0 / 24.0)
        + x5 * Simd::splat(1.0 / 120.0)
}

/// Fast scalar exp (same approximation)
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    1.0 + x + x * x * 0.5 + x.powi(3) / 6.0 + x.powi(4) / 24.0 + x.powi(5) / 120.0
}

/// Linear layer
#[inline(always)]
fn apply_linear_layer(weights: &DMatrix<f32>, aggregated: &Array2<f32>) -> Array2<f32> {
    let (n_feats_out, n_feats_in) = weights.shape();
    let n_atoms = aggregated.nrows();
    let mut output = Array2::<f32>::zeros((n_atoms, n_feats_out));

    let weights_flat = weights.as_slice();
    let aggregated_flat = aggregated.as_slice().unwrap();
    let output_flat = output.as_slice_mut().unwrap();
    const W: usize = 8;

    for atom in 0..n_atoms {
        for row_out in 0..n_feats_out {
            let mut sum = 0.0f32;
            let mut i = 0;
            while i + W <= n_feats_in {
                let w_chunk = Simd::<f32, W>::from_slice(
                    &weights_flat[row_out * n_feats_in + i..row_out * n_feats_in + i + W],
                );
                let f_chunk = Simd::<f32, W>::from_slice(
                    &aggregated_flat[atom * n_feats_in + i..atom * n_feats_in + i + W],
                );
                sum += (w_chunk * f_chunk).reduce_sum();
                i += W;
            }
            while i < n_feats_in {
                sum +=
                    weights_flat[row_out * n_feats_in + i] * aggregated_flat[atom * n_feats_in + i];
                i += 1;
            }
            output_flat[atom * n_feats_out + row_out] = sum;
        }
    }

    output
}
