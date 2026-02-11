#[cfg(feature = "codspeed")]
use codspeed_criterion_compat::{black_box, BenchmarkId, Criterion as BenchCriterion};
#[cfg(not(feature = "codspeed"))]
use criterion::{black_box, BenchmarkId, Criterion as BenchCriterion};

use hadronis::core::connectivity::build_batched_neighbors;
use hadronis::core::model::GNNModel;
use nalgebra::DMatrix;
use numpy::ndarray;

fn generate_batch_data(
    size: usize,
    feat_dim: usize,
) -> (
    ndarray::Array1<i32>,
    ndarray::Array2<f32>,
    ndarray::Array1<i32>,
    ndarray::Array2<f32>,
    GNNModel,
) {
    let atomic_numbers = ndarray::Array1::from_elem(size, 6);
    let positions = ndarray::Array2::from_shape_fn((size, 3), |_| rand::random::<f32>());
    let features = ndarray::Array2::from_elem((size, feat_dim), 1.0);
    let mol_ptrs = ndarray::Array1::from_vec(vec![0, size as i32]);
    let weights = DMatrix::from_element(feat_dim, feat_dim, 0.5);
    let model = GNNModel { weights };
    (atomic_numbers, positions, mol_ptrs, features, model)
}

fn bench_scaling(c: &mut BenchCriterion) {
    let mut group = c.benchmark_group("Inference Scaling");
    let feat_dim = 64;
    let num_rbf = 64;
    for &size in &[128, 256, 512, 1024] {
        let (atomic_numbers, positions, mol_ptrs, features, model) =
            generate_batch_data(size, feat_dim);
        let (edge_src, edge_dst, edge_relpos_vec) =
            build_batched_neighbors(&positions.view(), mol_ptrs.as_slice().unwrap(), 5.0, 16)
                .unwrap();
        let edge_relpos = ndarray::Array2::from_shape_vec(
            (edge_relpos_vec.len(), 3),
            edge_relpos_vec.into_iter().flatten().collect(),
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_s| {
            b.iter(|| {
                let result = model.run_batched(
                    atomic_numbers.as_slice().unwrap(),
                    &positions.view(),
                    &features.view(),
                    &edge_src,
                    &edge_dst,
                    &edge_relpos,
                    mol_ptrs.as_slice().unwrap(),
                    5.0,
                    num_rbf,
                );
                black_box(result)
            });
        });
    }
    group.finish();
}

#[cfg(feature = "codspeed")]
codspeed_criterion_compat::criterion_group!(benches, bench_scaling);
#[cfg(feature = "codspeed")]
codspeed_criterion_compat::criterion_main!(benches);
#[cfg(not(feature = "codspeed"))]
criterion::criterion_group!(benches, bench_scaling);
#[cfg(not(feature = "codspeed"))]
criterion::criterion_main!(benches);
