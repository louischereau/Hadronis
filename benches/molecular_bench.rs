#[cfg(feature = "codspeed")]
use codspeed_criterion_compat::{black_box, BenchmarkId, Criterion as BenchCriterion, Throughput};
#[cfg(not(feature = "codspeed"))]
use criterion::{black_box, BenchmarkId, Criterion as BenchCriterion, Throughput};
use hadronis::core::connectivity::build_batched_neighbors;
use hadronis::core::model::GNNModel;
use nalgebra::DMatrix;
use numpy::ndarray;

fn setup_batch_data(
    n_atoms: usize,
    feat_dim: usize,
) -> (
    ndarray::Array1<i32>,
    ndarray::Array2<f32>,
    ndarray::Array1<i32>,
    ndarray::Array2<f32>,
    GNNModel,
) {
    let atomic_numbers = ndarray::Array1::from_elem(n_atoms, 6);
    let positions = ndarray::Array2::from_shape_fn((n_atoms, 3), |_| rand::random::<f32>() * 20.0);
    let features = ndarray::Array2::from_elem((n_atoms, feat_dim), 1.0);
    let mol_ptrs = ndarray::Array1::from_vec(vec![0, n_atoms as i32]);
    let weights = DMatrix::from_element(feat_dim, feat_dim, 0.5);
    let model = GNNModel { weights };
    (atomic_numbers, positions, mol_ptrs, features, model)
}

fn bench_batch_inference_scaling(c: &mut BenchCriterion) {
    let mut group = c.benchmark_group("Hadronis_Batch_Performance");
    let feat_dim = 64;
    let num_rbf = 64;
    for n in &[100, 500, 1000] {
        let (atomic_numbers, positions, mol_ptrs, features, model) = setup_batch_data(*n, feat_dim);
        let (edge_src, edge_dst, edge_relpos_vec) =
            build_batched_neighbors(&positions.view(), mol_ptrs.as_slice().unwrap(), 5.0, 16)
                .unwrap();
        let edge_relpos = ndarray::Array2::from_shape_vec(
            (edge_relpos_vec.len(), 3),
            edge_relpos_vec.into_iter().flatten().collect(),
        )
        .unwrap();
        group.throughput(Throughput::Elements((*n * *n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
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
codspeed_criterion_compat::criterion_group!(benches, bench_batch_inference_scaling);
#[cfg(feature = "codspeed")]
codspeed_criterion_compat::criterion_main!(benches);
#[cfg(not(feature = "codspeed"))]
criterion::criterion_group!(benches, bench_batch_inference_scaling);
#[cfg(not(feature = "codspeed"))]
criterion::criterion_main!(benches);
