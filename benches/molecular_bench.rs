#[cfg(feature = "codspeed")]
use codspeed_criterion_compat::{black_box, BenchmarkId, Criterion as BenchCriterion, Throughput};
#[cfg(not(feature = "codspeed"))]
use criterion::{black_box, BenchmarkId, Criterion as BenchCriterion, Throughput};
use nalgebra::{DMatrix, Vector3};
use numpy::ndarray;
use valence::graph::MolecularGraph;
use valence::model::GNNModel;

fn setup_engine_data(
    n_atoms: usize,
    feat_dim: usize,
) -> (MolecularGraph, GNNModel, ndarray::Array2<f32>) {
    let atomic_numbers = vec![6; n_atoms];
    let positions = (0..n_atoms)
        .map(|_| {
            Vector3::new(
                rand::random::<f32>() * 20.0,
                rand::random::<f32>() * 20.0,
                rand::random::<f32>() * 20.0,
            )
        })
        .collect();

    let graph = MolecularGraph {
        atomic_numbers,
        positions,
    };
    let weights = DMatrix::from_element(feat_dim, feat_dim, 0.5);
    let model = GNNModel { weights };
    let features = ndarray::Array2::from_elem((n_atoms, feat_dim), 1.0);

    (graph, model, features)
}

fn bench_fused_inference_scaling(c: &mut BenchCriterion) {
    let mut group = c.benchmark_group("Valence_Engine_Performance");
    let feat_dim = 64;

    // Scaling analysis: Essential for showing PhysiocX how the engine handles large molecules
    for n in &[100, 500, 1000] {
        let (graph, model, feats) = setup_engine_data(*n, feat_dim);

        // N^2 throughput is the gold standard for GNN/Molecular interaction speed
        group.throughput(Throughput::Elements((*n * *n) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let result = graph.run_fused_with_model_internal(
                    &model,
                    &feats.view(),
                    black_box(5.0),
                    black_box(16),
                );
                black_box(result)
            });
        });
    }
    group.finish();
}

#[cfg(feature = "codspeed")]
codspeed_criterion_compat::criterion_group!(benches, bench_fused_inference_scaling);
#[cfg(feature = "codspeed")]
codspeed_criterion_compat::criterion_main!(benches);
#[cfg(not(feature = "codspeed"))]
criterion::criterion_group!(benches, bench_fused_inference_scaling);
#[cfg(not(feature = "codspeed"))]
criterion::criterion_main!(benches);
