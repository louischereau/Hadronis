use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::{DMatrix, Vector3};
use numpy::ndarray;
use valence::graph::MolecularGraph;
use valence::model::GNNModel;

/// Helper to generate a realistic mock environment entirely in Rust.
/// This bypasses PyO3/GIL overhead to measure the raw engine speed.
fn setup_engine_data(
    n_atoms: usize,
    feat_dim: usize,
) -> (MolecularGraph, GNNModel, ndarray::Array2<f32>) {
    let atomic_numbers = vec![6; n_atoms];

    // Distribute atoms randomly in a 3D box
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

    // Initialize weights (the learned parameters)
    let weights = DMatrix::from_element(feat_dim, feat_dim, 0.5);
    let model = GNNModel { weights };

    // Mock input features (e.g., atom types or embeddings)
    let features = ndarray::Array2::from_elem((n_atoms, feat_dim), 1.0);

    (graph, model, features)
}

fn bench_fused_inference_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Valence_Engine_Performance");
    let feat_dim = 64;

    for n in &[100, 500, 1000] {
        let (graph, model, feats) = setup_engine_data(*n, feat_dim);

        // --- NEW: Throughput Calculation ---
        // We define throughput as the number of pair-wise interactions (N^2)
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

criterion_group!(benches, bench_fused_inference_scaling);
criterion_main!(benches);
