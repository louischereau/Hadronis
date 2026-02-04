use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use nalgebra::{DMatrix, Vector3};
use numpy::ndarray;
use valence::graph::MolecularGraph;
use valence::model::GNNModel;

/// Data structure to hold our pre-generated benchmark data
struct BenchData {
    graph: MolecularGraph,
    model: GNNModel,
    features: ndarray::Array2<f32>,
}

/// Setup function - everything here is excluded from the primary
/// instruction count/metrics, isolating the inference engine.
fn setup_inference(n_atoms: usize) -> BenchData {
    let feat_dim = 64;
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

    BenchData {
        graph,
        model,
        features,
    }
}

// --- Benchmark Definitions ---

#[library_benchmark]
#[bench::small(setup_inference(100))]
#[bench::medium(setup_inference(500))]
#[bench::large(setup_inference(1000))]
fn bench_fused_inference(data: BenchData) {
    // The benchmarked code: pure inference performance
    let _ = data
        .graph
        .run_fused_with_model_internal(&data.model, &data.features.view(), 5.0, 16);
}

// --- Grouping and Main ---

library_benchmark_group!(
    name = inference_benchmarks;
    config = LibraryBenchmarkConfig::default();
    benchmarks = bench_fused_inference
);

main!(library_benchmark_groups = inference_benchmarks);
