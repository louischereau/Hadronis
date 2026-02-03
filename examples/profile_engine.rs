use nalgebra::{DMatrix, Vector3};
use numpy::ndarray;
use std::time::Instant;
use valence::graph::MolecularGraph;
use valence::model::GNNModel;

fn main() {
    println!("--- Valence Profiling Session Start ---");

    // 1. Setup a large system (2000 atoms) to saturate all CPU cores
    let n_atoms = 2000;
    let feat_dim = 64;

    let atomic_numbers = vec![6; n_atoms];
    let positions = (0..n_atoms)
        .map(|_| Vector3::new(rand::random(), rand::random(), rand::random()))
        .collect();

    let graph = MolecularGraph {
        atomic_numbers,
        positions,
    };
    let weights = DMatrix::from_element(feat_dim, feat_dim, 0.5);
    let model = GNNModel { weights };
    let feats = ndarray::Array2::from_elem((n_atoms, feat_dim), 1.0);

    println!("Engine initialized. Running 50 iterations for profiling...");

    // 2. Run a sustained loop
    let start = Instant::now();
    for i in 0..50 {
        // This is the "Zone" Tracy will monitor
        let _result = graph.run_fused_with_model_internal(&model, &feats.view(), 5.0, 16);
        if i % 10 == 0 {
            println!("Iteration {i}...");
        }
    }

    let duration = start.elapsed();
    println!("--- Profiling Session Complete ---");
    println!("Total time: {duration:?}");
}
