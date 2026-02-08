use hadronis::batch::build_batched_neighbors;
use hadronis::model::GNNModel;
use nalgebra::DMatrix;
use numpy::ndarray;
use std::time::Instant;

fn main() {
    println!("--- Hadronis Profiling Session Start ---");

    // 1. Setup a large batch (2000 atoms)
    let n_atoms = 2000;
    let feat_dim = 64;
    let k = 16;
    let cutoff = 5.0;

    // Atomic numbers for each atom in batch
    let atomic_numbers_batch = ndarray::Array1::from_elem(n_atoms, 6);
    // Positions: shape (n_atoms, 3)
    let positions_batch = ndarray::Array2::from_shape_fn((n_atoms, 3), |_| rand::random::<f32>());
    // Features: shape (n_atoms, feat_dim)
    let features_batch = ndarray::Array2::from_elem((n_atoms, feat_dim), 1.0);

    // Model weights: shape (feat_dim, feat_dim)
    let weights = DMatrix::from_element(feat_dim, feat_dim, 0.5);
    let model = GNNModel { weights };

    // Build mol_ptrs for single batch
    let mol_ptrs = vec![0, n_atoms as i32];
    // Build batched neighbor list
    let (edge_src, edge_dst, edge_relpos) =
        build_batched_neighbors(&positions_batch.view(), &mol_ptrs, cutoff, k);

    println!("Engine initialized. Running 50 iterations for profiling...");
    let start = Instant::now();
    for i in 0..50 {
        let _result = model.run_batched(
            &atomic_numbers_batch.as_slice().unwrap(),
            &positions_batch.view(),
            &features_batch.view(),
            &edge_src,
            &edge_dst,
            &edge_relpos,
            &mol_ptrs,
            cutoff,
            k,
        );
        if i % 10 == 0 {
            println!("Iteration {i}...");
        }
    }
    let duration = start.elapsed();
    println!("--- Profiling Session Complete ---");
    println!("Total time: {:?}", duration);
}
