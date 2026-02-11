use numpy::ndarray;
use rayon::prelude::*;
use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::Simd;

type NeighborResult = Result<(Vec<usize>, Vec<usize>, Vec<[f32; 3]>), &'static str>;

pub fn build_batched_neighbors(
    positions: &ndarray::ArrayView2<f32>,
    mol_ptrs: &[i32],
    cutoff: f32,
    k: usize,
) -> NeighborResult {
    let pos_slice = positions
        .as_slice()
        .ok_or("positions must be contiguous [N,3]")?;
    let edge_data: Vec<_> = (0..mol_ptrs.len() - 1)
        .into_par_iter()
        .map(|mol_idx| {
            let start = mol_ptrs[mol_idx] as usize;
            let end = mol_ptrs[mol_idx + 1] as usize;
            build_mol_neighbors(start, end, k, cutoff * cutoff, pos_slice)
        })
        .collect();
    let total_edges: usize = edge_data.iter().map(|(s, _, _)| s.len()).sum();
    let mut edge_src = Vec::with_capacity(total_edges);
    let mut edge_dst = Vec::with_capacity(total_edges);
    let mut edge_relpos = Vec::with_capacity(total_edges);
    for (s, d, r) in edge_data {
        edge_src.extend(s);
        edge_dst.extend(d);
        edge_relpos.extend(r);
    }
    Ok((edge_src, edge_dst, edge_relpos))
}

fn build_mol_neighbors(
    start: usize,
    end: usize,
    k: usize,
    cutoff2: f32,
    pos_slice: &[f32],
) -> (Vec<usize>, Vec<usize>, Vec<[f32; 3]>) {
    let n = end - start;
    let mut local_src = Vec::with_capacity(n * k);
    let mut local_dst = Vec::with_capacity(n * k);
    let mut local_relpos = Vec::with_capacity(n * k);
    for i in 0..n {
        let neighbors = compute_relpos_distances(pos_slice, i, start, n, cutoff2, k);
        let global_i = start + i;
        for (j, _dist2, rel) in neighbors {
            local_src.push(global_i);
            local_dst.push(j);
            local_relpos.push(rel);
        }
    }
    (local_src, local_dst, local_relpos)
}

fn compute_relpos_distances(
    pos_slice: &[f32],
    i: usize,
    start: usize,
    n: usize,
    cutoff2: f32,
    k: usize,
) -> Vec<(usize, f32, [f32; 3])> {
    const MAX_K: usize = 64;
    let mut best = [(usize::MAX, f32::INFINITY, [0.0; 3]); MAX_K];
    let mut best_count = 0;
    let base_i = (start + i) * 3;
    let pos_ix = pos_slice[base_i];
    let pos_iy = pos_slice[base_i + 1];
    let pos_iz = pos_slice[base_i + 2];
    const LANES: Simd<i32, 8> = Simd::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut j = 0;
    while j + 8 <= n {
        let idxs = LANES + Simd::splat(j as i32);
        let global_js = idxs + Simd::splat(start as i32);
        let valid = idxs.simd_ne(Simd::splat(i as i32));
        let rel_x = Simd::<f32, 8>::from_array([
            pos_slice[(global_js[0] as usize) * 3] - pos_ix,
            pos_slice[(global_js[1] as usize) * 3] - pos_ix,
            pos_slice[(global_js[2] as usize) * 3] - pos_ix,
            pos_slice[(global_js[3] as usize) * 3] - pos_ix,
            pos_slice[(global_js[4] as usize) * 3] - pos_ix,
            pos_slice[(global_js[5] as usize) * 3] - pos_ix,
            pos_slice[(global_js[6] as usize) * 3] - pos_ix,
            pos_slice[(global_js[7] as usize) * 3] - pos_ix,
        ]);
        let rel_y = Simd::<f32, 8>::from_array([
            pos_slice[(global_js[0] as usize) * 3 + 1] - pos_iy,
            pos_slice[(global_js[1] as usize) * 3 + 1] - pos_iy,
            pos_slice[(global_js[2] as usize) * 3 + 1] - pos_iy,
            pos_slice[(global_js[3] as usize) * 3 + 1] - pos_iy,
            pos_slice[(global_js[4] as usize) * 3 + 1] - pos_iy,
            pos_slice[(global_js[5] as usize) * 3 + 1] - pos_iy,
            pos_slice[(global_js[6] as usize) * 3 + 1] - pos_iy,
            pos_slice[(global_js[7] as usize) * 3 + 1] - pos_iy,
        ]);
        let rel_z = Simd::<f32, 8>::from_array([
            pos_slice[(global_js[0] as usize) * 3 + 2] - pos_iz,
            pos_slice[(global_js[1] as usize) * 3 + 2] - pos_iz,
            pos_slice[(global_js[2] as usize) * 3 + 2] - pos_iz,
            pos_slice[(global_js[3] as usize) * 3 + 2] - pos_iz,
            pos_slice[(global_js[4] as usize) * 3 + 2] - pos_iz,
            pos_slice[(global_js[5] as usize) * 3 + 2] - pos_iz,
            pos_slice[(global_js[6] as usize) * 3 + 2] - pos_iz,
            pos_slice[(global_js[7] as usize) * 3 + 2] - pos_iz,
        ]);
        let dist2 = rel_x * rel_x + rel_y * rel_y + rel_z * rel_z;
        let within = dist2.simd_le(Simd::splat(cutoff2));
        let active_mask = valid & within;
        let mut mask = active_mask.to_bitmask();
        while mask != 0 {
            let lane = mask.trailing_zeros() as usize;
            let global_j = global_js[lane] as usize;
            let rel = [rel_x[lane], rel_y[lane], rel_z[lane]];
            insert_best(&mut best, &mut best_count, k, global_j, dist2[lane], rel);
            mask &= mask - 1;
        }
        j += 8;
    }
    for idx in j..n {
        if idx == i {
            continue;
        }
        let global_j = start + idx;
        let base_j = global_j * 3;
        let rel = [
            pos_slice[base_j] - pos_ix,
            pos_slice[base_j + 1] - pos_iy,
            pos_slice[base_j + 2] - pos_iz,
        ];
        let dist2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
        if dist2 > cutoff2 {
            continue;
        }
        insert_best(&mut best, &mut best_count, k, global_j, dist2, rel);
    }
    best[..best_count.min(k)].to_vec()
}

fn insert_best(
    best: &mut [(usize, f32, [f32; 3])],
    best_count: &mut usize,
    k: usize,
    idx: usize,
    dist2: f32,
    rel: [f32; 3],
) {
    if *best_count < k {
        best[*best_count] = (idx, dist2, rel);
        *best_count += 1;
        if *best_count == k {
            best[..k].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
    } else if dist2 < best[k - 1].1 {
        best[k - 1] = (idx, dist2, rel);
        let mut swap_idx = k - 1;
        while swap_idx > 0 && best[swap_idx].1 < best[swap_idx - 1].1 {
            best.swap(swap_idx, swap_idx - 1);
            swap_idx -= 1;
        }
    }
}
