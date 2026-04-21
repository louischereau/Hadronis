#pragma once
#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

struct EdgeGraph {
  std::vector<int> edge_src;
  std::vector<int> edge_dst;
  std::vector<float> edge_rbf;
  std::vector<float> edge_rvec; // [E, 3] per-edge relative vectors r_ij
  std::vector<std::size_t>
      dst_offsets; // [n_atoms + 1] CSR row pointers into edge arrays

  std::size_t num_edges() const { return edge_src.size(); }

  // Build CSR dst_offsets for n_atoms atoms. Must be called after
  // sort_by_dst(). dst_offsets[i]..dst_offsets[i+1] is the half-open range of
  // incoming edges for atom i.
  void build_dst_offsets(int n_atoms) {
    dst_offsets.assign(static_cast<std::size_t>(n_atoms) + 1, 0);
    for (int dst : edge_dst)
      ++dst_offsets[static_cast<std::size_t>(dst) + 1];
    for (int i = 0; i < n_atoms; ++i)
      dst_offsets[static_cast<std::size_t>(i) + 1] +=
          dst_offsets[static_cast<std::size_t>(i)];
  }

  // Returns a span over the rvec blocks [begin*3 .. end*3) for atom i's
  // incoming edges.
  std::span<const float> incoming_rvec(int atom) const {
    const std::size_t begin = dst_offsets[static_cast<std::size_t>(atom)];
    const std::size_t end = dst_offsets[static_cast<std::size_t>(atom) + 1];
    return {edge_rvec.data() + begin * 3, (end - begin) * 3};
  }

  // Returns a span over the RBF blocks [begin*K .. end*K) for atom i's incoming
  // edges.
  std::span<const float> incoming_rbf(int atom, int n_rbf) const {
    const std::size_t begin = dst_offsets[static_cast<std::size_t>(atom)];
    const std::size_t end = dst_offsets[static_cast<std::size_t>(atom) + 1];
    return {edge_rbf.data() + begin * static_cast<std::size_t>(n_rbf),
            (end - begin) * static_cast<std::size_t>(n_rbf)};
  }

  // Sort edges by destination index to improve cache locality when
  // aggregating messages into destination atoms. All edge-aligned arrays
  // (src, dst, rbf blocks, rvec blocks) are permuted consistently.
  void sort_by_dst() {
    const std::size_t E = edge_src.size();
    if (E <= 1)
      return;

    std::vector<std::size_t> order(E);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return edge_dst[a] < edge_dst[b];
    });

    // Permute scalar edge fields
    std::vector<int> new_src(E);
    std::vector<int> new_dst(E);
    for (std::size_t i = 0; i < E; ++i) {
      const std::size_t idx = order[i];
      new_src[i] = edge_src[idx];
      new_dst[i] = edge_dst[idx];
    }

    edge_src.swap(new_src);
    edge_dst.swap(new_dst);

    // Permute RBF blocks [E, K]
    if (!edge_rbf.empty()) {
      const std::size_t K = edge_rbf.size() / E;
      std::vector<float> new_rbf(edge_rbf.size());
      for (std::size_t i = 0; i < E; ++i) {
        const std::size_t idx = order[i];
        const std::size_t src_off = idx * K;
        const std::size_t dst_off = i * K;
        std::copy_n(edge_rbf.begin() + src_off, K, new_rbf.begin() + dst_off);
      }
      edge_rbf.swap(new_rbf);
    }

    // Permute r_hat blocks [E, 3]
    if (!edge_rvec.empty()) {
      const std::size_t D = edge_rvec.size() / E; // typically 3
      std::vector<float> new_rvec(edge_rvec.size());
      for (std::size_t i = 0; i < E; ++i) {
        const std::size_t idx = order[i];
        const std::size_t src_off = idx * D;
        const std::size_t dst_off = i * D;
        std::copy_n(edge_rvec.begin() + src_off, D, new_rvec.begin() + dst_off);
      }
      edge_rvec.swap(new_rvec);
    }
  }
};
