#include <algorithm>
#include <numeric>
#include <vector>

struct EdgeGraph {
  std::vector<int> edge_src;
  std::vector<int> edge_dst;
  std::vector<float> edge_rbf;
  std::vector<float> edge_rvec; // [E, 3] per-edge unit vectors r_hat

  std::size_t num_edges() const { return edge_src.size(); }

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
