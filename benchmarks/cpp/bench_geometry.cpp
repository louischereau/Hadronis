#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../../src/neighbors/cell_list.hpp"
#include "../../src/neighbors/graph_builder.hpp"
#include "../../src/neighbors/rbf.hpp"
#include "models/vec3.hpp"

using Clock = std::chrono::high_resolution_clock;

namespace {

std::vector<Vec3> make_random_positions(int N, float box_size) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, box_size);

  std::vector<Vec3> pos;
  pos.reserve(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    pos.push_back(Vec3{dist(rng), dist(rng), dist(rng)});
  }
  return pos;
}

void bench_cell_list(int N, float box_size, float r_cut, int iters) {
  auto pos = make_random_positions(N, box_size);
  CellList cl(N, box_size, r_cut);

  // Warmup
  cl.build(pos);

  auto start = Clock::now();
  for (int i = 0; i < iters; ++i) {
    cl.build(pos);
  }
  auto end = Clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "CellList::build N=" << N << " iters=" << iters << " time=" << ms
            << " ms (" << (ms * 1e6 / (N * iters)) << " ns/particle)\n";
}

void bench_rbf(int num_rbf, float cutoff, int num_distances, int iters) {
  RadialBasis rb(num_rbf, cutoff);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, cutoff);

  std::vector<float> ds(num_distances);
  for (int i = 0; i < num_distances; ++i)
    ds[i] = dist(rng);

  std::vector<float> out;
  // Warmup: simulate typical usage where we append all distances into
  // a single pre-reserved buffer.
  out.clear();
  out.reserve(static_cast<std::size_t>(num_distances) * rb.centers.size());
  for (int i = 0; i < num_distances; ++i)
    rb.expand_append(ds[i], out);

  auto start = Clock::now();
  for (int it = 0; it < iters; ++it) {
    out.clear();
    out.reserve(static_cast<std::size_t>(num_distances) * rb.centers.size());
    for (int i = 0; i < num_distances; ++i)
      rb.expand_append(ds[i], out);
  }
  auto end = Clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  double calls = static_cast<double>(num_distances) * iters;
  std::cout << "RadialBasis::expand_append num_rbf=" << num_rbf
            << " distances=" << num_distances << " iters=" << iters
            << " time=" << ms << " ms (" << (ms * 1e3 / calls) << " us/call)\n";
}

void bench_graph_builder(int N, float box_size, float r_cut, float r_skin,
                         int num_rbf, int iters) {
  auto pos = make_random_positions(N, box_size);
  GraphBuilder gb(N, box_size, r_cut, r_skin, num_rbf);

  // Warmup
  gb.build(pos);

  auto start = Clock::now();
  for (int i = 0; i < iters; ++i) {
    gb.build(pos);
  }
  auto end = Clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  double total_edges = static_cast<double>(gb.edge_graph.num_edges());
  std::cout << "GraphBuilder::build N=" << N << " iters=" << iters
            << " time=" << ms << " ms (" << (ms * 1e6 / (N * iters))
            << " ns/particle, approx "
            << (total_edges > 0.0 ? (ms * 1e6 / (total_edges * iters)) : 0.0)
            << " ns/edge)\n";
}

} // namespace

int main() {
  const int N = 10000;
  const float box_size = 50.0f;
  const float r_cut = 5.0f;
  const float r_skin = 1.0f;
  const int iters = 10;
  const int num_rbf = 32;

  std::cout << "=== Geometry benchmarks ===\n";

  bench_cell_list(N, box_size, r_cut, iters);
  bench_rbf(num_rbf, r_cut, 100000, iters);
  bench_graph_builder(N, box_size, r_cut, r_skin, num_rbf, iters);

  return 0;
}
