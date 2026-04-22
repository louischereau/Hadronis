// Standalone benchmark for neighbor graph construction only
#include "../../src/neighbors/graph_builder.hpp"
#include "models/vec3.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static constexpr int kNumRbf = 20;
static constexpr float kRCut = 5.0f;
static constexpr float kSpacing = 3.5f;

static std::vector<Vec3> make_positions(int N) {
  std::mt19937 rng(42);
  const float side = std::cbrt(static_cast<float>(N)) * kSpacing;
  std::uniform_real_distribution<float> dist(0.0f, side);
  std::vector<Vec3> pos(static_cast<std::size_t>(N));
  for (auto &p : pos)
    p = {dist(rng), dist(rng), dist(rng)};
  return pos;
}

static float compute_box_size(const std::vector<Vec3> &pos) {
  float xmin = pos[0].x, xmax = pos[0].x;
  float ymin = pos[0].y, ymax = pos[0].y;
  float zmin = pos[0].z, zmax = pos[0].z;
  for (const auto &p : pos) {
    xmin = std::min(xmin, p.x);
    xmax = std::max(xmax, p.x);
    ymin = std::min(ymin, p.y);
    ymax = std::max(ymax, p.y);
    zmin = std::min(zmin, p.z);
    zmax = std::max(zmax, p.z);
  }
  const float extent = std::max({xmax - xmin, ymax - ymin, zmax - zmin});
  return std::max(2.0f * extent + 0.01f, kRCut + 0.01f);
}

static void BM_GraphBuild(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto pos = make_positions(N);
  const float box = compute_box_size(pos);
  GraphBuilder gb(N, box, kRCut, 0.0f, kNumRbf);
  gb.build(pos); // warmup
  for (auto _ : state) {
    gb.build(pos);
    benchmark::DoNotOptimize(gb.edge_graph.num_edges());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_CellList(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto pos = make_positions(N);
  const float box = compute_box_size(pos);
  for (auto _ : state) {
    CellList cl(N, box, kRCut);
    cl.build(pos);
    benchmark::DoNotOptimize(cl.head.data());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_RBFExpansion(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  RadialBasis rbf(kNumRbf, kRCut);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, kRCut);
  std::vector<float> ds(N);
  for (int i = 0; i < N; ++i)
    ds[i] = dist(rng);
  std::vector<float> out;
  for (auto _ : state) {
    out.clear();
    for (int i = 0; i < N; ++i)
      rbf.expand_append(ds[i], out);
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_EdgeGraph(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto pos = make_positions(N);
  const float box = compute_box_size(pos);
  GraphBuilder gb(N, box, kRCut, 0.0f, kNumRbf);
  gb.build(pos);
  for (auto _ : state) {
    gb.edge_graph.build_dst_offsets(N);
    benchmark::DoNotOptimize(gb.edge_graph.dst_offsets.data());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_GraphBuild_EndToEnd(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto pos = make_positions(N);
  const float box = compute_box_size(pos);
  for (auto _ : state) {
    GraphBuilder gb(N, box, kRCut, 0.0f, kNumRbf);
    gb.build(pos);
    benchmark::DoNotOptimize(gb.edge_graph.num_edges());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

BENCHMARK(BM_GraphBuild)->Arg(128)->Arg(256);
BENCHMARK(BM_CellList)->Arg(128)->Arg(256);
BENCHMARK(BM_RBFExpansion)->Arg(128)->Arg(256);
BENCHMARK(BM_EdgeGraph)->Arg(128)->Arg(256);
BENCHMARK(BM_GraphBuild_EndToEnd)->Arg(128)->Arg(256);

#include <string>
int main(int argc, char **argv) {
  const std::string kRepsFlag = "--benchmark_repetitions=";
  bool has_reps = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind(kRepsFlag, 0) == 0) {
      has_reps = true;
      break;
    }
  std::string default_reps = kRepsFlag + "5";
  std::vector<char *> new_argv(argv, argv + argc);
  if (!has_reps)
    new_argv.push_back(default_reps.data());
  int new_argc = static_cast<int>(new_argv.size());
  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data()))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
