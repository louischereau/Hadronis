// Standalone benchmark for PaINN forward pass only
#include "../../src/neighbors/graph_builder.hpp"
#include "models/vec3.hpp"
#include "painn/painn.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static constexpr int kHiddenDim = 128;
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

static std::vector<int> make_atomic_numbers(int N) {
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> dist(1, 17);
  std::vector<int> z(static_cast<std::size_t>(N));
  for (auto &zi : z)
    zi = dist(rng);
  return z;
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

static void BM_PaiNNForward(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto pos = make_positions(N);
  auto z = make_atomic_numbers(N);
  const float box = compute_box_size(pos);
  GraphBuilder gb(N, box, kRCut, 0.0f, kNumRbf);
  gb.build(pos);
  PaINN painn(kHiddenDim, 3, kNumRbf);
  painn.predict(z.data(), N, gb.edge_graph, kRCut); // warmup
  for (auto _ : state) {
    float result = painn.predict(z.data(), N, gb.edge_graph, kRCut);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_Embedding(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto z = make_atomic_numbers(N);
  AtomicEmbedding embedding(100, kHiddenDim);
  std::vector<float> s(static_cast<std::size_t>(N) * kHiddenDim, 0.0f);
  for (auto _ : state) {
    embedding.embed_all(z.data(), N, s.data());
    benchmark::DoNotOptimize(s.data());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_MessageBlock(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto pos = make_positions(N);
  auto z = make_atomic_numbers(N);
  const float box = compute_box_size(pos);
  GraphBuilder gb(N, box, kRCut, 0.0f, kNumRbf);
  gb.build(pos);
  std::vector<float> s(static_cast<std::size_t>(N) * kHiddenDim, 0.0f);
  std::vector<float> v(static_cast<std::size_t>(N) * kHiddenDim * 3u, 0.0f);
  AtomicEmbedding embedding(100, kHiddenDim);
  embedding.embed_all(z.data(), N, s.data());
  PaiNNMessage message(kHiddenDim, kNumRbf);
  std::vector<float> ds, dv;
  for (auto _ : state) {
    message.forward(N, s, v, gb.edge_graph, kRCut, ds, dv);
    benchmark::DoNotOptimize(ds.data());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_UpdateBlock(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  std::vector<float> s(static_cast<std::size_t>(N) * kHiddenDim, 0.0f);
  std::vector<float> v(static_cast<std::size_t>(N) * kHiddenDim * 3u, 0.0f);
  PaiNNUpdate update(kHiddenDim);
  std::vector<float> ds, dv;
  for (auto _ : state) {
    update.forward(N, s, v, ds, dv);
    benchmark::DoNotOptimize(ds.data());
  }
  state.SetItemsProcessed(state.iterations() * N);
}

static void BM_PaiNNForward_EndToEnd(benchmark::State &state) {
  const int N = static_cast<int>(state.range(0));
  auto pos = make_positions(N);
  auto z = make_atomic_numbers(N);
  const float box = compute_box_size(pos);
  GraphBuilder gb(N, box, kRCut, 0.0f, kNumRbf);
  gb.build(pos);
  PaINN painn(kHiddenDim, 3, kNumRbf);
  for (auto _ : state) {
    float result = painn.predict(z.data(), N, gb.edge_graph, kRCut);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations() * N);
}

BENCHMARK(BM_PaiNNForward)->Arg(128)->Arg(256);
BENCHMARK(BM_Embedding)->Arg(128)->Arg(256);
BENCHMARK(BM_MessageBlock)->Arg(128)->Arg(256);
BENCHMARK(BM_UpdateBlock)->Arg(128)->Arg(256);
BENCHMARK(BM_PaiNNForward_EndToEnd)->Arg(128)->Arg(256);

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
