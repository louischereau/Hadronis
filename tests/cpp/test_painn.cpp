#include <gtest/gtest.h>

#include "models/edgeGraph.hpp"
#include "painn/layout.hpp"
#include "painn/linear_layer.hpp"
#include "painn/message.hpp"
#include "painn/update.hpp"
#include "utils/activations.hpp"

#include <cmath>
#include <vector>

TEST(SanityCheck, BasicArithmetic) { EXPECT_EQ(2, 1 + 1); }

TEST(ActivationTest, SiLUMatchesDefinition) {
  EXPECT_NEAR(silu(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(silu(2.0f), 2.0f / (1.0f + std::exp(-2.0f)), 1e-6f);
  EXPECT_NEAR(silu(-2.0f), -2.0f / (1.0f + std::exp(2.0f)), 1e-6f);
}

TEST(ActivationTest, TensorActivationsApplyElementwise) {
  std::vector<float> sigmoid_values = {-2.0f, 0.0f, 2.0f};
  sigmoid_inplace(sigmoid_values);

  EXPECT_NEAR(sigmoid_values[0], 1.0f / (1.0f + std::exp(2.0f)), 1e-6f);
  EXPECT_NEAR(sigmoid_values[1], 0.5f, 1e-6f);
  EXPECT_NEAR(sigmoid_values[2], 1.0f / (1.0f + std::exp(-2.0f)), 1e-6f);

  std::vector<float> silu_values = {-2.0f, 0.0f, 2.0f};
  silu_inplace(silu_values);

  EXPECT_NEAR(silu_values[0], -2.0f / (1.0f + std::exp(2.0f)), 1e-6f);
  EXPECT_NEAR(silu_values[1], 0.0f, 1e-6f);
  EXPECT_NEAR(silu_values[2], 2.0f / (1.0f + std::exp(-2.0f)), 1e-6f);
}

TEST(PaiNNLayoutTest, VectorFeatureIndexIsComponentMajor) {
  EXPECT_EQ(vector_feature_index(0, 0, 4), 0u);
  EXPECT_EQ(vector_feature_index(1, 0, 4), 4u);
  EXPECT_EQ(vector_feature_index(2, 3, 4), 11u);
  EXPECT_EQ(atom_vector_feature_index(1, 2, 3, 4), 23u);
}

TEST(LinearLayerTest, ConcatForwardMatchesFlatInput) {
  LinearLayer layer(4, 2);
  layer.set_weight({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -0.5f});
  layer.set_bias({0.25f, -1.25f});

  const std::vector<float> full = {1.0f, -2.0f, 0.5f, 3.0f};
  const std::vector<float> expected = layer.forward(full);

  std::vector<float> fused;
  layer.forward(full.data(), full.data() + 2, fused);

  ASSERT_EQ(fused.size(), expected.size());
  EXPECT_NEAR(fused[0], expected[0], 1e-6f);
  EXPECT_NEAR(fused[1], expected[1], 1e-6f);
}

TEST(LinearLayerTest, ForwardComputesExpectedOutputs) {
  LinearLayer layer(3, 2);
  layer.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  layer.set_bias({0.5f, -1.0f});

  const std::vector<float> single = {1.0f, 0.0f, -1.0f};
  const std::vector<float> single_out = layer.forward(single);

  ASSERT_EQ(single_out.size(), 2u);
  EXPECT_NEAR(single_out[0], -1.5f, 1e-6f);
  EXPECT_NEAR(single_out[1], -3.0f, 1e-6f);

  const std::vector<float> batch = {1.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f};
  std::vector<float> batch_out;
  layer.forward(batch, 2, batch_out);

  ASSERT_EQ(batch_out.size(), 4u);
  EXPECT_NEAR(batch_out[0], -1.5f, 1e-6f);
  EXPECT_NEAR(batch_out[1], -3.0f, 1e-6f);
  EXPECT_NEAR(batch_out[2], 3.5f, 1e-6f);
  EXPECT_NEAR(batch_out[3], 6.5f, 1e-6f);
}

// dim == 0 ---------------------------------------------------------------

TEST(LinearLayerTest, ZeroDimLayerForwardProducesEmptyOutput) {
  // out_dim=0: output must always be empty.
  LinearLayer layer_zero_out(3, 0);
  std::vector<float> out;
  const std::vector<float> input(3, 1.0f);
  layer_zero_out.forward(input, out);
  EXPECT_TRUE(out.empty());

  // in_dim=0: output size == out_dim (zeroed, no weights applied).
  LinearLayer layer_zero_in(0, 3);
  layer_zero_in.forward(std::span<const float>{}, out);
  EXPECT_EQ(out.size(), 3u);
}

TEST(LinearLayerTest, ZeroDimBatchForwardProducesEmptyOutput) {
  LinearLayer layer(2, 3);
  std::vector<float> out;
  layer.forward(std::span<const float>{}, 0, out);
  EXPECT_TRUE(out.empty());
}

// Runtime errors ---------------------------------------------------------

TEST(LinearLayerTest, SetWeightThrowsOnWrongSize) {
  LinearLayer layer(2, 3);
  EXPECT_THROW(layer.set_weight({1.0f, 2.0f}), std::runtime_error);
  EXPECT_THROW(layer.set_weight(std::vector<float>(7, 0.0f)),
               std::runtime_error);
}

TEST(LinearLayerTest, SetBiasThrowsOnWrongSize) {
  LinearLayer layer(2, 3);
  EXPECT_THROW(layer.set_bias({1.0f, 2.0f}), std::runtime_error);
  EXPECT_THROW(layer.set_bias(std::vector<float>(5, 0.0f)), std::runtime_error);
}

TEST(LinearLayerTest, ForwardThrowsOnInputSizeMismatch) {
  LinearLayer layer(3, 2);
  const std::vector<float> bad(2, 0.0f);
  EXPECT_THROW(layer.forward(bad), std::runtime_error);
}

TEST(LinearLayerTest, BatchedForwardThrowsOnNegativeBatchSize) {
  LinearLayer layer(2, 3);
  const std::vector<float> input(4, 0.0f);
  std::vector<float> out;
  EXPECT_THROW(layer.forward(std::span<const float>(input), -1, out),
               std::runtime_error);
}

TEST(LinearLayerTest, BatchedForwardThrowsOnSizeMismatch) {
  LinearLayer layer(2, 3);
  const std::vector<float> input(5, 0.0f);
  std::vector<float> out;
  EXPECT_THROW(layer.forward(std::span<const float>(input), 2, out),
               std::runtime_error);
}

TEST(LinearLayerTest, ConcatForwardThrowsOnSizeMismatch) {
  LinearLayer layer(4, 2);
  const std::vector<float> left(3, 0.0f);
  const std::vector<float> right(3, 0.0f);
  std::vector<float> out;
  EXPECT_THROW(layer.forward(std::span<const float>(left),
                             std::span<const float>(right), out),
               std::runtime_error);
}

TEST(LinearLayerTest, HalfSplitForwardThrowsOnOddInputDim) {
  LinearLayer layer(3, 2);
  const float a = 0.0f;
  std::vector<float> out;
  EXPECT_THROW(layer.forward(&a, &a, out), std::runtime_error);
}

TEST(PaiNNUpdateTest, LayerDimensionsMatchArchitecture) {
  PaiNNUpdate update(128);

  EXPECT_EQ(update.U.in_dim, 128);
  EXPECT_EQ(update.U.out_dim, 128);
  EXPECT_EQ(update.V.in_dim, 128);
  EXPECT_EQ(update.V.out_dim, 128);
  EXPECT_EQ(update.linear1.in_dim, 256);
  EXPECT_EQ(update.linear1.out_dim, 128);
  EXPECT_EQ(update.linear2.in_dim, 128);
  EXPECT_EQ(update.linear2.out_dim, 384);
}

TEST(PaiNNUpdateTest, ForwardOutputSizesMatchAtomCount) {
  const int F = 4;
  const int N = 3;
  PaiNNUpdate update(F);

  const std::vector<float> s(N * F, 0.0f);
  const std::vector<float> v(N * 3 * F, 0.0f);
  std::vector<float> ds_atom, dv_atom;

  update.forward(N, s, v, ds_atom, dv_atom);

  EXPECT_EQ(ds_atom.size(), static_cast<std::size_t>(N * F));
  EXPECT_EQ(dv_atom.size(), static_cast<std::size_t>(N * 3 * F));
}

// With all weights zero, gate a_ss = bias of linear2[2F..3F].
// ds[atom] = a_ss + a_sv * dot(Uv, Vv). With zero weights, U=0, V=0 so dot=0.
// Therefore ds[atom] = a_ss for every atom, independent of input.
TEST(PaiNNUpdateTest, ZeroWeightsProduceBiasOnlyScalarDelta) {
  const int F = 2;
  const int N = 2;
  PaiNNUpdate update(F);

  // All weights zero (default), set only linear2 bias for a_ss slot (indices
  // 2F..3F-1 in the 3F-wide output).
  const float bias_val = 1.5f;
  update.linear2.set_bias({0.0f, 0.0f,           // a_vv
                           0.0f, 0.0f,           // a_sv
                           bias_val, bias_val}); // a_ss

  const std::vector<float> s(N * F, 1.0f);
  const std::vector<float> v(N * 3 * F, 1.0f);
  std::vector<float> ds_atom, dv_atom;

  update.forward(N, s, v, ds_atom, dv_atom);

  for (int atom = 0; atom < N; ++atom)
    for (int f = 0; f < F; ++f)
      EXPECT_NEAR(ds_atom[atom * F + f], bias_val, 1e-5f);
}

// With identity U and V, zero linear1/linear2 weights, and a_vv bias = k,
// dv[atom] = k * Uv = k * v.  Doubling v must double dv.
TEST(PaiNNUpdateTest, VectorDeltaScalesWithVectorInput) {
  const int F = 2;
  const int N = 1;
  PaiNNUpdate update(F);

  // U = identity
  update.U.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
  update.U.set_bias({0.0f, 0.0f});
  // V = identity
  update.V.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
  update.V.set_bias({0.0f, 0.0f});
  // a_vv bias = 1, others zero
  const float k = 2.0f;
  update.linear2.set_bias({k, k, 0.0f, 0.0f, 0.0f, 0.0f});

  const std::vector<float> s(N * F, 0.0f);

  // v1 = all 1s
  const std::vector<float> v1(N * 3 * F, 1.0f);
  std::vector<float> ds1, dv1;
  update.forward(N, s, v1, ds1, dv1);

  // v2 = all 2s
  const std::vector<float> v2(N * 3 * F, 2.0f);
  std::vector<float> ds2, dv2;
  update.forward(N, s, v2, ds2, dv2);

  ASSERT_EQ(dv1.size(), dv2.size());
  for (std::size_t i = 0; i < dv1.size(); ++i)
    EXPECT_NEAR(dv2[i], 2.0f * dv1[i], 1e-5f);
}

TEST(PaiNNMessageTest, ForwardOutputShapesMatchHiddenDim) {
  EdgeGraph graph;
  graph.edge_src = {0};
  graph.edge_dst = {1};
  graph.edge_rvec = {1.0f, 0.0f, 0.0f};
  graph.edge_rbf = {1.0f, 0.5f, 0.25f, 0.125f};
  graph.build_dst_offsets(2);

  PaiNNMessage message(128, 4);

  std::vector<float> s(2 * 128, 1.0f);
  std::vector<float> v(2 * 3 * 128, 0.0f);
  std::vector<float> ds_atom, dv_atom;
  message.forward(2, s, v, graph, 5.0f, ds_atom, dv_atom);
  EXPECT_EQ(ds_atom.size(), 2u * 128u);
  EXPECT_EQ(dv_atom.size(), 2u * 3u * 128u);
}

TEST(PaiNNMessageTest, CutoffZerosMessagesAtRadius) {
  const std::vector<float> s(2 * 2, 0.0f);
  const std::vector<float> v(2 * 6, 1.0f);

  PaiNNMessage message(2, 1);
  message.mlp_linear1.set_weight(std::vector<float>(4, 0.0f));
  message.mlp_linear1.set_bias({1.0f, 1.0f});
  message.mlp_linear2.set_weight(std::vector<float>(12, 0.0f));
  message.mlp_linear2.set_bias(std::vector<float>(6, 1.0f));
  message.mlp_linear3.set_weight(std::vector<float>(6, 0.0f));
  message.mlp_linear3.set_bias(std::vector<float>(6, 1.0f));

  // Edge within cutoff: atom 0 -> atom 1 at r=1, r_cut=2
  {
    EdgeGraph graph;
    graph.edge_src = {0};
    graph.edge_dst = {1};
    graph.edge_rvec = {1.0f, 0.0f, 0.0f};
    graph.edge_rbf = {1.0f};
    graph.build_dst_offsets(2);

    std::vector<float> ds_atom, dv_atom;
    message.forward(2, s, v, graph, 2.0f, ds_atom, dv_atom);
    EXPECT_GT(ds_atom[2], 0.0f); // atom 1, feature 0
  }

  // Edge at cutoff boundary: r=2, r_cut=2 -> f_cut=0, no contribution
  {
    EdgeGraph graph;
    graph.edge_src = {0};
    graph.edge_dst = {1};
    graph.edge_rvec = {2.0f, 0.0f, 0.0f};
    graph.edge_rbf = {1.0f};
    graph.build_dst_offsets(2);

    std::vector<float> ds_atom, dv_atom;
    message.forward(2, s, v, graph, 2.0f, ds_atom, dv_atom);
    for (float val : ds_atom)
      EXPECT_NEAR(val, 0.0f, 1e-6f);
  }
}

TEST(PaiNNMessageTest, ForwardSumsContributions) {
  // N=2 atoms, F=2
  const std::vector<float> s(2 * 2, 0.0f);
  const std::vector<float> v(2 * 6, 1.0f);

  PaiNNMessage message(2, 1);
  message.mlp_linear1.set_weight(std::vector<float>(4, 0.0f));
  message.mlp_linear1.set_bias({1.0f, 1.0f});
  message.mlp_linear2.set_weight(std::vector<float>(12, 0.0f));
  message.mlp_linear2.set_bias(std::vector<float>(6, 1.0f));
  message.mlp_linear3.set_weight(std::vector<float>(6, 0.0f));
  message.mlp_linear3.set_bias(std::vector<float>(6, 1.0f));

  // Reference: single edge atom 0 -> atom 1
  std::vector<float> ds_ref, dv_ref;
  {
    EdgeGraph graph;
    graph.edge_src = {0};
    graph.edge_dst = {1};
    graph.edge_rvec = {1.0f, 0.0f, 0.0f};
    graph.edge_rbf = {1.0f};
    graph.build_dst_offsets(2);
    message.forward(2, s, v, graph, 3.0f, ds_ref, dv_ref);
  }

  // Two identical edges atom 0 -> atom 1: result must be 2x the single-edge
  std::vector<float> ds_atom, dv_atom;
  {
    EdgeGraph graph;
    graph.edge_src = {0, 0};
    graph.edge_dst = {1, 1};
    graph.edge_rvec = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    graph.edge_rbf = {1.0f, 1.0f};
    graph.build_dst_offsets(2);
    message.forward(2, s, v, graph, 3.0f, ds_atom, dv_atom);
  }

  ASSERT_EQ(ds_atom.size(), ds_ref.size());
  for (std::size_t i = 0; i < ds_atom.size(); ++i)
    EXPECT_NEAR(ds_atom[i], 2.0f * ds_ref[i], 1e-5f);
}
