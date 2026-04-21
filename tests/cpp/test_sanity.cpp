#include <gtest/gtest.h>

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
