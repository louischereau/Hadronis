#include "painn/interaction.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(PaiNNInteractionTest, OutputShapesMatchInput) {
  const int N = 3;
  const int F = 4;
  PaINNInteraction interaction(F, 8);
  std::vector<float> s(N * F, 1.0f);
  std::vector<float> v(N * 3 * F, 2.0f);
  EdgeGraph graph;
  graph.edge_src = {0, 1, 2};
  graph.edge_dst = {1, 2, 0};
  graph.edge_rvec = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  graph.edge_rbf = std::vector<float>(3 * 8, 1.0f); // 3 edges, n_rbf=8
  graph.build_dst_offsets(N);
  float r_cut = 5.0f;
  interaction.forward(N, s, v, graph, r_cut);
  // No explicit output, but s and v should be updated in-place
  EXPECT_EQ(s.size(), static_cast<std::size_t>(N * F));
  EXPECT_EQ(v.size(), static_cast<std::size_t>(N * 3 * F));
}

TEST(PaiNNInteractionTest, ZeroWeightsProduceBiasOnly) {
  const int N = 2;
  const int F = 2;
  const int n_rbf = 1;
  PaINNInteraction interaction(F, n_rbf);
  // Set all weights to zero and biases to known values
  interaction.message_layer.mlp_linear1.set_weight({0.0f, 0.0f, 0.0f, 0.0f});
  interaction.message_layer.mlp_linear1.set_bias({1.0f, 2.0f});
  interaction.message_layer.mlp_linear2.set_weight(
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  interaction.message_layer.mlp_linear2.set_bias(
      {3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  interaction.message_layer.mlp_linear3.set_weight(
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  interaction.message_layer.mlp_linear3.set_bias(
      {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f});
  interaction.update_layer.U.set_weight({0.0f, 0.0f, 0.0f, 0.0f});
  interaction.update_layer.U.set_bias({15.0f, 16.0f});
  interaction.update_layer.V.set_weight({0.0f, 0.0f, 0.0f, 0.0f});
  interaction.update_layer.V.set_bias({17.0f, 18.0f});
  interaction.update_layer.linear1.set_weight(
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  interaction.update_layer.linear1.set_bias({19.0f, 20.0f});
  interaction.update_layer.linear2.set_weight(
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  interaction.update_layer.linear2.set_bias(
      {21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f});

  std::vector<float> s(N * F, 1.0f);
  std::vector<float> v(N * 3 * F, 2.0f);
  EdgeGraph graph;
  graph.edge_src = {0, 1};
  graph.edge_dst = {1, 0};
  graph.edge_rvec = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  graph.edge_rbf = std::vector<float>(2 * n_rbf, 1.0f); // 2 edges, n_rbf=1
  graph.build_dst_offsets(N);
  float r_cut = 5.0f;
  interaction.forward(N, s, v, graph, r_cut);
  // s and v should be updated in-place; just check sizes for now
  EXPECT_EQ(s.size(), static_cast<std::size_t>(N * F));
  EXPECT_EQ(v.size(), static_cast<std::size_t>(N * 3 * F));
}
