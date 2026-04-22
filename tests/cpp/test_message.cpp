#include "models/edgeGraph.hpp"
#include "painn/message.hpp"
#include <gtest/gtest.h>
#include <vector>

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
  const std::vector<float> s(2 * 2, 0.0f);
  const std::vector<float> v(2 * 6, 1.0f);

  PaiNNMessage message(2, 1);
  message.mlp_linear1.set_weight(std::vector<float>(4, 0.0f));
  message.mlp_linear1.set_bias({1.0f, 1.0f});
  message.mlp_linear2.set_weight(std::vector<float>(12, 0.0f));
  message.mlp_linear2.set_bias(std::vector<float>(6, 1.0f));
  message.mlp_linear3.set_weight(std::vector<float>(6, 0.0f));
  message.mlp_linear3.set_bias(std::vector<float>(6, 1.0f));

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
