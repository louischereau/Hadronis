#include "../../src/neighbors/edge_graph.hpp"
#include <gtest/gtest.h>
#include <vector>

// dst_offsets[i]..dst_offsets[i+1] must span exactly the edges whose dst==i.
TEST(EdgeGraphTest, DstOffsetsPartitionEdgesByDestination) {
  EdgeGraph g;
  g.edge_src = {0, 1, 0, 2};
  g.edge_dst = {1, 0, 2, 0}; // dst=0 has 2 edges, dst=1 has 1, dst=2 has 1
  g.edge_rvec = std::vector<float>(4 * 3, 0.0f);
  g.edge_rbf = std::vector<float>(4 * 2, 0.0f);
  g.sort_by_dst();
  g.build_dst_offsets(3);

  ASSERT_EQ(g.dst_offsets.size(), 4u); // n_atoms+1
  EXPECT_EQ(g.dst_offsets[0], 0u);
  EXPECT_EQ(g.dst_offsets[1], 2u); // 2 edges into atom 0
  EXPECT_EQ(g.dst_offsets[2], 3u); // 1 edge  into atom 1
  EXPECT_EQ(g.dst_offsets[3], 4u); // 1 edge  into atom 2

  // Verify every edge in each range really targets that atom
  for (int atom = 0; atom < 3; ++atom) {
    for (std::size_t e = g.dst_offsets[atom]; e < g.dst_offsets[atom + 1]; ++e)
      EXPECT_EQ(g.edge_dst[e], atom);
  }
}

// sort_by_dst must permute edge_src, edge_dst, edge_rvec, and edge_rbf
// consistently so that all per-edge arrays stay aligned.
TEST(EdgeGraphTest, SortByDstPermutesAllArraysConsistently) {
  EdgeGraph g;
  // Two edges in reverse dst order: (src=1,dst=1) then (src=0,dst=0)
  g.edge_src = {1, 0};
  g.edge_dst = {1, 0};
  g.edge_rvec = {9.0f, 8.0f, 7.0f,  // rvec for edge 0
                 1.0f, 2.0f, 3.0f}; // rvec for edge 1
  g.edge_rbf = {0.9f, 0.8f,         // rbf for edge 0
                0.1f, 0.2f};        // rbf for edge 1
  g.sort_by_dst();

  // After sort, edge with dst=0 must come first
  ASSERT_EQ(g.edge_dst.size(), 2u);
  EXPECT_EQ(g.edge_dst[0], 0);
  EXPECT_EQ(g.edge_src[0], 0);
  EXPECT_NEAR(g.edge_rvec[0], 1.0f,
              1e-6f); // rvec that belonged to (src=0,dst=0)
  EXPECT_NEAR(g.edge_rbf[0], 0.1f,
              1e-6f); // rbf  that belonged to (src=0,dst=0)

  EXPECT_EQ(g.edge_dst[1], 1);
  EXPECT_EQ(g.edge_src[1], 1);
  EXPECT_NEAR(g.edge_rvec[3], 9.0f,
              1e-6f); // x-component of rvec for (src=1,dst=1)
  EXPECT_NEAR(g.edge_rvec[5], 7.0f,
              1e-6f); // z-component of rvec for (src=1,dst=1)
  EXPECT_NEAR(g.edge_rbf[2], 0.9f, 1e-6f); // first rbf  for (src=1,dst=1)
}

// incoming_rvec and incoming_rbf must return spans of the correct size.
TEST(EdgeGraphTest, IncomingSpanSizesAreCorrect) {
  EdgeGraph g;
  g.edge_src = {0, 0};
  g.edge_dst = {1, 1};
  g.edge_rvec = std::vector<float>(2 * 3, 1.0f);
  g.edge_rbf = std::vector<float>(2 * 4, 1.0f);
  g.build_dst_offsets(2);

  EXPECT_EQ(g.incoming_rvec(0).size(), 0u);      // atom 0: no incoming edges
  EXPECT_EQ(g.incoming_rvec(1).size(), 2u * 3u); // atom 1: 2 edges × 3 floats
  EXPECT_EQ(g.incoming_rbf(1, 4).size(), 2u * 4u);
}

TEST(EdgeGraphTest, BuildDstOffsetsCorrectness) {
  EdgeGraph graph;
  graph.edge_src = {0, 0, 1, 2, 2, 2};
  graph.edge_dst = {1, 2, 2, 0, 1, 2};
  graph.sort_by_dst();
  graph.build_dst_offsets(3);
  // For 3 atoms, expect dst_offsets.size() == 4
  ASSERT_EQ(graph.dst_offsets.size(), 4u);
  // dst_offsets[atom] gives the start index in edge_dst for that atom
  EXPECT_EQ(graph.dst_offsets[0], 0u);
  EXPECT_EQ(graph.dst_offsets[1], 1u);
  EXPECT_EQ(graph.dst_offsets[2], 3u);
  EXPECT_EQ(graph.dst_offsets[3], 6u);
}

TEST(EdgeGraphTest, EdgeCountMatchesInput) {
  EdgeGraph graph;
  graph.edge_src = {0, 1, 2};
  graph.edge_dst = {1, 2, 0};
  graph.build_dst_offsets(3);
  EXPECT_EQ(graph.edge_src.size(), 3u);
  EXPECT_EQ(graph.edge_dst.size(), 3u);
}

TEST(EdgeGraphTest, HandlesEmptyGraph) {
  EdgeGraph graph;
  graph.build_dst_offsets(2);
  EXPECT_EQ(graph.dst_offsets.size(), 3u);
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(graph.dst_offsets[i], 0u);
}
