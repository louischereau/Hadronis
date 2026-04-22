#include <gtest/gtest.h>

#include "graphBuilder.hpp"
#include "models/vec3.hpp"

#include <algorithm>
#include <vector>

// Check that GraphBuilder builds the expected edges and RBF features
TEST(GraphBuilderTest, BuildsExpectedGraph) {
  const int N = 3;
  const float box_size = 10.0f;
  const float r_cut = 3.0f;
  const float r_skin = 0.0f;
  const int num_rbf = 4;

  std::vector<Vec3> pos = {
      Vec3{0.5f, 0.5f, 0.5f}, // atom 0
      Vec3{2.5f, 0.5f, 0.5f}, // atom 1 (within r_cut of 0)
      Vec3{9.5f, 9.5f, 9.5f}  // atom 2 (within r_cut of 0 via PBC)
  };

  GraphBuilder gb(N, box_size, r_cut, r_skin, num_rbf);
  gb.build(pos);

  // Expect four directed edges: (0,1), (1,0), (0,2) and (2,0).
  // GraphBuilder treats the neighbour graph as directed, emitting an edge
  // for each (i,j) within the cutoff when iterating over central atoms i.
  ASSERT_EQ(gb.edge_graph.num_edges(), 4u);

  std::vector<std::pair<int, int>> edges;
  for (std::size_t k = 0; k < gb.edge_graph.edge_src.size(); ++k) {
    edges.emplace_back(gb.edge_graph.edge_src[k], gb.edge_graph.edge_dst[k]);
  }

  std::sort(edges.begin(), edges.end());
  ASSERT_EQ(edges.size(), 4u);
  EXPECT_EQ(edges[0].first, 0);
  EXPECT_EQ(edges[0].second, 1);
  EXPECT_EQ(edges[1].first, 0);
  EXPECT_EQ(edges[1].second, 2);
  EXPECT_EQ(edges[2].first, 1);
  EXPECT_EQ(edges[2].second, 0);
  EXPECT_EQ(edges[3].first, 2);
  EXPECT_EQ(edges[3].second, 0);

  // RBF features should have num_rbf entries per edge
  EXPECT_EQ(gb.edge_graph.edge_rbf.size(),
            edges.size() * static_cast<std::size_t>(num_rbf));
}
