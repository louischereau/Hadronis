#include <gtest/gtest.h>

#include "graphBuilder.hpp"
#include "models/cellList.hpp"
#include "models/edgeGraph.hpp"
#include "models/rbf.hpp"
#include "models/vec3.hpp"

#include <algorithm>
#include <vector>

// Basic checks for RadialBasis construction and expansion
TEST(RadialBasisTest, CentersAndWidths) {
  const int num_rbf = 3;
  const float cutoff = 3.0f;

  RadialBasis rb(num_rbf, cutoff);

  ASSERT_EQ(static_cast<int>(rb.centers.size()), num_rbf);

  // For num_rbf = 3 and cutoff = 3, centers should be [0, 1.5, 3]
  EXPECT_NEAR(rb.centers[0], 0.0f, 1e-5f);
  EXPECT_NEAR(rb.centers[1], 1.5f, 1e-5f);
  EXPECT_NEAR(rb.centers[2], 3.0f, 1e-5f);

  // Expanding at the middle center should give maximum response there
  std::vector<float> out;
  rb.expand_append(1.5f, out);

  ASSERT_EQ(out.size(), rb.centers.size());
  EXPECT_GT(out[1], out[0]);
  EXPECT_GT(out[1], out[2]);
  EXPECT_NEAR(out[1], 1.0f, 1e-4f);
}

// Check that CellList assigns particles to expected cells
TEST(CellListTest, AssignsParticlesToCells) {
  const int N = 3;
  const float box_size = 10.0f;
  const float r_cut = 2.0f;

  CellList cl(N, box_size, r_cut);

  std::vector<Vec3> pos = {
      Vec3{0.5f, 0.5f, 0.5f}, // near origin
      Vec3{2.5f, 0.5f, 0.5f}, // shifted in x
      Vec3{9.5f, 9.5f, 9.5f}  // far corner
  };

  cl.build(pos);

  // Compute expected cell indices using the public API
  int c0 = cl.cell_of(pos[0]);
  int c1 = cl.cell_of(pos[1]);
  int c2 = cl.cell_of(pos[2]);

  // Particles 0 and 1 should be in different cells along x
  EXPECT_NE(c0, c1);
  // Particle 2 should also occupy some (possibly different) cell
  EXPECT_NE(c0, c2);
  EXPECT_NE(c1, c2);
}

// (VerletList tests removed: GraphBuilder now performs fused neighbor
// search + RBF expansion directly; VerletList is no longer part of the
// latency-critical path.)

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
