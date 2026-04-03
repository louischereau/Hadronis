#include <gtest/gtest.h>

#include "graphBuilder.hpp"
#include "models/cellList.hpp"
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

  // Expect exactly two unique edges: (0,1) and (0,2), stored once each
  ASSERT_EQ(gb.edge_graph.num_edges(), 2u);

  std::vector<std::pair<int, int>> edges;
  for (std::size_t k = 0; k < gb.edge_graph.edge_src.size(); ++k) {
    edges.emplace_back(gb.edge_graph.edge_src[k], gb.edge_graph.edge_dst[k]);
  }

  std::sort(edges.begin(), edges.end());
  ASSERT_EQ(edges.size(), 2u);
  EXPECT_EQ(edges[0].first, 0);
  EXPECT_EQ(edges[0].second, 1);
  EXPECT_EQ(edges[1].first, 0);
  EXPECT_EQ(edges[1].second, 2);

  // RBF features should have num_rbf entries per edge
  EXPECT_EQ(gb.edge_graph.edge_rbf.size(),
            edges.size() * static_cast<std::size_t>(num_rbf));
}
