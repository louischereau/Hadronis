#include <gtest/gtest.h>

#include "models/cellList.hpp"
#include <vector>

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
