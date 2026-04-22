#include <gtest/gtest.h>

#include "../../src/neighbors/rbf.hpp"
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
