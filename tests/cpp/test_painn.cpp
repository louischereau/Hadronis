#include <gtest/gtest.h>

#include "painn/layout.hpp"

TEST(PaiNNLayoutTest, VectorFeatureIndexIsComponentMajor) {
  EXPECT_EQ(vector_feature_index(0, 0, 4), 0u);
  EXPECT_EQ(vector_feature_index(1, 0, 4), 4u);
  EXPECT_EQ(vector_feature_index(2, 3, 4), 11u);
  EXPECT_EQ(atom_vector_feature_index(1, 2, 3, 4), 23u);
}
