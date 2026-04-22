#include "painn/silu.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

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
