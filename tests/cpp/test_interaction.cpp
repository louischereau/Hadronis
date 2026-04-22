#include "painn/interaction.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(PaiNNInteractionTest, OutputShapesMatchInput) {
  const int N = 3;
  const int F = 4;
  PaINNInteraction interaction(F, 8);
  std::vector<float> s(N * F, 1.0f);
  std::vector<float> v(N * 3 * F, 2.0f);
  std::vector<float> ds_atom, dv_atom;
  interaction.forward(N, s, v, ds_atom, dv_atom);
  EXPECT_EQ(ds_atom.size(), static_cast<std::size_t>(N * F));
  EXPECT_EQ(dv_atom.size(), static_cast<std::size_t>(N * 3 * F));
}

TEST(PaiNNInteractionTest, ZeroWeightsProduceBiasOnly) {
  const int N = 2;
  const int F = 2;
  PaINNInteraction interaction(F, 1);
  interaction.linear1.set_weight({0.0f, 0.0f, 0.0f, 0.0f});
  interaction.linear1.set_bias({1.0f, 2.0f});
  interaction.linear2.set_weight({0.0f, 0.0f, 0.0f, 0.0f});
  interaction.linear2.set_bias({3.0f, 4.0f});
  std::vector<float> s(N * F, 1.0f);
  std::vector<float> v(N * 3 * F, 2.0f);
  std::vector<float> ds_atom, dv_atom;
  interaction.forward(N, s, v, ds_atom, dv_atom);
  for (int i = 0; i < N * F; ++i)
    EXPECT_NEAR(ds_atom[i], 1.0f, 1e-6f);
  for (int i = 0; i < N * F; ++i)
    EXPECT_NEAR(dv_atom[i], 3.0f, 1e-6f);
}
