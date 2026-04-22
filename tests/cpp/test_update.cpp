#include "painn/update.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(PaiNNUpdateTest, ThrowsOnNegativeAtomCount) {
  PaiNNUpdate update(4);
  std::vector<float> s(4, 0.0f);
  std::vector<float> v(12, 0.0f);
  std::vector<float> ds, dv;
  EXPECT_THROW(update.forward(-1, s, v, ds, dv), std::runtime_error);
}

TEST(PaiNNUpdateTest, ThrowsOnScalarShapeMismatch) {
  PaiNNUpdate update(4);
  // n_atoms = 2, hidden_dim = 4, so s.size() should be 8
  std::vector<float> s(7, 0.0f);  // wrong size
  std::vector<float> v(24, 0.0f); // correct size for 2 atoms
  std::vector<float> ds, dv;
  EXPECT_THROW(update.forward(2, s, v, ds, dv), std::runtime_error);
}

TEST(PaiNNUpdateTest, ThrowsOnVectorShapeMismatch) {
  PaiNNUpdate update(4);
  // n_atoms = 2, hidden_dim = 4, so v.size() should be 24
  std::vector<float> s(8, 0.0f);  // correct size
  std::vector<float> v(23, 0.0f); // wrong size
  std::vector<float> ds, dv;
  EXPECT_THROW(update.forward(2, s, v, ds, dv), std::runtime_error);
}
#include "painn/update.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(PaiNNUpdateTest, LayerDimensionsMatchArchitecture) {
  PaiNNUpdate update(128);

  EXPECT_EQ(update.U.in_dim, 128);
  EXPECT_EQ(update.U.out_dim, 128);
  EXPECT_EQ(update.V.in_dim, 128);
  EXPECT_EQ(update.V.out_dim, 128);
  EXPECT_EQ(update.linear1.in_dim, 256);
  EXPECT_EQ(update.linear1.out_dim, 128);
  EXPECT_EQ(update.linear2.in_dim, 128);
  EXPECT_EQ(update.linear2.out_dim, 384);
}

TEST(PaiNNUpdateTest, ForwardOutputSizesMatchAtomCount) {
  const int F = 4;
  const int N = 3;
  PaiNNUpdate update(F);

  const std::vector<float> s(N * F, 0.0f);
  const std::vector<float> v(N * 3 * F, 0.0f);
  std::vector<float> ds_atom, dv_atom;

  update.forward(N, s, v, ds_atom, dv_atom);

  EXPECT_EQ(ds_atom.size(), static_cast<std::size_t>(N * F));
  EXPECT_EQ(dv_atom.size(), static_cast<std::size_t>(N * 3 * F));
}

TEST(PaiNNUpdateTest, ZeroWeightsProduceBiasOnlyScalarDelta) {
  const int F = 2;
  const int N = 2;
  PaiNNUpdate update(F);

  const float bias_val = 1.5f;
  update.linear2.set_bias({0.0f, 0.0f, 0.0f, 0.0f, bias_val, bias_val});

  const std::vector<float> s(N * F, 1.0f);
  const std::vector<float> v(N * 3 * F, 1.0f);
  std::vector<float> ds_atom, dv_atom;

  update.forward(N, s, v, ds_atom, dv_atom);

  for (int atom = 0; atom < N; ++atom)
    for (int f = 0; f < F; ++f)
      EXPECT_NEAR(ds_atom[atom * F + f], bias_val, 1e-5f);
}

TEST(PaiNNUpdateTest, VectorDeltaScalesWithVectorInput) {
  const int F = 2;
  const int N = 1;
  PaiNNUpdate update(F);

  update.U.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
  update.U.set_bias({0.0f, 0.0f});
  update.V.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
  update.V.set_bias({0.0f, 0.0f});
  const float k = 2.0f;
  update.linear2.set_bias({k, k, 0.0f, 0.0f, 0.0f, 0.0f});

  const std::vector<float> s(N * F, 0.0f);
  const std::vector<float> v1(N * 3 * F, 1.0f);
  std::vector<float> ds1, dv1;
  update.forward(N, s, v1, ds1, dv1);

  const std::vector<float> v2(N * 3 * F, 2.0f);
  std::vector<float> ds2, dv2;
  update.forward(N, s, v2, ds2, dv2);

  ASSERT_EQ(dv1.size(), dv2.size());
  for (std::size_t i = 0; i < dv1.size(); ++i)
    EXPECT_NEAR(dv2[i], 2.0f * dv1[i], 1e-5f);
}
