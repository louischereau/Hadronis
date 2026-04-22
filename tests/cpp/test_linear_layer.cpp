#include "painn/linear_layer.hpp"
#include <gtest/gtest.h>
#include <span>
#include <vector>

TEST(LinearLayerTest, ConcatForwardMatchesFlatInput) {
  LinearLayer layer(4, 2);
  layer.set_weight({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -0.5f});
  layer.set_bias({0.25f, -1.25f});

  const std::vector<float> full = {1.0f, -2.0f, 0.5f, 3.0f};
  const std::vector<float> expected = layer.forward(full);

  std::vector<float> fused;
  layer.forward(full.data(), full.data() + 2, fused);

  ASSERT_EQ(fused.size(), expected.size());
  EXPECT_NEAR(fused[0], expected[0], 1e-6f);
  EXPECT_NEAR(fused[1], expected[1], 1e-6f);
}

TEST(LinearLayerTest, ForwardComputesExpectedOutputs) {
  LinearLayer layer(3, 2);
  layer.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  layer.set_bias({0.5f, -1.0f});

  const std::vector<float> single = {1.0f, 0.0f, -1.0f};
  const std::vector<float> single_out = layer.forward(single);

  ASSERT_EQ(single_out.size(), 2u);
  EXPECT_NEAR(single_out[0], -1.5f, 1e-6f);
  EXPECT_NEAR(single_out[1], -3.0f, 1e-6f);

  const std::vector<float> batch = {1.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f};
  std::vector<float> batch_out;
  layer.forward(batch, 2, batch_out);

  ASSERT_EQ(batch_out.size(), 4u);
  EXPECT_NEAR(batch_out[0], -1.5f, 1e-6f);
  EXPECT_NEAR(batch_out[1], -3.0f, 1e-6f);
  EXPECT_NEAR(batch_out[2], 3.5f, 1e-6f);
  EXPECT_NEAR(batch_out[3], 6.5f, 1e-6f);
}

TEST(LinearLayerTest, ZeroDimLayerForwardProducesEmptyOutput) {
  LinearLayer layer_zero_out(3, 0);
  std::vector<float> out;
  const std::vector<float> input(3, 1.0f);
  layer_zero_out.forward(input, out);
  EXPECT_TRUE(out.empty());

  LinearLayer layer_zero_in(0, 3);
  layer_zero_in.forward(std::span<const float>{}, out);
  EXPECT_EQ(out.size(), 3u);
}

TEST(LinearLayerTest, ZeroDimBatchForwardProducesEmptyOutput) {
  LinearLayer layer(2, 3);
  std::vector<float> out;
  layer.forward(std::span<const float>{}, 0, out);
  EXPECT_TRUE(out.empty());
}

TEST(LinearLayerTest, SetWeightThrowsOnWrongSize) {
  LinearLayer layer(2, 3);
  EXPECT_THROW(layer.set_weight({1.0f, 2.0f}), std::runtime_error);
  EXPECT_THROW(layer.set_weight(std::vector<float>(7, 0.0f)),
               std::runtime_error);
}

TEST(LinearLayerTest, SetBiasThrowsOnWrongSize) {
  LinearLayer layer(2, 3);
  EXPECT_THROW(layer.set_bias({1.0f, 2.0f}), std::runtime_error);
  EXPECT_THROW(layer.set_bias(std::vector<float>(5, 0.0f)), std::runtime_error);
}

TEST(LinearLayerTest, ForwardThrowsOnInputSizeMismatch) {
  LinearLayer layer(3, 2);
  const std::vector<float> bad(2, 0.0f);
  EXPECT_THROW(layer.forward(bad), std::runtime_error);
}

TEST(LinearLayerTest, BatchedForwardThrowsOnNegativeBatchSize) {
  LinearLayer layer(2, 3);
  const std::vector<float> input(4, 0.0f);
  std::vector<float> out;
  EXPECT_THROW(layer.forward(std::span<const float>(input), -1, out),
               std::runtime_error);
}

TEST(LinearLayerTest, BatchedForwardThrowsOnSizeMismatch) {
  LinearLayer layer(2, 3);
  const std::vector<float> input(5, 0.0f);
  std::vector<float> out;
  EXPECT_THROW(layer.forward(std::span<const float>(input), 2, out),
               std::runtime_error);
}

TEST(LinearLayerTest, ConcatForwardThrowsOnSizeMismatch) {
  LinearLayer layer(4, 2);
  const std::vector<float> left(3, 0.0f);
  const std::vector<float> right(3, 0.0f);
  std::vector<float> out;
  EXPECT_THROW(layer.forward(std::span<const float>(left),
                             std::span<const float>(right), out),
               std::runtime_error);
}

TEST(LinearLayerTest, HalfSplitForwardThrowsOnOddInputDim) {
  LinearLayer layer(3, 2);
  const float a = 0.0f;
  std::vector<float> out;
  EXPECT_THROW(layer.forward(&a, &a, out), std::runtime_error);
}
