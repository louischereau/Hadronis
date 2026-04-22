#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

inline float sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  }

  const float z = expf(x);
  return z / (1.0f + z);
}

inline float silu(float x) { return x * sigmoid(x); }

template <typename Fn>
inline void apply_inplace(float *data, std::size_t size, Fn fn) {
  for (std::size_t i = 0; i < size; ++i)
    data[i] = fn(data[i]);
}

inline void sigmoid_inplace(float *data, std::size_t size) {
  apply_inplace(data, size, [](float x) { return sigmoid(x); });
}

inline void silu_inplace(float *data, std::size_t size) {
  apply_inplace(data, size, [](float x) { return silu(x); });
}

inline void sigmoid_inplace(std::vector<float> &tensor) {
  sigmoid_inplace(tensor.data(), tensor.size());
}

inline void silu_inplace(std::vector<float> &tensor) {
  silu_inplace(tensor.data(), tensor.size());
}
