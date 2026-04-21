#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

// File-scope portability macro for compiler-specific restrict qualifiers.
#if defined(_MSC_VER)
#define HADRONIS_RESTRICT __restrict
#define HADRONIS_ASSUME(cond) __assume(cond)
#elif defined(__clang__)
#define HADRONIS_RESTRICT __restrict__
#define HADRONIS_ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUG__)
#define HADRONIS_RESTRICT __restrict__
#define HADRONIS_ASSUME(cond)                                                  \
  do {                                                                         \
    if (!(cond))                                                               \
      __builtin_unreachable();                                                 \
  } while (0)
#else
#define HADRONIS_RESTRICT
#define HADRONIS_ASSUME(cond) ((void)0)
#endif

// Free helper for the hot linear kernel so the compiler only sees raw pointers.
inline void linear_layer_compute(const float *HADRONIS_RESTRICT input,
                                 const float *HADRONIS_RESTRICT weight_ptr,
                                 const float *HADRONIS_RESTRICT bias_ptr,
                                 float *HADRONIS_RESTRICT out_ptr, int n_rows,
                                 int in_dim, int out_dim) {
  if (n_rows == 0 || in_dim == 0 || out_dim == 0) {
    return;
  }

  HADRONIS_ASSUME(in_dim > 0);
  HADRONIS_ASSUME(out_dim > 0);
  HADRONIS_ASSUME(n_rows > 0);

  for (int row = 0; row < n_rows; ++row) {
    const float *HADRONIS_RESTRICT in_row = input + row * in_dim;
    float *HADRONIS_RESTRICT out_row = out_ptr + row * out_dim;

    for (int out_idx = 0; out_idx < out_dim; ++out_idx) {
      float acc = bias_ptr[out_idx];
      const float *HADRONIS_RESTRICT w_row = weight_ptr + out_idx * in_dim;

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUG__)
#pragma GCC ivdep
#endif
      for (int in_idx = 0; in_idx < in_dim; ++in_idx) {
        acc += w_row[in_idx] * in_row[in_idx];
      }

      out_row[out_idx] = acc;
    }
  }
}

inline void
linear_layer_compute_concat(const float *HADRONIS_RESTRICT left, int left_dim,
                            const float *HADRONIS_RESTRICT right, int right_dim,
                            const float *HADRONIS_RESTRICT weight_ptr,
                            const float *HADRONIS_RESTRICT bias_ptr,
                            float *HADRONIS_RESTRICT out_ptr, int out_dim) {
  const int in_dim = left_dim + right_dim;
  if (in_dim == 0 || out_dim == 0) {
    return;
  }

  HADRONIS_ASSUME(in_dim > 0);
  HADRONIS_ASSUME(out_dim > 0);

  for (int out_idx = 0; out_idx < out_dim; ++out_idx) {
    float acc = bias_ptr[out_idx];
    const float *HADRONIS_RESTRICT w_row = weight_ptr + out_idx * in_dim;

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUG__)
#pragma GCC ivdep
#endif
    for (int in_idx = 0; in_idx < left_dim; ++in_idx) {
      acc += w_row[in_idx] * left[in_idx];
    }

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUG__)
#pragma GCC ivdep
#endif
    for (int in_idx = 0; in_idx < right_dim; ++in_idx) {
      acc += w_row[left_dim + in_idx] * right[in_idx];
    }

    out_ptr[out_idx] = acc;
  }
}

// Linear layer parameters and forward pass over flat contiguous tensors.
// Weight layout is row-major: [out_dim, in_dim].
struct LinearLayer {
  int in_dim;
  int out_dim;
  std::vector<float> weight; // [out_dim * in_dim]
  std::vector<float> bias;   // [out_dim]

  LinearLayer() : in_dim(0), out_dim(0) {}

  LinearLayer(int in_dim, int out_dim)
      : in_dim(in_dim), out_dim(out_dim),
        weight(static_cast<std::size_t>(in_dim) *
                   static_cast<std::size_t>(out_dim),
               0.0f),
        bias(static_cast<std::size_t>(out_dim), 0.0f) {}

  void set_weight(const std::vector<float> &w) {
    const std::size_t expected =
        static_cast<std::size_t>(in_dim) * static_cast<std::size_t>(out_dim);
    if (w.size() != expected) {
      throw std::runtime_error("LinearLayer::set_weight shape mismatch");
    }
    weight = w;
  }

  void set_bias(const std::vector<float> &b) {
    const std::size_t expected = static_cast<std::size_t>(out_dim);
    if (b.size() != expected) {
      throw std::runtime_error("LinearLayer::set_bias shape mismatch");
    }
    bias = b;
  }

  void forward(std::span<const float> input, std::vector<float> &output) const {
    if (input.size() != static_cast<std::size_t>(in_dim)) {
      throw std::runtime_error("LinearLayer::forward input shape mismatch");
    }

    forward(input.data(), 1, output);
  }

  std::vector<float> forward(std::span<const float> input) const {
    std::vector<float> output;
    forward(input, output);
    return output;
  }

  std::vector<float> forward(const std::vector<float> &input) const {
    return forward(std::span<const float>(input.data(), input.size()));
  }

  void forward(std::span<const float> input, int n_rows,
               std::vector<float> &output) const {
    if (n_rows < 0) {
      throw std::runtime_error(
          "LinearLayer::forward received negative batch size");
    }

    const std::size_t expected =
        static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(in_dim);
    if (input.size() != expected) {
      throw std::runtime_error(
          "LinearLayer::forward batched input shape mismatch");
    }

    forward(input.data(), n_rows, output);
  }

  void forward(const std::vector<float> &input, int n_rows,
               std::vector<float> &output) const {
    forward(std::span<const float>(input.data(), input.size()), n_rows, output);
  }

  void forward(std::span<const float> left, std::span<const float> right,
               std::vector<float> &output) const {
    const std::size_t combined = left.size() + right.size();
    if (combined != static_cast<std::size_t>(in_dim)) {
      throw std::runtime_error(
          "LinearLayer::forward concatenated input shape mismatch");
    }

    forward(left.data(), static_cast<int>(left.size()), right.data(),
            static_cast<int>(right.size()), output);
  }

  void forward(const float *HADRONIS_RESTRICT left,
               const float *HADRONIS_RESTRICT right,
               std::vector<float> &output) const {
    if (in_dim % 2 != 0) {
      throw std::runtime_error(
          "LinearLayer::forward half-split requires even input dimension");
    }

    const int half_dim = in_dim / 2;
    forward(left, half_dim, right, half_dim, output);
  }

  void forward(const float *HADRONIS_RESTRICT left, int left_dim,
               const float *HADRONIS_RESTRICT right, int right_dim,
               std::vector<float> &output) const {
    if (left_dim < 0 || right_dim < 0) {
      throw std::runtime_error(
          "LinearLayer::forward received negative split dimension");
    }
    if (left_dim + right_dim != in_dim) {
      throw std::runtime_error(
          "LinearLayer::forward concatenated input shape mismatch");
    }
    if ((left_dim > 0 && left == nullptr) ||
        (right_dim > 0 && right == nullptr)) {
      throw std::runtime_error(
          "LinearLayer::forward received null concatenated input");
    }

    const std::size_t required = static_cast<std::size_t>(out_dim);
    if (output.size() != required) {
      output.resize(required);
    }

    linear_layer_compute_concat(left, left_dim, right, right_dim, weight.data(),
                                bias.data(), output.data(), out_dim);
  }

  void forward(const float *HADRONIS_RESTRICT input, int n_rows,
               std::vector<float> &output) const {
    if (n_rows < 0) {
      throw std::runtime_error(
          "LinearLayer::forward received negative batch size");
    }
    if (n_rows > 0 && input == nullptr) {
      throw std::runtime_error("LinearLayer::forward received null input");
    }

    const std::size_t required =
        static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(out_dim);
    if (output.size() != required) {
      output.resize(required);
    }

    linear_layer_compute(input, weight.data(), bias.data(), output.data(),
                         n_rows, in_dim, out_dim);
  }
};
