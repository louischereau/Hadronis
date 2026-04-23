
#pragma once

#include <Eigen/Dense>
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

// Eigen-based batched linear layer compute
inline void linear_layer_compute(const float *input, const float *weight_ptr,
                                 const float *bias_ptr, float *out_ptr,
                                 int n_rows, int in_dim, int out_dim) {
  if (n_rows == 0 || in_dim == 0 || out_dim == 0) {
    return;
  }
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      in_mat(input, n_rows, in_dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      w_mat(weight_ptr, out_dim, in_dim);
  Eigen::Map<const Eigen::VectorXf> b_vec(bias_ptr, out_dim);
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      out_mat(out_ptr, n_rows, out_dim);
  out_mat = in_mat * w_mat.transpose();
  out_mat.rowwise() += b_vec.transpose();
}

inline void linear_layer_compute_concat(const float *left, int left_dim,
                                        const float *right, int right_dim,
                                        const float *weight_ptr,
                                        const float *bias_ptr, float *out_ptr,
                                        int out_dim) {
  const int in_dim = left_dim + right_dim;
  if (in_dim == 0 || out_dim == 0) {
    return;
  }
  Eigen::VectorXf concat(in_dim);
  if (left_dim > 0) {
    Eigen::Map<const Eigen::VectorXf> lvec(left, left_dim);
    concat.head(left_dim) = lvec;
  }
  if (right_dim > 0) {
    Eigen::Map<const Eigen::VectorXf> rvec(right, right_dim);
    concat.tail(right_dim) = rvec;
  }
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      w_mat(weight_ptr, out_dim, in_dim);
  Eigen::Map<const Eigen::VectorXf> b_vec(bias_ptr, out_dim);
  Eigen::Map<Eigen::VectorXf> out_vec(out_ptr, out_dim);
  out_vec = w_mat * concat + b_vec;
}

// Linear layer parameters and forward pass over flat contiguous tensors.
// Weight layout is row-major: [out_dim, in_dim].
struct LinearLayer {
  // ...existing code...

  // Restore single-row forward overloads for compatibility with existing code
  void forward(std::span<const float> input, std::vector<float> &output) const {
    if (input.size() != static_cast<std::size_t>(in_dim)) {
      throw std::runtime_error("LinearLayer::forward input shape mismatch");
    }
    forward(input, 1, output);
  }

  void forward(const std::vector<float> &input,
               std::vector<float> &output) const {
    if (input.size() != static_cast<std::size_t>(in_dim)) {
      throw std::runtime_error("LinearLayer::forward input shape mismatch");
    }
    forward(std::span<const float>(input.data(), input.size()), 1, output);
  }
  int in_dim;
  int out_dim;
  std::vector<float> weight; // [out_dim * in_dim]
  std::vector<float> bias;   // [out_dim]

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

  // Removed single-row forward pass and related overloads (not used in repo)

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
    if (output.size() !=
        static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(out_dim)) {
      output.resize(static_cast<std::size_t>(n_rows) *
                    static_cast<std::size_t>(out_dim));
    }
    linear_layer_compute(input.data(), weight.data(), bias.data(),
                         output.data(), n_rows, in_dim, out_dim);
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
    if (output.size() != static_cast<std::size_t>(out_dim)) {
      output.resize(static_cast<std::size_t>(out_dim));
    }
    linear_layer_compute_concat(left.data(), static_cast<int>(left.size()),
                                right.data(), static_cast<int>(right.size()),
                                weight.data(), bias.data(), output.data(),
                                out_dim);
  }

  // Removed pointer-based and half-split forward overloads (not used in repo)
};
