#pragma once

#include <cmath>
#include <vector>

struct RadialBasis {
  // Default parameters for the radial basis construction.
  static constexpr float kDefaultWidth = 1.0f;
  static constexpr float kInvWidth2Eps = 1e-8f;

  std::vector<float> centers;
  float width;      // same for all centers
  float inv_width2; // precomputed 1 / (width^2 + eps)

  RadialBasis() : centers(), width(kDefaultWidth), inv_width2(1.0f) {}

  // Construct a set of `num_rbf` Gaussian centers linearly spaced between
  // 0 and `cutoff` (inclusive). All basis functions share the same width,
  // derived from this spacing. The small epsilon in `inv_width2` prevents
  // numerical issues when `width` is very small.
  RadialBasis(int num_rbf, float cutoff) {
    centers.resize(num_rbf);
    if (num_rbf <= 1) {
      centers[0] = 0.0f;
      width = cutoff;
    } else {
      float delta = cutoff / static_cast<float>(num_rbf - 1);
      for (int i = 0; i < num_rbf; ++i) {
        centers[i] = delta * i; // 0 .. cutoff
      }
      width = delta;
    }
    inv_width2 = 1.0f / (width * width + kInvWidth2Eps);
  }

  // Expand a single distance d → [num_rbf] features and append to an output
  // buffer.
  void expand_append(float d, std::vector<float> &out) const {
    out.reserve(out.size() + centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
      float diff = d - centers[i];
      out.push_back(std::exp(-(diff * diff) * inv_width2));
    }
  }
};
