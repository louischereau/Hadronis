#pragma once
#include <cmath>
#include <vector>

struct RadialBasis {
  static constexpr float kDefaultWidth = 1.0f;
  static constexpr float kInvWidth2Eps = 1e-8f;

  std::vector<float> centers;
  float width;
  float inv_width2;

  RadialBasis() : centers(), width(kDefaultWidth), inv_width2(1.0f) {}

  RadialBasis(int num_rbf, float cutoff) {
    centers.resize(num_rbf);
    if (num_rbf <= 1) {
      centers[0] = 0.0f;
      width = cutoff;
    } else {
      float delta = cutoff / static_cast<float>(num_rbf - 1);
      for (int i = 0; i < num_rbf; ++i) {
        centers[i] = delta * i;
      }
      width = delta;
    }
    inv_width2 = 1.0f / (width * width + kInvWidth2Eps);
  }

  void expand_append(float d, std::vector<float> &out) const {
    const std::size_t n = centers.size();
    const std::size_t offset = out.size();
    out.resize(offset + n);

    const float *__restrict center_ptr = centers.data();
    float *__restrict out_ptr = out.data() + offset;

    for (std::size_t i = 0; i < n; ++i) {
      float diff = d - center_ptr[i];
      out_ptr[i] = expf(-(diff * diff) * inv_width2);
    }
  }
};
