#pragma once

#include <cmath>
#include <vector>

struct RadialBasis {
  std::vector<float> centers;
  float width; // same for all centers
  float inv_width2;

  RadialBasis() : centers(), width(1.0f), inv_width2(1.0f) {}

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
    inv_width2 = 1.0f / (width * width + 1e-8f);
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
