#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

// Atomic embedding lookup table.
//
// Stores a weight matrix of shape [max_z, hidden_dim] (row-major).
// Each row contains the learned embedding for a given atomic number.
struct AtomicEmbedding {
  int max_z;
  int hidden_dim;
  std::vector<float> weights; // [max_z * hidden_dim], row-major

  AtomicEmbedding() : max_z(0), hidden_dim(0) {}

  // Construct from pre-loaded weight data (e.g., read from a binary file).
  // weights_data must have max_z * hidden_dim elements in row-major order.
  AtomicEmbedding(int max_z, int hidden_dim, const float *weights_data)
      : max_z(max_z), hidden_dim(hidden_dim),
        weights(weights_data, weights_data + max_z * hidden_dim) {}

  // Construct with zero-initialised weights (useful before loading from disk).
  AtomicEmbedding(int max_z, int hidden_dim)
      : max_z(max_z), hidden_dim(hidden_dim),
        weights(static_cast<std::size_t>(max_z) * hidden_dim, 0.0f) {}

  // Write the embedding for atomic number z into out[0..hidden_dim).
  // z is clamped to [0, max_z - 1] to match the Python
  // `z.clamp(max=max_z - 1)` guard.
  void lookup(int z, float *out) const {
    const int clamped = std::clamp(z, 0, max_z - 1);
    const float *row =
        weights.data() + static_cast<std::size_t>(clamped) * hidden_dim;
    std::copy_n(row, hidden_dim, out);
  }

  // Fill a per-atom scalar feature matrix s[n_atoms * hidden_dim] by
  // looking up each atomic number in the provided array.
  // s must be pre-allocated to n_atoms * hidden_dim floats.
  void embed_all(const int *atomic_numbers, int n_atoms, float *s) const {
    for (int i = 0; i < n_atoms; ++i) {
      lookup(atomic_numbers[i], s + static_cast<std::size_t>(i) * hidden_dim);
    }
  }
};
