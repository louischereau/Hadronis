#pragma once

#include <algorithm>
#include <numbers>

// Shared indexing helpers for PaiNN vector features.
// Per-atom vector states are stored in component-major layout [3, F]:
// [x_0..x_F-1, y_0..y_F-1, z_0..z_F-1].
inline std::size_t vector_feature_index(int component, int feature,
                                        int hidden_dim) {
  return static_cast<std::size_t>(component) *
             static_cast<std::size_t>(hidden_dim) +
         static_cast<std::size_t>(feature);
}

// Atom-major storage for a full tensor [N, 3, F].
inline std::size_t atom_vector_feature_index(int atom, int component,
                                             int feature, int hidden_dim) {
  return static_cast<std::size_t>(atom) * 3u *
             static_cast<std::size_t>(hidden_dim) +
         vector_feature_index(component, feature, hidden_dim);
}
