#pragma once

#include "../models/vec3.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

inline void wrap_minimum_image(Vec3 &d, float box) {
  const float half_box = 0.5f * box;

  if (d.x > half_box)
    d.x -= box;
  else if (d.x < -half_box)
    d.x += box;

  if (d.y > half_box)
    d.y -= box;
  else if (d.y < -half_box)
    d.y += box;

  if (d.z > half_box)
    d.z -= box;
  else if (d.z < -half_box)
    d.z += box;
}

// Heuristic estimate of the maximum expected neighbours per particle in a
// homogeneous system, used to reserve edge list capacity. Returns a value
// clamped to a reasonable range to avoid pathological sizes.
inline int estimate_max_neighbours(int N, float box, float r_cut) {
  if (N <= 0 || box <= 0.0f || r_cut <= 0.0f)
    return 64;

  const float volume = box * box * box;
  const float neighbour_volume =
      (4.0f / 3.0f) * std::numbers::pi_v<float> * r_cut * r_cut * r_cut;
  const float density = static_cast<float>(N) / volume;
  const float expected_neighbours = density * neighbour_volume;

  // Safety factor to account for fluctuations; clamp to a sane range.
  const float approx = expected_neighbours * 2.0f;
  return std::clamp(static_cast<int>(approx), 16, 512);
}
