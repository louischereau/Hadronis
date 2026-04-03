#pragma once

#include "../models/vec3.hpp"

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
