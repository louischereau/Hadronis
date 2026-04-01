#pragma once

struct Vec3 {
  float x, y, z;
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  float norm2() const { return x * x + y * y + z * z; }
};
