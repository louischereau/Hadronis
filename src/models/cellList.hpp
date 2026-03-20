#include "vec3.hpp"
#include <algorithm>
#include <vector>
#pragma once

struct CellList {
  int nx, ny, nz;  // grid dimensions - number of cells along each axis
  float cell_size; // >= r_cutoff
  float box;       // cubic box side length

  std::vector<int> head; // head[cell] = first particle (-1 if empty)
  std::vector<int> next; // next[i]    = next particle in same cell (-1 if last)

  CellList() = default;

  CellList(int N, float box_size, float r_cut) : box(box_size) {
    // At least 1 cell per dimension, each cell >= r_cut
    nx = std::max(1, static_cast<int>(box / r_cut));
    ny = nx;
    nz = nx;
    cell_size = box / nx;

    head.assign(nx * ny * nz, -1);
    next.resize(N, -1);
  }

  // Flatten 3-D cell index → 1-D
  int cell_index(int cx, int cy, int cz) const {
    return cx + nx * (cy + ny * cz);
  }

  // Particle position → cell index (with PBC clamp)
  int cell_of(const Vec3 &p) const {
    int cx = static_cast<int>(p.x / cell_size);
    cx = std::clamp(cx, 0, nx - 1);
    int cy = static_cast<int>(p.y / cell_size);
    cy = std::clamp(cy, 0, ny - 1);
    int cz = static_cast<int>(p.z / cell_size);
    cz = std::clamp(cz, 0, nz - 1);
    return cell_index(
        cx, cy, cz); // cx, cy, cz are cell coordinates, not particle indices
  }

  // Build linked lists from current positions — O(N)
  void build(const std::vector<Vec3> &pos) {
    std::fill(head.begin(), head.end(), -1); // reset all cells to empty
    for (int i = static_cast<int>(pos.size()) - 1; i >= 0;
         --i) // iterate over each particle
    {
      int c = cell_of(pos[i]);
      next[i] =
          head[c]; // link the new particle in front of the previous first one
      head[c] = i; // i is now the new first particle in cell c
    }
  }
};
