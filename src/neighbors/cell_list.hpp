#pragma once
#include "../models/vec3.hpp"
#include <algorithm>
#include <array>
#include <vector>

struct CellList {
  int nx, ny, nz;  // grid dimensions - number of cells along each axis
  float cell_size; // >= r_cutoff
  float box;       // cubic box side length

  std::vector<int> head; // head[cell] = first particle (-1 if empty)
  std::vector<int> next; // next[i]    = next particle in same cell (-1 if last)

  CellList() = default;

  CellList(int N, float box_size, float r_cut) : box(box_size) {
    nx = std::max(1, static_cast<int>(box / r_cut));
    ny = nx;
    nz = nx;
    cell_size = box / nx;

    head.assign(nx * ny * nz, -1);
    next.resize(N, -1);
  }

  int cell_index(int cx, int cy, int cz) const {
    return cx + nx * (cy + ny * cz);
  }

  int neighbour_cell_index(int cx, int cy, int cz, int dx, int dy,
                           int dz) const {
    const int ncx = (cx + dx + nx) % nx;
    const int ncy = (cy + dy + ny) % ny;
    const int ncz = (cz + dz + nz) % nz;
    return cell_index(ncx, ncy, ncz);
  }

  std::array<int, 3> cell_coords(const Vec3 &p) const {
    int cx = static_cast<int>(p.x / cell_size);
    int cy = static_cast<int>(p.y / cell_size);
    int cz = static_cast<int>(p.z / cell_size);

    cx = std::clamp(cx, 0, nx - 1);
    cy = std::clamp(cy, 0, ny - 1);
    cz = std::clamp(cz, 0, nz - 1);

    return {cx, cy, cz};
  }

  int cell_of(const Vec3 &p) const {
    auto [cx, cy, cz] = cell_coords(p);
    return cell_index(cx, cy, cz);
  }

  void build(const std::vector<Vec3> &pos) {
    if (next.size() < pos.size()) {
      next.resize(pos.size(), -1);
    }

    std::fill(head.begin(), head.end(), -1);
    for (int i = static_cast<int>(pos.size()) - 1; i >= 0; --i) {
      int c = cell_of(pos[i]);
      next[i] = head[c];
      head[c] = i;
    }
  }
};
