#pragma once
#include "models/cellList.hpp"
#include "models/edgeGraph.hpp"
#include "models/rbf.hpp"
#include "utils/utils.hpp"
#include <cmath>

struct GraphBuilder {
  int N;
  float box_size;
  float r_list; // neighbor search radius (typically r_cut + r_skin)
  CellList cell_list;
  RadialBasis rbf;
  EdgeGraph edge_graph;

  GraphBuilder() = default;

  GraphBuilder(int N, float box_size, float r_cut, float r_skin, int num_rbf,
               float cutoff)
      : N(N), box_size(box_size), r_list(r_cut + r_skin),
        cell_list(N, box_size, r_cut), rbf(num_rbf, cutoff) {}

  void build(const std::vector<Vec3> &pos) {
    // Rebuild cell list from positions
    cell_list.build(pos);

    // Reset edge graph storage for this build
    edge_graph.edge_src.clear();
    edge_graph.edge_dst.clear();
    edge_graph.edge_rbf.clear();

    const int N_pos = static_cast<int>(pos.size());
    const float r2 = r_list * r_list;
    const float box = cell_list.box;

    const int n_rbf = static_cast<int>(rbf.centers.size());

    // Conservative reserve: assume up to 64 neighbors per atom by default
    const int approx_max_neighbors = 64;
    const std::size_t reserve_edges =
        static_cast<std::size_t>(N_pos) * approx_max_neighbors;
    edge_graph.edge_src.reserve(reserve_edges);
    edge_graph.edge_dst.reserve(reserve_edges);
    if (n_rbf > 0)
      edge_graph.edge_rbf.reserve(reserve_edges *
                                  static_cast<std::size_t>(n_rbf));

    for (int i = 0; i < N_pos; ++i) {
      // Cell of particle i
      auto [cx, cy, cz] = cell_list.cell_coords(pos[i]);

      // Loop over 27 neighbouring cells (periodic)
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            int c = cell_list.neighbour_cell_index(cx, cy, cz, dx, dy, dz);

            for (int j = cell_list.head[c]; j != -1; j = cell_list.next[j]) {
              if (j <= i)
                continue; // store each pair once

              Vec3 d = pos[j] - pos[i];
              // Minimum-image correction (cubic periodic box)
              wrap_minimum_image(d, box);

              const float d2 = d.norm2();
              if (d2 < r2) {
                const float dist = std::sqrt(d2);
                edge_graph.edge_src.push_back(i);
                edge_graph.edge_dst.push_back(j);
                rbf.expand_append(dist, edge_graph.edge_rbf);
              }
            }
          }
    }
  }
};
