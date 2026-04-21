#pragma once
#include "models/cellList.hpp"
#include "models/edgeGraph.hpp"
#include "models/rbf.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <cmath>

struct GraphBuilder {
  int N;
  float box_size;
  float r_cut;  // physical interaction / RBF cutoff
  float r_skin; // neighbor-list skin
  float r_list; // neighbor search radius = r_cut + r_skin
  CellList cell_list;
  RadialBasis rbf;
  EdgeGraph edge_graph;

  GraphBuilder() = default;

  GraphBuilder(int N, float box_size, float r_cut, float r_skin, int num_rbf)
      : N(N), box_size(box_size), r_cut(r_cut), r_skin(r_skin),
        r_list(r_cut + r_skin),
        // Cell size must be >= neighbour search radius so all pairs within
        // r_cut are discoverable when scanning neighbouring cells.
        cell_list(N, box_size, r_list),
        // Tie the RBF cutoff to the physical interaction cutoff.
        rbf(num_rbf, r_cut) {}

  void build(const std::vector<Vec3> &pos) {
    // Rebuild cell list from positions
    cell_list.build(pos);

    // Reset edge graph storage for this build
    edge_graph.edge_src.clear();
    edge_graph.edge_dst.clear();
    edge_graph.edge_rbf.clear();
    edge_graph.edge_rvec.clear();

    const int N_pos = static_cast<int>(pos.size());
    // Only keep edges within the physical cutoff. The neighbour list radius
    // (r_list) may be larger to allow reuse with a skin, but r_cut defines
    // the maximum distance for RBF features.
    const float r2 = r_cut * r_cut;
    const float box = cell_list.box;

    const int n_rbf = static_cast<int>(rbf.centers.size());

    // Heuristic reserve: estimate neighbour count from density so we avoid
    // repeated reallocations in very dense systems.
    const int approx_max_neighbors = estimate_max_neighbours(N_pos, box, r_cut);
    const std::size_t reserve_edges =
        static_cast<std::size_t>(N_pos) * approx_max_neighbors;
    edge_graph.edge_src.reserve(reserve_edges);
    edge_graph.edge_dst.reserve(reserve_edges);
    if (n_rbf > 0)
      edge_graph.edge_rbf.reserve(reserve_edges *
                                  static_cast<std::size_t>(n_rbf));
    // Always reserve space for per-edge unit vectors r_hat (3 floats/edge).
    edge_graph.edge_rvec.reserve(reserve_edges * 3u);

    for (int i = 0; i < N_pos; ++i) {
      // Cell of particle i
      auto [cx, cy, cz] = cell_list.cell_coords(pos[i]);

      // Loop over 27 neighbouring cells (periodic)
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            int c = cell_list.neighbour_cell_index(cx, cy, cz, dx, dy, dz);

            for (int j = cell_list.head[c]; j != -1; j = cell_list.next[j]) {
              Vec3 d = pos[j] - pos[i];
              // Minimum-image correction (cubic periodic box)
              wrap_minimum_image(d, box);

              const float d2 = d.norm2();
              if (d2 > 1e-10f && d2 < r2) {
                const float dist = std::sqrt(d2);
                edge_graph.edge_src.push_back(i);
                edge_graph.edge_dst.push_back(j);
                edge_graph.edge_rvec.push_back(d.x);
                edge_graph.edge_rvec.push_back(d.y);
                edge_graph.edge_rvec.push_back(d.z);
                rbf.expand_append(dist, edge_graph.edge_rbf);
              }
            }
          }
    }

    // Sort edges by destination to improve cache locality when aggregating
    // messages into destination atoms (e.g. PaiNN-style models).
    edge_graph.sort_by_dst();
  }
};
