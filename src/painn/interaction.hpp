#pragma once

#include "message.hpp"
#include "update.hpp"
#include <vector>

struct PaINNInteraction {
  PaiNNMessage message_layer;
  PaiNNUpdate update_layer;
  int hidden_dim;

  PaINNInteraction(int hidden_dim, int n_rbf)
      : hidden_dim(hidden_dim), message_layer(hidden_dim, n_rbf),
        update_layer(hidden_dim) {}

  void forward(int natoms, std::vector<float> &s, std::vector<float> &v,
               const EdgeGraph &graph, float r_cut) {
    message_layer.forward(natoms, s, v, graph, r_cut, ds_edge_, dv_edge_);
    accumulate(s, v, ds_edge_, dv_edge_);

    update_layer.forward(natoms, s, v, ds_atom_, dv_atom_);
    accumulate(s, v, ds_atom_, dv_atom_);
  }

private:
  std::vector<float> ds_edge_, dv_edge_, ds_atom_, dv_atom_;

  static void accumulate(std::vector<float> &s, std::vector<float> &v,
                         const std::vector<float> &ds,
                         const std::vector<float> &dv) {
    for (std::size_t i = 0; i < s.size(); ++i)
      s[i] += ds[i];
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] += dv[i];
  }
};
