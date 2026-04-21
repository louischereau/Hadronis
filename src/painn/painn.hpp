#pragma once

#include "embedding.hpp"
#include "interaction.hpp"
#include "linear_layer.hpp"
#include <string>
#include <vector>

struct PaINN {
  int hidden_dim;
  int n_interactions;
  int n_rbf;

  AtomicEmbedding embedding_layer;

  std::vector<PaINNInteraction> interaction_layers;

  // Readout MLP: F -> F -> 1
  LinearLayer linear1;
  LinearLayer linear2;

  PaINN(int hidden_dim, int n_interactions, int n_rbf)
      : hidden_dim(hidden_dim), n_interactions(n_interactions), n_rbf(n_rbf),
        embedding_layer(100, hidden_dim), linear1(hidden_dim, hidden_dim),
        linear2(hidden_dim, 1) {
    interaction_layers.reserve(static_cast<std::size_t>(n_interactions));
    for (int i = 0; i < n_interactions; ++i)
      interaction_layers.emplace_back(hidden_dim, n_rbf);
  }

  float predict(const int *atomic_numbers, int n_atoms, const EdgeGraph &graph,
                float r_cut) {
    float energy = 0.0f;

    const std::size_t scalar_size = static_cast<std::size_t>(n_atoms) *
                                    static_cast<std::size_t>(hidden_dim);
    std::vector<float> s(scalar_size, 0.0f);
    std::vector<float> v(scalar_size * 3u, 0.0f);
    embedding_layer.embed_all(atomic_numbers, n_atoms, s.data());

    for (auto &interaction : interaction_layers) {
      interaction.forward(n_atoms, s, v, graph, r_cut);
    }

    std::vector<float> hidden(static_cast<std::size_t>(hidden_dim)), out(1);
    for (int atom = 0; atom < n_atoms; ++atom) {
      std::span<const float> s_atom(s.data() + atom * hidden_dim, hidden_dim);
      linear1.forward(s_atom, hidden);
      silu_inplace(hidden);
      linear2.forward(hidden, out);
      energy += out[0];
    }

    return energy;
  }
};
