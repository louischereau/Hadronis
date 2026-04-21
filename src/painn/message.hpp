#pragma once

#include "../utils/activations.hpp"
#include "models/edgeGraph.hpp"
#include "models/vec3.hpp"
#include "painn/layout.hpp"
#include "painn/linear_layer.hpp"
#include "painn/thread_pool.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

struct PaiNNMessage {
  int hidden_dim;
  int n_rbf;

  // Message MLP over per-edge features.
  LinearLayer mlp_linear1; // [F -> F]
  LinearLayer mlp_linear2; // [F -> 3F]
  LinearLayer mlp_linear3; // [K -> 3F] learned RBF filter

  PaiNNMessage() : PaiNNMessage(128, 20) {}

  PaiNNMessage(int hidden_dim, int n_rbf)
      : hidden_dim(hidden_dim), n_rbf(n_rbf),
        mlp_linear1(hidden_dim, hidden_dim),
        mlp_linear2(hidden_dim, 3 * hidden_dim),
        mlp_linear3(n_rbf, 3 * hidden_dim), main_scratch_(hidden_dim),
        thread_pool_(
            std::make_unique<PersistentThreadPool<PaiNNMessageScratch>>(
                std::max(1u, std::thread::hardware_concurrency()),
                [hidden_dim]() { return PaiNNMessageScratch(hidden_dim); })) {}

private:
  PaiNNMessageScratch main_scratch_;
  std::unique_ptr<PersistentThreadPool<PaiNNMessageScratch>> thread_pool_;

  std::size_t validate_inputs(int n_atoms, const std::vector<float> &s,
                              const std::vector<float> &v,
                              const EdgeGraph &graph, float r_cut) const {
    if (n_atoms < 0) {
      throw std::runtime_error(
          "PaiNNMessage::forward received negative atom count");
    }
    if (r_cut <= 0.0f) {
      throw std::runtime_error("PaiNNMessage::forward cutoff must be positive");
    }
    const std::size_t scalar_size = static_cast<std::size_t>(n_atoms) *
                                    static_cast<std::size_t>(hidden_dim);
    if (s.size() != scalar_size) {
      throw std::runtime_error("PaiNNMessage::forward scalar shape mismatch");
    }
    if (v.size() != scalar_size * 3u) {
      throw std::runtime_error("PaiNNMessage::forward vector shape mismatch");
    }
    return scalar_size;
  }

public:
  // Processes all destination atoms in a batch, accumulating per-destination
  // message contributions from their respective source atoms using a thread
  // pool. This matches the batched interface of PaiNNUpdate::forward.
  void forward(int n_atoms, const std::vector<float> &s,
               const std::vector<float> &v, const EdgeGraph &graph, float r_cut,
               std::vector<float> &ds_atom, std::vector<float> &dv_atom) {
    const std::size_t scalar_size =
        validate_inputs(n_atoms, s, v, graph, r_cut);
    const std::size_t F = static_cast<std::size_t>(hidden_dim);

    ds_atom.assign(scalar_size, 0.0f);
    dv_atom.assign(scalar_size * 3u, 0.0f);

    const float r_cut2 = r_cut * r_cut;

    auto process_atom = [&](int dst, PaiNNMessageScratch &scratch) {
      const std::size_t dst_s_base = static_cast<std::size_t>(dst) * F;
      const std::size_t dst_v_base = 3u * dst_s_base;

      std::fill(scratch.ds_sum.begin(), scratch.ds_sum.end(), 0.0f);
      std::fill(scratch.dv_sum.begin(), scratch.dv_sum.end(), 0.0f);

      const std::size_t e_begin =
          graph.dst_offsets[static_cast<std::size_t>(dst)];
      const std::size_t e_end =
          graph.dst_offsets[static_cast<std::size_t>(dst) + 1];

      for (std::size_t e = e_begin; e < e_end; ++e) {
        const int src = graph.edge_src[e];
        const float rx = graph.edge_rvec[e * 3];
        const float ry = graph.edge_rvec[e * 3 + 1];
        const float rz = graph.edge_rvec[e * 3 + 2];
        const float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 <= 1e-10f || r2 >= r_cut2)
          continue;

        const float r = std::sqrt(r2);
        const float cutoff = f_cut(r, r_cut);
        const float inv_r = 1.0f / r;
        const float r_hat_x = rx * inv_r;
        const float r_hat_y = ry * inv_r;
        const float r_hat_z = rz * inv_r;

        const std::size_t src_s_base = static_cast<std::size_t>(src) * F;
        const std::size_t src_v_base = 3u * src_s_base;

        mlp_linear1.forward(std::span<const float>(s.data() + src_s_base, F),
                            scratch.phi);
        silu_inplace(scratch.phi);
        mlp_linear2.forward(std::span<const float>(scratch.phi.data(), F),
                            scratch.mix);

        const std::span<const float> rbf_e(
            graph.edge_rbf.data() + e * static_cast<std::size_t>(n_rbf),
            static_cast<std::size_t>(n_rbf));
        mlp_linear3.forward(rbf_e, scratch.filter);

        for (std::size_t i = 0; i < 3u * F; ++i)
          scratch.mix[i] *= scratch.filter[i] * cutoff;

        for (int f = 0; f < hidden_dim; ++f) {
          const std::size_t idx = static_cast<std::size_t>(f);
          const std::size_t x_idx = vector_feature_index(0, f, hidden_dim);
          const std::size_t y_idx = vector_feature_index(1, f, hidden_dim);
          const std::size_t z_idx = vector_feature_index(2, f, hidden_dim);

          scratch.ds_sum[idx] += scratch.mix[2u * F + idx];
          scratch.dv_sum[x_idx] += v[src_v_base + x_idx] * scratch.mix[idx] +
                                   r_hat_x * scratch.mix[F + idx];
          scratch.dv_sum[y_idx] += v[src_v_base + y_idx] * scratch.mix[idx] +
                                   r_hat_y * scratch.mix[F + idx];
          scratch.dv_sum[z_idx] += v[src_v_base + z_idx] * scratch.mix[idx] +
                                   r_hat_z * scratch.mix[F + idx];
        }
      }

      std::copy(scratch.ds_sum.begin(), scratch.ds_sum.end(),
                ds_atom.begin() + static_cast<std::ptrdiff_t>(dst_s_base));
      std::copy(scratch.dv_sum.begin(), scratch.dv_sum.end(),
                dv_atom.begin() + static_cast<std::ptrdiff_t>(dst_v_base));
    };

    thread_pool_->run(n_atoms, process_atom, main_scratch_);
  }

  float f_cut(float r, float r_cut) {
    if (r >= r_cut)
      return 0.0f;
    const float omega = std::numbers::pi_v<float> / r_cut;
    return 0.5f * (std::cos(r * omega) + 1.0f);
  }
};
