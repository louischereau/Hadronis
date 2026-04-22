#pragma once

#include "linear_layer.hpp"
#include "silu.hpp"
#include "thread_pool.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

struct PaiNNUpdate {
  int hidden_dim;

  // PaiNN update block projections over per-atom states.
  // U and V act on each vector component block independently: [F -> F].
  // The scalar MLP follows the diagram: [2F -> F] -> SiLU -> [F -> 3F].
  LinearLayer U;       // [F -> F]
  LinearLayer V;       // [F -> F]
  LinearLayer linear1; // [2F -> F]
  LinearLayer linear2; // [F -> 3F]

private:
  PaiNNUpdateScratch main_scratch_;
  std::unique_ptr<PersistentThreadPool<PaiNNUpdateScratch>> thread_pool_;

  std::size_t validate_inputs(int n_atoms, const std::vector<float> &s,
                              const std::vector<float> &v) const {
    if (n_atoms < 0) {
      throw std::runtime_error(
          "PaiNNUpdate::forward received negative atom count");
    }
    const std::size_t scalar_size = static_cast<std::size_t>(n_atoms) *
                                    static_cast<std::size_t>(hidden_dim);
    if (s.size() != scalar_size) {
      throw std::runtime_error("PaiNNUpdate::forward scalar shape mismatch");
    }
    if (v.size() != scalar_size * 3u) {
      throw std::runtime_error("PaiNNUpdate::forward vector shape mismatch");
    }
    return scalar_size;
  }

public:
  explicit PaiNNUpdate(int hidden_dim)
      : hidden_dim(hidden_dim), U(hidden_dim, hidden_dim),
        V(hidden_dim, hidden_dim), linear1(2 * hidden_dim, hidden_dim),
        linear2(hidden_dim, 3 * hidden_dim), main_scratch_(hidden_dim),
        thread_pool_(std::make_unique<PersistentThreadPool<PaiNNUpdateScratch>>(
            1u, [hidden_dim]() { return PaiNNUpdateScratch(hidden_dim); })) {}

  void forward(int n_atoms, const std::vector<float> &s,
               const std::vector<float> &v, std::vector<float> &ds_atom,
               std::vector<float> &dv_atom) {
    const std::size_t scalar_size = validate_inputs(n_atoms, s, v);
    const std::size_t vector_size = scalar_size * 3u;

    ds_atom.assign(scalar_size, 0.0f);
    dv_atom.assign(vector_size, 0.0f);

    auto process_atom = [&](int atom, PaiNNUpdateScratch &scratch) {
      const std::size_t s_base =
          static_cast<std::size_t>(atom) * static_cast<std::size_t>(hidden_dim);
      const std::size_t v_base = 3u * s_base;
      const std::size_t x_base = v_base;
      const std::size_t y_base = v_base + static_cast<std::size_t>(hidden_dim);
      const std::size_t z_base =
          v_base + 2u * static_cast<std::size_t>(hidden_dim);

      const std::span<const float> v_atom_block(
          v.data() + v_base, 3u * static_cast<std::size_t>(hidden_dim));
      U.forward(v_atom_block, 3, scratch.u);
      V.forward(v_atom_block, 3, scratch.v);

      const float *ux = scratch.u.data();
      const float *uy = ux + hidden_dim;
      const float *uz = ux + 2 * hidden_dim;
      const float *vx = scratch.v.data();
      const float *vy = vx + hidden_dim;
      const float *vz = vx + 2 * hidden_dim;
      float *stack_norm = scratch.stacked.data();

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUG__)
#pragma GCC ivdep
#endif
      for (int f = 0; f < hidden_dim; ++f) {
        stack_norm[f] =
            std::sqrt(vx[f] * vx[f] + vy[f] * vy[f] + vz[f] * vz[f] + 1e-8f);
      }

      linear1.forward(stack_norm, s.data() + s_base, scratch.hidden);
      silu_inplace(scratch.hidden);
      linear2.forward(scratch.hidden, scratch.gates);

      const float *a_vv = scratch.gates.data();
      const float *a_sv = a_vv + hidden_dim;
      const float *a_ss = a_vv + 2 * hidden_dim;

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUG__)
#pragma GCC ivdep
#endif
      for (int f = 0; f < hidden_dim; ++f) {
        const std::size_t idx = static_cast<std::size_t>(f);
        const float dot_uv = ux[f] * vx[f] + uy[f] * vy[f] + uz[f] * vz[f];

        ds_atom[s_base + idx] = a_ss[f] + a_sv[f] * dot_uv;

        dv_atom[x_base + idx] = a_vv[f] * ux[f];
        dv_atom[y_base + idx] = a_vv[f] * uy[f];
        dv_atom[z_base + idx] = a_vv[f] * uz[f];
      }
    };

    thread_pool_->run(n_atoms, process_atom, main_scratch_);
  }
};
