
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "neighbors/graph_builder.hpp"
#include "painn/painn.hpp"
#include "safetensors/safetensors.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Default PaiNN hyper-parameters (must match the saved weight file).
static constexpr int kHiddenDim = 128;
constexpr int kNumRbf = 20; // must match painn_(kHiddenDim, 3, kNumRbf)

class HadronisEngine {
  GraphBuilder graph_builder_;
  PaINN painn_;
  float r_cut_;
  int max_neighbors_;
  int n_threads_;
  // Cached construction parameters: GraphBuilder is expensive to rebuild
  // (allocates ~420 KB of edge vectors). When N and box_size are stable
  // across calls (the common case), reuse the existing object.
  int cached_N_ = 0;
  float cached_box_size_ = -1.0f;

public:
  HadronisEngine(const std::string &weight_path, float cutoff,
                 int max_neighbors, int n_threads = 1)
      : painn_(kHiddenDim, 3, kNumRbf, n_threads), r_cut_(cutoff),
        max_neighbors_(max_neighbors), n_threads_(n_threads) {
    load_weights(weight_path, painn_);
  }

  float predict(
      py::array_t<int, py::array::c_style | py::array::forcecast>
          atomic_numbers,
      py::array_t<float, py::array::c_style | py::array::forcecast> positions) {
    validate_inputs(atomic_numbers, positions);

    const int N = static_cast<int>(atomic_numbers.shape(0));

    auto pos_view = positions.unchecked<2>();
    std::vector<Vec3> pos(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
      pos[static_cast<std::size_t>(i)] = {pos_view(i, 0), pos_view(i, 1),
                                          pos_view(i, 2)};
    }

    // Compute the bounding box of the system so that the minimum-image
    // convention never wraps any real pair (effectively disabling PBC for
    // isolated molecules).  We need box > 2 * max_extent so that the
    // half-box is always larger than the largest inter-atom displacement.
    float xmin = pos[0].x, xmax = pos[0].x;
    float ymin = pos[0].y, ymax = pos[0].y;
    float zmin = pos[0].z, zmax = pos[0].z;
    for (const auto &p : pos) {
      xmin = std::min(xmin, p.x);
      xmax = std::max(xmax, p.x);
      ymin = std::min(ymin, p.y);
      ymax = std::max(ymax, p.y);
      zmin = std::min(zmin, p.z);
      zmax = std::max(zmax, p.z);
    }
    const float extent = std::max({xmax - xmin, ymax - ymin, zmax - zmin});
    // Minimum safe box: half-box must exceed the largest atom-pair displacement
    // (= extent) so PBC never wraps real neighbours.  Add a small margin and
    // ensure the box is wide enough for at least one cell-list cell.
    const float box_size = std::max(2.0f * extent + 0.01f, r_cut_ + 0.01f);

    // Shift positions into [r_cut_, r_cut_ + extent] so they sit well inside
    // the box and the cell-list clamping never clips real neighbours.
    const float sx = r_cut_ - xmin;
    const float sy = r_cut_ - ymin;
    const float sz = r_cut_ - zmin;
    for (auto &p : pos) {
      p.x += sx;
      p.y += sy;
      p.z += sz;
    }

    if (N != cached_N_ || box_size != cached_box_size_) {
      // r_skin=0: we always rebuild the graph from scratch so a neighbour-list
      // skin provides no benefit; using r_list=r_cut yields smaller cells and
      // fewer redundant pair checks.
      graph_builder_ = GraphBuilder(N, box_size, r_cut_, 0.0f, kNumRbf);
      cached_N_ = N;
      cached_box_size_ = box_size;
    }
    graph_builder_.build(pos);

    return painn_.predict(&atomic_numbers.unchecked<1>()(0), N,
                          graph_builder_.edge_graph, r_cut_);
  }

private:
  static void validate_inputs(const py::array_t<int> &atomic_numbers,
                              const py::array_t<float> &positions) {
    if (atomic_numbers.ndim() != 1)
      throw std::runtime_error("atomic_numbers must be 1D [n_atoms]");
    if (positions.ndim() != 2 || positions.shape(1) != 3)
      throw std::runtime_error("positions must have shape (n_atoms, 3)");
  }

  // Load weights from a .safetensors file into the PaINN model.
  //
  // Expected key schema (all float32, row-major [out_dim, in_dim]):
  //
  //   embedding.weight                                [max_z, hidden_dim]
  //
  //   interactions.{i}.message.mlp_linear1.weight     [hidden_dim, hidden_dim]
  //   interactions.{i}.message.mlp_linear1.bias       [hidden_dim]
  //   interactions.{i}.message.mlp_linear2.weight     [3*hidden_dim,
  //   hidden_dim] interactions.{i}.message.mlp_linear2.bias [3*hidden_dim]
  //   interactions.{i}.message.mlp_linear3.weight     [3*hidden_dim, n_rbf]
  //   interactions.{i}.message.mlp_linear3.bias       [3*hidden_dim]
  //
  //   interactions.{i}.update.U.weight                [hidden_dim, hidden_dim]
  //   interactions.{i}.update.U.bias                  [hidden_dim]
  //   interactions.{i}.update.V.weight                [hidden_dim, hidden_dim]
  //   interactions.{i}.update.V.bias                  [hidden_dim]
  //   interactions.{i}.update.linear1.weight          [hidden_dim,
  //   2*hidden_dim] interactions.{i}.update.linear1.bias [hidden_dim]
  //   interactions.{i}.update.linear2.weight          [3*hidden_dim,
  //   hidden_dim] interactions.{i}.update.linear2.bias [3*hidden_dim]
  //
  //   readout.linear1.weight                          [hidden_dim, hidden_dim]
  //   readout.linear1.bias                            [hidden_dim]
  //   readout.linear2.weight                          [1, hidden_dim]
  //   readout.linear2.bias                            [1]
  static void load_weights(const std::string &path, PaINN &model) {
    const Safetensors st(path);

    auto load_linear = [&](LinearLayer &layer, const std::string &prefix) {
      layer.set_weight(to_vec(st.get(prefix + ".weight").data));
      layer.set_bias(to_vec(st.get(prefix + ".bias").data));
    };

    // Embedding
    {
      const auto &t = st.get("embedding.weight");
      model.embedding_layer.weights = to_vec(t.data);
    }

    // Interaction layers
    for (int i = 0; i < model.n_interactions; ++i) {
      const std::string base = "interactions." + std::to_string(i);
      PaINNInteraction &layer =
          model.interaction_layers[static_cast<std::size_t>(i)];

      load_linear(layer.message_layer.mlp_linear1,
                  base + ".message.mlp_linear1");
      load_linear(layer.message_layer.mlp_linear2,
                  base + ".message.mlp_linear2");
      load_linear(layer.message_layer.mlp_linear3,
                  base + ".message.mlp_linear3");

      load_linear(layer.update_layer.U, base + ".update.U");
      load_linear(layer.update_layer.V, base + ".update.V");
      load_linear(layer.update_layer.linear1, base + ".update.linear1");
      load_linear(layer.update_layer.linear2, base + ".update.linear2");
    }

    // Readout MLP
    load_linear(model.linear1, "readout.linear1");
    load_linear(model.linear2, "readout.linear2");
  }

  static std::vector<float> to_vec(std::span<const float> s) {
    return {s.begin(), s.end()};
  }
};

PYBIND11_MODULE(_lowlevel, m) {
  py::class_<HadronisEngine>(m, "HadronisEngine")
      .def(py::init<const std::string &, float, int, int>(),
           py::arg("weight_path"), py::arg("cutoff") = 5.0f,
           py::arg("max_neighbors") = 64, py::arg("n_threads") = 1)
      .def("predict", &HadronisEngine::predict, py::arg("atomic_numbers"),
           py::arg("positions"));
}
