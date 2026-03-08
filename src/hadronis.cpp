#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

class HadronisEngine {
public:
  HadronisEngine(const std::string &weight_path, float cutoff,
                 int max_neighbors, int n_threads)
      : cutoff_(cutoff), max_neighbors_(max_neighbors), n_threads_(n_threads) {
    (void)weight_path; // Stub: load weights from disk in a real implementation
  }

  py::array_t<float> predict(
      py::array_t<int, py::array::c_style | py::array::forcecast>
          atomic_numbers,
      py::array_t<float, py::array::c_style | py::array::forcecast> positions,
      py::array_t<int, py::array::c_style | py::array::forcecast> batch) {
    // Basic shape checks (1D Z and batch, 2D positions with last dim = 3)
    if (atomic_numbers.ndim() != 1) {
      throw std::runtime_error("atomic_numbers must be 1D [n_atoms]");
    }
    if (positions.ndim() != 2 || positions.shape(1) != 3) {
      throw std::runtime_error("positions must have shape (n_atoms, 3)");
    }
    if (batch.ndim() != 1 || batch.shape(0) != atomic_numbers.shape(0)) {
      throw std::runtime_error("batch must have shape (n_atoms,)");
    }

    const py::ssize_t n_atoms = atomic_numbers.shape(0);

    // Stub implementation: return a zero vector of per-atom predictions.
    // A real implementation would build neighbor lists using `cutoff_` and
    // `max_neighbors_`, then run the GNN.
    py::array_t<float> out(n_atoms);
    auto out_mut = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n_atoms; ++i) {
      out_mut(i) = 0.0f;
    }
    return out;
  }

private:
  float cutoff_;
  int max_neighbors_;
  int n_threads_;
};

PYBIND11_MODULE(_lowlevel, m) {
  py::class_<HadronisEngine>(m, "HadronisEngine")
      .def(py::init<const std::string &, float, int, int>(),
           py::arg("weight_path"), py::arg("cutoff") = 5.0f,
           py::arg("max_neighbors") = 64, py::arg("n_threads") = 1)
      .def("predict", &HadronisEngine::predict, py::arg("atomic_numbers"),
           py::arg("positions"), py::arg("batch"));
}
