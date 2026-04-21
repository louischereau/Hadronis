#pragma once

// Thin wrapper around syoyo/safetensors-cpp that exposes a simple
// float32-focused interface compatible with the rest of the codebase.
//
// syoyo/safetensors-cpp is fetched via CMake FetchContent (see CMakeLists.txt).
// Include path is set via target_include_directories using
// ${safetensors_cpp_SOURCE_DIR}.

#include "safetensors.hh"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Safetensors {
  struct Tensor {
    std::span<const float> data; // zero-copy view into the loaded buffer
    std::vector<int64_t> shape;
  };

  explicit Safetensors(const std::string &path) {
    std::string warn, err;
    if (!safetensors::mmap_from_file(path, &st_, &warn, &err))
      throw std::runtime_error("Safetensors: failed to load '" + path +
                               "': " + err);

    // Build the typed index over all F32 tensors.
    // Spans point directly into the mmap'd region — no copy of tensor data.
    for (const auto &name : st_.tensors.keys()) {
      safetensors::tensor_t t;
      if (!st_.tensors.at(name, &t))
        continue;
      if (t.dtype != safetensors::kFLOAT32)
        continue;

      const auto *raw = reinterpret_cast<const float *>(st_.databuffer_addr +
                                                        t.data_offsets[0]);
      const std::size_t n_elems =
          (t.data_offsets[1] - t.data_offsets[0]) / sizeof(float);

      std::vector<int64_t> shape;
      shape.reserve(t.shape.size());
      for (auto d : t.shape)
        shape.push_back(static_cast<int64_t>(d));

      tensors_[name] =
          Tensor{std::span<const float>(raw, n_elems), std::move(shape)};
    }
  }

  // Returns the tensor with the given name. Throws if not found or not F32.
  const Tensor &get(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
      throw std::runtime_error("Safetensors: tensor not found: '" + name + "'");
    return it->second;
  }

  bool has(const std::string &name) const { return tensors_.count(name) > 0; }

  // Returns all F32 tensor names (useful for debugging key mismatches).
  std::vector<std::string> keys() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto &[k, _] : tensors_)
      out.push_back(k);
    return out;
  }

private:
  safetensors::safetensors_t st_;
  std::unordered_map<std::string, Tensor> tensors_;
};
