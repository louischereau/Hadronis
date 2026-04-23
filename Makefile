# Variables
VENV := .venv
UV := uv
INSTALL_STAMP := $(VENV)/.install_stamp
PROFILE_INSTALL_STAMP := $(VENV)/.install_stamp_profiled
CPP_SOURCES := $(wildcard src/*.cpp)

.PHONY: help dev release test test-cpp test-all lint format clean bench benchmark bench-cpp format-cpp lint-cpp perf perf-report perf-stat perf-threads bench-memory

help:
	@echo "Hadronis Development:"
	@echo "  build-cpp   Build C++/pybind11 engine"
	@echo "  dev         Build & sync the engine (Python)"
	@echo "  format      Automatically fix code style (Python)"
	@echo "  lint        Check code quality without fixing"
	@echo "  format-cpp  Format C++ sources with clang-format"
	@echo "  lint-cpp    Lint C++ sources with clang-tidy"
	@echo "  test        Run Python test suites"
	@echo "  test-cpp    Build and run C++ tests (ctest)"
	@echo "  test-all    Run both Python and C++ tests"
	@echo "  bench-cpp   Run C++ geometry microbenchmarks"
	@echo "  clean       Nuke build artifacts and venv"
	@echo "  perf        Profile Hadronis single-molecule latency with perf"
	@echo "  perf-report Open perf report for latest run"
	@echo "  perf-stat   perf stat on single-molecule latency benchmark"
	@echo "  perf-threads perf stat on thread-scaling benchmark"
	@echo "  bench-memory Benchmark RSS growth under repeated inference"

$(VENV):
	$(UV) venv $(VENV)

$(INSTALL_STAMP): pyproject.toml | $(VENV)
	@echo "--- Syncing Dependencies ---"
	$(UV) pip install -e .[dev]
	@touch $(INSTALL_STAMP)

build:
	mkdir -p build
	cmake -S . -B build -DHADRONIS_ENABLE_SIMD=ON
	cmake --build build
	@echo "--- C++/pybind11 engine built (SIMD enabled) ---"

format: $(INSTALL_STAMP)
	@echo "--- Formatting C++ sources with clang-format ---"
	clang-format -i $(CPP_SOURCES)
	@echo "--- Formatting Python ---"
	$(UV) run ruff format python/

# Updated Lint: This now checks if formatting is correct without changing it
lint: $(INSTALL_STAMP)
	@echo "--- Linting C++ sources with clang-tidy ---"
	clang-tidy $(CPP_SOURCES) -p build
	@echo "--- Checking Python ---"
	$(UV) run ruff check python/
	$(UV) run ruff format --check python/

test-python: $(INSTALL_STAMP)
	$(UV) run pytest -s tests/

test-cpp:
	mkdir -p build
	cmake -S . -B build -DHADRONIS_ENABLE_SIMD=ON
	cmake --build build
	ctest --test-dir build

test: test-python test-cpp

$(PROFILE_INSTALL_STAMP): pyproject.toml | $(VENV)
	@echo "--- Syncing Dependencies (profiled build) ---"
	CMAKE_ARGS="-DHADRONIS_PROFILE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo" $(UV) pip install -e .[dev]
	@touch $(PROFILE_INSTALL_STAMP)

perf: $(PROFILE_INSTALL_STAMP)
	perf record -F 99 -g -- \
		$(UV) run python benchmarks/python/benchmark_single_molecule_latency.py \
			--backend hadronis --sizes 256 --n-warmup 100 --n-iters 2000

perf-report:
	perf report

perf-stat: $(PROFILE_INSTALL_STAMP)
	perf stat -r 5 -d -e cycles,instructions,branches,branch-misses,cache-references,cache-misses -- \
		$(UV) run python benchmarks/python/benchmark_single_molecule_latency.py \
			--backend hadronis --sizes 256 --n-warmup 100 --n-iters 2000

perf-threads: $(PROFILE_INSTALL_STAMP)
	perf stat -r 3 -d -e cycles,instructions,branches,branch-misses,cache-references,cache-misses -- \
		$(UV) run python benchmarks/python/benchmark_thread_scaling.py \
			--sizes 64,256,1024 --threads 1,2,4,8,16 --n-warmup 50 --n-iters 1000

bench: $(PROFILE_INSTALL_STAMP)
	@echo "--- Benchmark: single-molecule latency (Hadronis + PyTorch PaiNN) ---"
	$(UV) run python benchmarks/python/benchmark_single_molecule_latency.py \
		--backend both --sizes 64 --n-warmup 5 --n-iters 20
	@echo
	@echo "--- Benchmark: MD-style trace ---"
	$(UV) run python benchmarks/python/benchmark_md_trace.py
	@echo
	@echo "--- Benchmark: thread-scaling ---"
	$(UV) run python benchmarks/python/benchmark_thread_scaling.py
	@echo
	@echo "--- Benchmark: memory growth under repeated inference ---"
	$(UV) run python benchmarks/python/benchmark_memory_growth.py --n-atoms 128 --n-iters 5

bench-cpp:
	rm -rf build
	mkdir -p build
	cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$(shell nproc)
	./build/src/hadronis_bench_neighbors
	./build/src/hadronis_bench_painn

bench-memory: $(INSTALL_STAMP)
	@echo "--- Benchmark: memory growth under repeated inference ---"
	$(UV) run python benchmarks/python/benchmark_memory_growth.py --n-atoms 128 --n-iters 5
