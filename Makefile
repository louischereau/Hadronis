# Variables
VENV := .venv
UV := uv
INSTALL_STAMP := $(VENV)/.install_stamp
CPP_SOURCES := $(wildcard src/*.cpp)

.PHONY: help dev release test test-cpp test-all lint format clean bench format-cpp lint-cpp perf perf-report perf-stat perf-threads

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
	@echo "  clean       Nuke build artifacts and venv"
	@echo "  perf        Profile Hadronis single-molecule latency with perf"
	@echo "  perf-report Open perf report for latest run"
	@echo "  perf-stat   perf stat on single-molecule latency benchmark"
	@echo "  perf-threads perf stat on thread-scaling benchmark"

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
	$(UV) run pytest tests/

test-cpp:
	mkdir -p build
	cmake -S . -B build -DHADRONIS_ENABLE_SIMD=ON
	cmake --build build
	ctest --test-dir build

test: test-python test-cpp

perf: $(INSTALL_STAMP)
	perf record -F 99 -g -- \
		$(UV) run python benchmarks/python/benchmark_single_molecule_latency.py \
			--backend hadronis --sizes 256 --n-warmup 100 --n-iters 2000

perf-report:
	perf report

perf-stat: $(INSTALL_STAMP)
	perf stat -r 5 -d -e cycles,instructions,branches,branch-misses,cache-references,cache-misses -- \
		$(UV) run python benchmarks/python/benchmark_single_molecule_latency.py \
			--backend hadronis --sizes 256 --n-warmup 100 --n-iters 2000

perf-threads: $(INSTALL_STAMP)
	perf stat -r 3 -d -e cycles,instructions,branches,branch-misses,cache-references,cache-misses -- \
		$(UV) run python benchmarks/python/benchmark_thread_scaling.py \
			--sizes 64,256,1024 --threads 1,2,4,8,16 --n-warmup 50 --n-iters 1000
