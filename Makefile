# Variables
VENV := .venv
UV := uv
CARGO := cargo
INSTALL_STAMP := $(VENV)/.install_stamp

.PHONY: help dev release test lint format clean bench

help:
	@echo "Valence (uv-powered) Development:"
	@echo "  dev        Build & sync the engine"
	@echo "  format     Automatically fix code style (Rust & Python)"
	@echo "  lint       Check code quality without fixing"
	@echo "  test       Run Rust & Python test suites"
	@echo "  release    Package the engine into a .whl"
	@echo "  clean      Nuke build artifacts and venv"

$(VENV):
	$(UV) venv $(VENV)

$(INSTALL_STAMP): pyproject.toml | $(VENV)
	@echo "--- Syncing Dependencies ---"
	$(UV) pip install maturin ruff pytest numpy
	@touch $(INSTALL_STAMP)

dev: $(INSTALL_STAMP)
	@echo "--- Compiling Rust Engine ---"
	$(UV) run maturin develop

# NEW: The "Auto-Fix" command
format: $(INSTALL_STAMP)
	@echo "--- Formatting Rust ---"
	$(CARGO) fmt
	@echo "--- Formatting Python ---"
	$(UV) run ruff format python/

# Updated Lint: This now checks if formatting is correct without changing it
lint: $(INSTALL_STAMP)
	@echo "--- Checking Rust ---"
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy -- -D warnings
	@echo "--- Checking Python ---"
	$(UV) run ruff check python/
	$(UV) run ruff format --check python/

test: $(INSTALL_STAMP)
	$(CARGO) test --locked
	$(UV) run pytest tests/

release: $(INSTALL_STAMP)
	$(UV) run maturin build --release --out dist/


bench:
	$(CARGO) bench --bench molecular_bench

# Generate the X-ray of performance
profile:
	CARGO_PROFILE_RELEASE_DEBUG=true sudo $(CARGO) flamegraph --bench molecular_bench

# Build with the ultimate machine-specific optimizations (SIMD)
build-native:
	RUSTFLAGS="-C target-cpu=native" $(CARGO) build --release

# Audit the generated assembly to verify SIMD (AVX/SSE) instructions
# Audit the Fused Kernel (The method name changed to run_fused_with_model)
audit-asm:
	@echo "Searching for SIMD vectors in the fused kernel..."
	RUSTFLAGS="-C target-cpu=native" $(CARGO) asm valence::graph::MolecularGraph::run_fused_with_model_internal | grep -E "vaddps|vmulps|ymm|zmm" || echo "No SIMD detected."

sample:
	@samply record $(shell ls target/release/deps/molecular_bench-*) --bench

tracy:
	RUSTFLAGS="-C target-cpu=native" $(CARGO) run --features tracy --example profile_engine

# Run everything
optimize: build-native bench profile

clean:
	$(CARGO) clean
	rm -rf $(VENV) dist/
