# SGLang

SGLang is a high-performance serving framework for large language models and vision-language models. It features RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, and support for various model architectures.

## Project Structure

- **python/sglang**: The main Python package containing the SGLang core logic.
- **sgl-kernel**: Custom C++/CUDA kernels for optimized compute primitives.
- **sgl-router**: High-performance Rust-based router for request scheduling and load balancing.
- **benchmark**: Scripts for benchmarking performance (throughput, latency).
- **examples**: Usage examples for both frontend and backend.
- **docs**: Documentation source files.
- **test**: Integration and unit tests.

For further architectural details and implementation specifics (e.g., MoE runner modes), please refer to `arch.md`.

## Setup & Installation

### Python Package
To install the main package in editable mode:
```bash
pip install -e python
```

### SGL Kernel
To build the custom kernels:
```bash
cd sgl-kernel
make build
# Or to install:
pip install .
```
*Note: Changes to kernels require rebuilding and potentially reinstalling.*

### SGL Router
To build the Rust router:
```bash
cd sgl-router
cargo build --release
```
*Note: Python bindings for the router are in `sgl-router/bindings/python` and can be built with `maturin develop`.*

## Development Workflow

### Code Formatting
The project uses `isort` and `black` for Python formatting.
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
Or use the Makefile target:
```bash
make format
```

### Running Tests
SGLang uses `unittest` for the main package.

**Backend Runtime Tests (`test/srt`):**
```bash
cd test/srt
# Run a specific test file
python3 test_srt_endpoint.py
# Run a specific test case
python3 test_srt_endpoint.py TestSRTEndpoint.test_simple_decode
# Run the per-commit suite
python3 run_suite.py --suite per-commit
```

**Frontend Language Tests (`test/lang`):**
```bash
cd test/lang
python3 test_choices.py
```

**Kernel Tests:**
```bash
pytest sgl-kernel/tests
```

### Router Tests
```bash
cd sgl-router
cargo test
```

## Contribution Guidelines
- **Branching**: Create feature branches (e.g., `feature/my-new-feature`) and open PRs against `main`.
- **Testing**: Add unit tests for new features or bug fixes. Ensure tests pass locally before pushing.
- **Kernels**: Updating kernels involves a 3-step process:
    1. PR to update `sgl-kernel` source.
    2. Bump `sgl-kernel` version (triggers release).
    3. Update `python/pyproject.toml` in the main package to use the new kernel version.
- **Efficiency**: Optimize for runtime performance (minimize CPU-GPU sync, use vectorized code).

## Useful Commands
- `make help`: Show available Makefile targets.
- `make update <version>`: Update version numbers across project files.
