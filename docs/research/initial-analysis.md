# mlx-mamba3 Initial Research Notes

Date: 2026-07-07

## Snapshot

- Upstream: https://github.com/Jada42/mlx-mamba3
- Local fork: https://github.com/ShinKwangsub/mlx-mamba3
- Upstream baseline: `c4e50f5d5f04630fe474c24db28b5387e9424557`
- Package name: `mlx-mamba-native`
- Scope: native MLX implementation of Mamba-3 concepts for Apple Silicon.

## What It Implements

- SISO and MIMO Mamba-3 sequence mixers.
- Exponential-trapezoidal recurrence with `Bx_prev` cache state.
- RoPE-style rotating state projections for B/C.
- Hybrid Mamba + causal attention layers.
- Prefill, chunked prefill, and token-by-token decode cache paths.
- Basic generation helper.
- Safetensors save/load utilities.
- LoRA wrappers for Mamba mixer `in_proj` and `out_proj`.

## Verified So Far

- GitHub Actions latest visible run passed on macOS.
- Local non-Torch tests passed:
  - `tests.test_step_eq_prefill`
  - `tests.test_hybrid`
  - `tests.test_cache_chunking`
- Local toy examples passed with `PYTHONPATH=.`:
  - hybrid generation
  - LoRA fine-tuning demo
  - serialization demo when `safetensors` is installed
  - benchmark script

## Early Concerns

- Repository is very young: created 2026-06-30, no tags, no releases, single contributor.
- Numerical parity is against the bundled readable PyTorch reference, not necessarily a direct official CUDA/Triton checkpoint path.
- `convert_to_lora()` assumes every layer has `layer.mixer`; it fails on hybrid models containing `AttentionBlock`.
- README says inference compilation via `mx.compile`, but the only explicit `mx.compile` call is for the training step.
- Examples require package installation or `PYTHONPATH=.` when run directly from the repository root.
- Real pretrained checkpoint conversion/loading is not established yet.

## Research Questions

1. How close is this implementation to the official `state-spaces/mamba` Mamba-3 implementation at the config and weight layout level?
2. Can we make a reliable checkpoint conversion path from a real Mamba-3 checkpoint into MLX safetensors?
3. Where is the performance bottleneck on Apple Silicon: scan, RoPE, projection, attention cache, or Python loop overhead?
4. Can LoRA support hybrid models by selectively wrapping Mamba blocks and optionally attention projections?
5. What is the smallest useful benchmark suite for SISO vs MIMO vs hybrid on M1/M2/M3/M4?

## Suggested Next Experiments

- Run the full upstream test suite inside a clean `.venv` with `torch` installed.
- Add a regression test for `convert_to_lora()` on hybrid configs.
- Compare MLX model parameter names/shapes with the bundled PyTorch reference.
- Profile `examples/benchmark.py` with a few sequence lengths and MIMO ranks.
- Build a small notebook or script for architecture diagrams and tensor-shape tracing.
