# Paired frozen MLP candidate — standalone, not enabled by apply()

Current production configuration remains chunk=128/precompute/half storage,
with INT8 saved MLP activations. Its historical full-run result at dee0170 was
67.66s / 4269MiB and all ten checks passed. The no-env code pin 8c1a8db still
needs a complete default-only validation; do not transfer historical timings
to a new pin without running the official harness.

The current profile puts about 54% of CUDA time in frozen projection GEMM
ranges. Retaining decoded weights and changing dX layout were insufficient:
retaining large weights yielded only ~3% at high memory cost, while transposed
dX layout regressed and stays rejected. Attention is not the next target.

## Structural change

For a supported bias-free frozen MLP, both gate and up receive the same input.
The candidate writes their exact NF4 dequantizations into two slices of a
caller-owned [2H,D] temporary, avoiding a concatenate copy. One cuBLAS GEMM
emits [rows,2H]. A strided compressed-SwiGLU kernel directly reads its halves
without making separate contiguous gate/up copies. It retains the same group-64
INT8 gate/up and FP32 scales as the production compressor.

Backward computes dHidden through the down projection as usual. Its SwiGLU
kernel writes dGate/dUp directly into [rows,2H]. A single cuBLAS GEMM computes
the sum of their input gradients. This removes one forward GEMM, one backward
GEMM and the separate input-gradient addition per eligible layer. Dense weights
are recomputed normally for backward; none are cached across layers or steps.
LoRA projections, dropout, attention, optimizer, batches, masks and steps stay
outside this experiment. Supported input is FP16, >=512 token rows; unsupported
patch calls fall back to the existing MLP. No graph replay, torch.compile or
third-party kernel library is introduced.

## Correctness and admission limits

Combining GEMMs can select a different cuBLAS algorithm. One backward GEMM
accumulates both branches before rounding, unlike summing two rounded FP16
results. Therefore this candidate is not claimed bitwise identical. It also
inherits the already-used INT8 backward approximation. Forward/gradient tests
bound discrepancies; full loss curves, adapter/output checks and eval loss must
pass again before adoption. RNG is not consumed by this MLP.

SwiGLU formulas derive from the supplied example. Their example_-prefixed
kernels are explicitly excluded from original-kernel credit. The NF4 decoder
is reused, with RMSNorm/LoRA/CE original kernels unchanged. The >=5% check
must still be measured for any eventual integrated full run.

The paired dense buffer is larger than either individual matrix. Temporary
peak memory may increase even though launches/GEMMs decrease. Do not claim
both time and memory improve until measured. The benchmark prints complete
MLP forward/backward wall time in ABBA order, incremental allocated peak and
live allocations after forward. These are diagnostics, not whole-run scores.

```bash
python -m pytest solution/test_paired_mlp.py -q
python -u -m solution.bench_paired_mlp --tokens 2048
python -u -m solution.bench_paired_mlp --tokens 512
```

There is no env flag that enables this in production. An explicit
patch_paired_mlp() exists only for future diagnostic integration after the
microbenchmarks pass. No full-run speed/quality or CUDA compilation result is
available yet. Local verification: AST, CLI-independent checks and existing
CPU format tests; CUDA tests skip on this development environment.

First T4 test at 16c034c stopped at input validation before launching the paired
kernel: the fixture's torch.randn weights defaulted to FP32, so quant_state.dtype
was FP32 even though Linear4bit.compute_dtype was FP16. The fixture now explicitly
quantizes FP16 source weights, matching the intended T4 model route; the kernel's
FP16 restriction is retained. Tests assert quant-state dtype before numerical
comparison. Benchmark uses the same corrected fixture. No CUDA correctness or
performance result for the paired kernels has been established yet.

## First corrected T4 measurement (458144c)

Both candidate tests passed (2 passed, no skips). Results supplied by the user:

| Token rows | Baseline fwd+bwd | Paired fwd+bwd | Speedup | Incremental allocated peak base/paired | Live after fwd base/paired |
|---|---:|---:|---:|---:|---:|
| 2048 | 16.135ms | 16.025ms | 1.007x | 178.7 / 173.0MiB | 44.5 / 44.5MiB |
| 512 | 4.908ms | 4.773ms | 1.028x | 63.3 / 82.3MiB | 10.8 / 10.8MiB |

Paired ratios at 2048: 1.083, 1.006, 1.011, 0.996; at 512: 1.072, 1.053,
1.024, 1.033. This is a small/near-noise benefit for large matrices, a modest
benefit for smaller matrices, and no saved-activation-memory reduction. The
smaller case's allocated peak grows by 19MiB. Do not enable by default or claim
a meaningful whole-model speedup. No full-task quality/admission run exists.
No official full training run is being requested for this candidate yet.

Two outstanding pieces of stronger evidence should be preserved/recovered
before additional speculative microbenchmarks: the completed runs_retained_kv
result/checks, and no-env full validation of the fixed-default configuration.
