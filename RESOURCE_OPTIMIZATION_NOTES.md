# Next resource experiments — not enabled in production

Validated fallback is 85b03db58bf420b39be9759a2b892807d2c37372: 64.61s,
4315MiB, 56 solution tests, 79 Soup tests, all ten checks. Below 40s and
2500--3500MiB are goals, not achieved or promised measurements.

## 1. Unique saved activation and startup audit

`python -u -m solution.resource_audit` executes the existing training entry
point with its current solution, stopping after four steps via a diagnostic
callback. Task settings (including the 64-step scheduler) are not edited.
This instrumented short run is NOT a score or task-quality evaluation.

Python import observation installs hooks when the original training program
loads modules, rather than preloading torch/model code in the driver. It records
inclusive import wall times, model setup and trainer initialization. These
phases overlap; do not add them. YAML preparation occurs outside those phases.
Import-hook and module-hook overhead means these are diagnostic clues, not an
exact decomposition of official whole-command time.
It uses the harness's offline environment (model/tokenizer must already be
cached), and a fresh per-diagnostic Triton cache like the measured cold runs.
The first original CUDA allocation is timed without pre-initializing CUDA in
the driver. Step times are saved in the report; no warm-cache speed is claimed.

Saved-tensor hooks wrap, but do not modify/compress, saved activations. An
allocation is counted once by unique CUDA storage while any saved alias is
live. Actual model parameters/buffers are excluded using their storage pointers
on the first real forward, after half storage has been applied. Temporary cast
weights/dense decoded K/V can still be counted because they really allocate.
Group attribution uses the scope of the first save. This measures saved storage,
not total nvidia-smi memory; unsaved temporaries, CUDA workspace, model weights
and allocator reservation still contribute to the official peak. Forward
snapshots also show PyTorch allocated memory. Warnings are emitted if no forward
snapshots were obtained. Report: runs_resource_audit/resource_audit.json.

## 2. Four-bit saved MLP activation experiment

`nf4_saved_mlp.NF4SavedSwiGLU` uses the same original FP16 forward formula.
For backward only, each group of 64 gate/up values is scaled by its absmax and
encoded by the nearest member of the existing 16-value NF4 numeric table.
One byte stores the gate/up pair at the same position, plus two FP32 group
scales. It replaces 2N+8*ceil(N/64) saved bytes with N+8*ceil(N/64) bytes, about
47% less for large N. That is a codec-storage calculation, not a claim of 47%
less model memory. It alone may not reach 3500MiB. Model weights stay unchanged.

Backward is more approximate than the validated INT8 compressor. Unit tests
require bit-identical standalone forward, finite gradients, zero/tail handling
and reduced saved bytes. Their loose gradient-error sanity bound is NOT a task
quality tolerance. Only a full official evaluation can establish whether this
candidate preserves task quality; reject it if any official check fails.
The derived example_ kernels do not count as original-kernel contributions.
No patch enables it in apply(). Benchmark full MLP, not only the codec:

```bash
python -u -m pytest solution/test_nf4_saved_mlp.py -q -s
python -u -m solution.bench_paired_mlp --candidate nf4-saved --tokens 2048
python -u -m solution.bench_paired_mlp --candidate nf4-saved --tokens 512
```

## 3. Built-in cuBLASLt preference diagnostic

PyTorch 2.11 exposes preferred_blas_library("cublaslt"). It may use another
library where the preferred one does not implement an operation. This is a
preference, not proof every GEMM was executed by cuBLASLt. No precision flags,
LoRA settings or task settings change. No external kernel library is installed.
Source: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/backends/cuda/__init__.py

The existing NF4 layer benchmark now accepts --blas cublaslt, compares against
the original backend with identical weights/input/gradient, checks forward and
dX numerics before timing, and restores the original global backend on exit.
It includes dequantization and forward/backward in ABBA pairs. Unsupported
hardware/build errors are not silently turned into claimed speedups.

```bash
python -u -m solution.bench_retained_nf4 --retain 0 --dx-layout original --blas cublaslt --tokens 2048
```

All three are standalone diagnostics. Production apply() stays unchanged and
does not read env switches for them. Do not combine candidates until component
results justify the combination, then repeat the full 64-step official run and
all ten checks. Preserve the validated fallback artifacts before experiments.
