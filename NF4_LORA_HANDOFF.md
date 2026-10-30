# Participant 3: NF4 and ввсеLoRA

CURRENT STATUS: apply() enables ONLY the validated NF4 decoder plus the starter
SwiGLU. Both custom dX GEMM experiments are DISABLED in training because they
regressed in the user's T4 measurements. The dense candidate passed 15 tests,
but q/o were ~3.5x slower and gate/up/down ~3.9-6.5x slower than baseline.
The experimental kernels remain callable for reproducing the measurements.

Decoder-only own-kernel coverage was 1.7% in the initial full run, so this module
alone does NOT meet the 5% admission gate. Team Loss/Fusion kernels are not yet
available. Final admission requires real useful work totaling >=5%, plus a clean
repeat measurement. Do not claim admission or a verified NF4-only memory gain.

Next: benchmark decoder-only with the command below, including 512 and 2048
tokens. Then perform a clean full repeated measurement, in a separate output
directory to preserve old results. Adding team kernels is still required if
coverage stays below 5%.

```bash
python -m solution.bench_nf4 --tokens 512 --iterations 100 --summary
python -m solution.bench_nf4 --tokens 2048 --iterations 100 --summary
python harness/run_task.py --repeats 2 --out runs_decoder_clean
python harness/check_rules.py --help
```

The first full run passed 79 Soup tests, quality and output comparisons, but
failed coverage and idle-GPU checks. Local machine has no CUDA.
This is not a final hackathon submission.

## Rejected candidate: separate decoding and dense GEMM

The user's 512-token microbenchmark at 5821ca4 showed inline-decode dX caused
severe regressions: gate 1.044ms -> 24.183ms, up 1.030ms -> 24.514ms,
down 1.087ms -> 23.491ms. Standalone decode was 1.11-1.81x faster, error 0.
Avoid a full training run with that inline-decode candidate.

When explicitly called, patch_input_grad() selects input_grad_dense(): decode W once with
our decoder, then call our tiled dense GEMM, accumulating in FP32 with FP16
inputs/output. Dense W is a temporary of this backward call; no persistent cache.
This reintroduces the one-weight temporary but avoids redundant NF4 unpacking
inside every GEMM tile/reduction. No performance or >=5% coverage claim yet.
Experimental fused=True is only for reproduction of the rejected experiment.

```bash
python -m pytest solution -q
python -m solution.bench_nf4 --dense-dx --tokens 512 --iterations 20 --summary
```

Measured at fb49813: q 0.718 -> 2.475ms, k 0.702 -> 0.689ms,
v 0.662 -> 0.739ms, o 0.681 -> 2.472ms, gate 2.061 -> 8.045ms,
up 1.263 -> 8.190ms, down 1.301 -> 8.073ms. This candidate is rejected.

## History of earlier candidates

UPDATE after user's first Kaggle run (2026-10-30): decoder candidate at 02fb461
passed 12 GPU tests and 79 Soup tests. Full exploratory run: 117.8s/14449MiB
base, 85.15s/12385MiB solution, eval losses 2.04968/2.04969. NOT an admitted
result: own-kernel share is 1.7% (<5%); GPUs were not idle at baseline start
(14345MiB). Time/memory also include the starter SwiGLU contribution. Do not
attribute the total improvement to NF4 or treat these as clean official numbers.

Second candidate adds nf4_matmul.py: tiled FP16 dY @ NF4-W with inline nested
decode, so backward need not allocate full dense W. apply() enabled it at 5821ca4;
CURRENT apply() does not enable it. Frozen base weight/bias only; bnb version 0.50.2 only;
unsupported paths use original backward. Forward and PEFT dropout stay unchanged.
This candidate is NOT CUDA-tested yet. CPU checks do not compile Triton.

Validate before full measurement:

```bash
python -m pytest solution -q
python -m solution.bench_nf4 --fused-dx --tokens 512 --iterations 20 > runs/nf4_dx_512.json
```

Review forward_backward_speedup for regressions. Kernel contribution must come
from useful dX computation; do not pad kernel time or repeat unused computations.
Before an official run, stop all other GPU workloads, restart notebook session
if its kernel retains CUDA allocations, and inspect nvidia-smi. Both GPUs must be
idle according to the harness. Never kill arbitrary processes or alter harness
allocator/timing settings. Then run --repeats 2 and check_rules.py.

## Changes to the proposed work plan

1. Inspect the INSTALLED bitsandbytes version on Kaggle before replacing a path.
   In 0.50.2, MatMul4Bit.forward calls gemm_4bit; the CUDA dispatcher chooses
   custom GEMM or dequantization + linear depending on hardware and geometry.
   MatMul4Bit.backward reconstructs W for dX. It does not save a dense W in ctx.
   Thus removing a supposed persistent dense W per layer is not an established
   saving. Measure lifetimes and allocation peaks first.
2. First deliverable: independently written Triton NF4 decoder, including fused
   nested-scale decode. Candidate patch changes functional dequantize_4bit only,
   primarily used by backward. Direct torch.ops calls in forward are unaffected.
3. Keep the existing PEFT LoRA forward and dropout unchanged in this milestone.
4. A global output scratch buffer must NOT be returned by a generic dequant API:
   callers can retain it for autograd or ordinary use. If implementing a custom
   linear later, control the lifetime within forward/backward and synchronize
   reuse across streams. Do not save scratch in ctx.
5. Decode + torch matmul still materializes ONE dense W. Only fused tiled
   dequant-matmul eliminates that full temporary. Scratch reuse alone does not.
6. No attention implementation in this workstream. It expands scope and requires
   its own causal/GQA/masking/backward validation for a relatively small hotspot.

## Integration contract

All executable files are under solution/. The team's integrator should preserve
their combined apply() and ADD this call, rather than replace their __init__.py:

```python
from .nf4 import patch_dequantize
patch_dequantize()
```

The patch is idempotent; unsupported dtype, CPU, non-NF4 and unsupported nested
formats fall back to bitsandbytes. No model configuration, parameters, LoRA,
dropout, optimizer, dataset or metrics are changed. A fresh output is allocated
for each public dequant call (unless caller supplies out). The candidate preserves
the bnb legacy transposed packed-input return convention. Kernels launch normally
without CUDA graphs, hidden inputs or measurement-dependent behavior.

## NF4 data format

Canonical source weight is [out_features, in_features], flattened row-major.
One byte contains two indices: high nibble first, low nibble second. Each index
selects one of 16 NF4 values; its scale is selected by flat_index // blocksize.
Use quant_state.shape, NOT packed.shape, for dense geometry.

Nested statistics: quant_state.absmax is uint8, not a float scale. Decode it via
state2.code[absmax[b]] * state2.absmax[b // state2.blocksize], then add offset,
in FP32. Multiply NF4 value by that scale and round to quant_state.dtype.
The Triton launch disables multiply-add fusion to preserve separate rounding.
No persistent decoded scales or weights are cached.

Expected Qwen shapes [out, in], to verify on the actual model:

| Projection | Shape |
|---|---|
| q_proj / o_proj | 1536 x 1536 |
| k_proj / v_proj | 256 x 1536 |
| gate_proj / up_proj | 8960 x 1536 |
| down_proj | 1536 x 8960 |

LoRA is only on q_proj/v_proj for this task. Its effective scaling is alpha/r=2.
The actual branch is B(A(dropout(x))), NOT B(A(x)). Keep the original dropout
call/order/RNG consumption. Preserve autocast and dtype rounding boundaries.
Do not train base W. Incoming backward gradients already include loss scaling;
do not divide or multiply by the scaler a second time.

## Kaggle validation sequence

Use the official notebook's pinned environment; do not install packages during
the measured process. Copy solution files into the platform-created team repo.

```bash
python -m unittest solution.test_nf4 -v
mkdir -p runs
python -m solution.bench_nf4 --tokens 512 --iterations 100 > runs/nf4_micro_512.json
python -m solution.bench_nf4 --tokens 2048 --iterations 100 > runs/nf4_micro_2048.json
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
```

CPU-only tests are explicitly skipped for CUDA cases: a CPU pass is NOT a GPU
pass. GPU tests compare all projection geometries, nested/plain scales, FP16/FP32,
all 16 codes, odd tails, supported block sizes, out=, transposed input, FP4
fallback, scaled dX and a two-layer backward. Microbenchmarks compare preallocated
decode and actual bnb forward+backward. They exclude JIT warmup and do not replace
official cold-cache full-command timing or nvidia-smi peak measurements.

Acceptance: all GPU tests pass, complete training passes official checks, and
the combined solution satisfies >=5% useful own-kernel time in EVERY measured
run. A decoder alone may not reach 5%; measure team-wide coverage. Keep the patch
only if the combined measured outcome improves. Benchmark full load/train/save
memory before claiming a memory reduction.

## Next milestones after CUDA feedback

- Record actual quant states and profiler evidence; fix correctness before tuning.
- Evaluate decode bandwidth and full-step benefit; reduce repeated scale loads if
  profiling justifies it. Avoid large autotuning searches: compilation counts.
- Consider controlled scratch in a custom linear backward only after evidence
  that allocation/lifetime is a bottleneck.
- Try LoRA second GEMM plus scaling/add epilogue with existing dropout. Validate
  gradients of x, A and B at actual FP32/FP16/autocast boundaries against PEFT.
- Fused NF4 matmul is a stretch goal only if decoder/integration are stable.

## Sources (format/API/ideas only; no foreign kernel code copied)

- QLoRA, Dettmers et al., 2023: https://arxiv.org/abs/2305.14314
- bnb 0.50.2 autograd and ctx lifetime:
  https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py
- bnb 0.50.2 nested quant state and dequant API:
  https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py
- bnb 0.50.2 CUDA dispatcher:
  https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py
- PEFT 0.20.0 LoRA behavior:
  https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py
- Model dimensions:
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
