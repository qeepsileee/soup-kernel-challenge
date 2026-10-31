# Runtime trial: CUDA and full-task validation pending

Branch: intoxicated/runtime-trial. Parent functional pin:
c6f6096a58c29ef319e6d7ad40cfac01a561d369, validated at 69.835s / 3279MiB
with all ten public-task checks passed. Preserve that pin and its Kaggle outputs.
The INT8 fallback remains 85b03db58bf420b39be9759a2b892807d2c37372.
This branch has not established a faster or smaller official result yet.

## Changes

* Keep all validated memory-branch defaults, including NF4 saved MLP activations,
  compact SDPA K/V, chunk-128 head statistics/gradient and exact half storage.
* Native PyTorch dropout forward still generates both values and bool masks.
  Its RNG calls, probability, shape and dtype are preserved. The masks are
  packed, then backward reads those bits and scales the gradient in one own
  Triton kernel. The old unpack-to-bool allocation/kernel is removed on this path.
* For FP16 LoRA inputs inside one Qwen attention call, retain a single immutable
  snapshot shared by q/v rather than both large dropped-out inputs saved for A.
  Save separate masks and reconstruct each actual A input using native forward
  rounding. Fuse the first snapshot copy with bool-mask packing. Snapshot values
  survive later source mutation; guards reject stale pointers/changed outputs.
  Source caches are scoped to the forward call and cleared on exit.
* Own RoPE forward/backward kernels fuse multiplication, rotation and addition.
  Preserve the separate FP16 product rounding; positional embeddings and the
  original native attention backend stay unchanged. Query and key activation
  values are not needed for this derivative, so only cos/sin are saved.
  Unsupported geometry, dtype or differentiable cos/sin use HF's original path.

The shared-input scheme can remove roughly one [4,512,1536] FP16 saved tensor
per layer (about 6MiB at full length); this is an allocation argument, NOT an
official peak prediction. Actual peaks include first steps, setup, temporaries
and workspace. Added Python hooks, reconstruction and cold compilation can
offset kernel savings, so the microbenchmark and full harness decide.
There is no claimed 40-second or sub-3000MiB result.

Model weights, adapter ranks/scales, optimizer, samples/order, masks and step
count are unchanged. These changes target first-order training; they do not
add any further gradient quantization beyond the parent's validated NF4 codec.
Starter-derived example_* kernels remain excluded from own-kernel coverage.

## Validation

Locally: Python AST validation, existing CPU accounting/summary tests and native
dropout-scale arithmetic checks. This Mac has no torch/triton/CUDA; the new GPU
tests are skipped locally, not reported as passes. GPU tests cover strided/tail
RoPE and scaled gradients, RNG, exact dropped-input reconstruction, real PEFT
q/v x/A/B gradients, snapshot sharing, fused pack tails and a real Qwen attention
forward/backward with compact-KV hooks composed with the new hooks.

On the pinned Kaggle T4 environment, with no SOUP_EXPERIMENT_* variables:

```bash
python -u -m pytest solution/test_runtime_trial.py -q
python -u -m solution.bench_runtime_trial --tokens 512
python -u -m solution.bench_runtime_trial --tokens 128
python -u -m pytest solution -q
python -u task/prepare_data.py
python -u harness/run_task.py --repeats 2 --out runs_runtime_trial
python -u harness/check_rules.py --result runs_runtime_trial/result.json
```

The diagnostic benchmark compares against the parent's packed-mask dropout and
native RoPE, not against an unoptimized solution. It alternates A/B order with
warmup outside timings and reports local live/peak allocations as diagnostics.
Do not treat those numbers as whole-model nvidia-smi memory or scores.
Only accept a new pin after all ten full checks, including >=5% own-kernel
coverage, unchanged task/outputs, quality/curve tolerances and 79 Soup tests.
No final README/SUBMISSION work or merge into either validated branch is done.

Format/API references only; Triton implementations written in this repository:

* https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/Dropout.cpp
* https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu
* https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
