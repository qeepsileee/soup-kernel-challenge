# Runtime trial: fa12d01 validated; RoPE broadcast follow-up pending

Branch: intoxicated/runtime-trial. Parent functional pin:
c6f6096a58c29ef319e6d7ad40cfac01a561d369, validated at 69.835s / 3279MiB
with all ten public-task checks passed. Preserve that pin and its Kaggle outputs.
The INT8 fallback remains 85b03db58bf420b39be9759a2b892807d2c37372.
Tested pin: fa12d016036918faf37ef30a7a86f23e089b6be3, 67.44s / 3119MiB.
All 75 solution tests, all ten official checks and 79 Soup regressions passed
on the user's supplied Kaggle run. Eval loss +0.032% is within the +0.5% limit;
the first-step/mean loss deviations are 0.04% / 0.06%. Own-kernel share is 9.8%.
The paired base was 84.51s / 14345MiB, speedup 1.253x, memory ratio 4.599x.
See VALIDATED_RESULTS.md; raw official JSON is still on Kaggle, not downloaded here.

The peak is 160MiB below the parent. Cross-session absolute time is not an
isolated causal estimate because the paired base also changed. The component
shared-q/v path saves memory but regressed in time; retain that tradeoff explicitly.
RoPE was not launched in every training run, despite its faster microbenchmark:
the original strict shape guard rejected batch=1 positional embedding broadcast.
Its component speedup is not evidence of an active whole-training RoPE speedup.

## Follow-up: batch-broadcast RoPE, not yet CUDA/full-task validated

Qwen2Model's default position_ids are [1,T], so the rotary module returns
cos/sin [1,T,128] shared across B examples. The previous guard required exactly
[B,T,128] and therefore rejects that normal default model path. This is a
specific dispatch gap consistent with the reported lack of launches; the
prior console output does not expose each actual call's metadata.

The follow-up accepts independent batch dimensions 1 or B for cos and sin,
uses zero batch stride for shared tensors without expanding/materializing them,
and otherwise preserves the validated math, layout guards and native fallback.
New CUDA tests compare exact forward/backward for broadcast and mixed broadcast,
and run a tiny real Qwen2Model with default position_ids, asserting own dispatch.
The component benchmark now uses cos_batch=1 and verifies its geometry is eligible.
These follow-up GPU tests are skipped locally on the Mac, not reported passed.

After the unchanged official checks, the standalone launcher also audits raw
per-run triton_launches for rotary_forward_kernel and rotary_backward_kernel.
This additional experiment check reads artifacts only and changes no official
measurement or checks. It prevents crediting a microbenchmark optimization that
did not actually run in every measured solution. Preserve fa12d01 as the current
validated 67.44s / 3119MiB fallback until the new pin passes the complete run.

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
* The exact half-storage check compares FP32 parameters directly with the FP16
  candidate using torch.equal. CUDA compares them in the common FP32 dtype
  without materializing a full FP32 copy of the half candidate. This preserves
  the acceptance/rejection test, including NaNs, overflow and underflow, while
  removing an avoidable large setup temporary. No new kernel compilation is
  required for this change. A GPU allocation test checks it against the old path.

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
forward/backward with compact-KV hooks composed with the new hooks. Half-storage
tests compare exact values and allocation peaks against the explicit FP32 copy.

On the pinned Kaggle T4 environment, with no SOUP_EXPERIMENT_* variables:

```bash
bash tools/run_kaggle_runtime_trial.sh
```

The standalone script checks CUDA/T4, restores missing or mismatched pinned
training dependencies before pytest, and verifies actual NF4/PEFT imports.
It skips pip if all pinned distributions are already present. Package setup
finishes before any measured harness child starts; no packages are installed
by solution.apply() or during training. Experiment variables are unset.
Existing output directories are preserved by choosing a timestamped output name.

The first supplied d481ed6 run used a fresh Kaggle session without bitsandbytes:
40 tests passed, 31 were skipped and four integration tests stopped at the
missing import. The full harness was not reached. These failures do not establish
a numerical failure or full-task success. The subsequent fa12d01 launcher
restored the environment and all 75 tests/full checks passed as recorded above.

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
* https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/cuda/Equal.cpp
* https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/TensorIterator.cpp
