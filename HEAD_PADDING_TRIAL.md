# Head-padding trial — T4/full-task validation pending

Branch intoxicated/head-padding starts from the validated
fa12d016036918faf37ef30a7a86f23e089b6be3 (67.44s / 3119MiB, all ten checks,
75 solution tests and 79 Soup regressions passed). This is a new unvalidated
candidate, not a claimed speed or memory record. Preserve the parent pin.
Final README/SUBMISSION work remains deferred at the user's request.

## Evidence behind the change

The later RoPE broadcast pin 96fd073 passed all checks and actual dispatch
but measured 72.75s / 3141MiB, versus the parent's 67.44s / 3119MiB.
The standalone fresh-cache RoPE diagnostic supplied by the user showed one
forward/backward variant at T=512 (first use 2.3020s), a second pair at T=430
(0.1920s), then no new variants for T=406..235 or repeat T=512 (about .0006s).
This disproves compilation on every length in that diagnostic. Those timings
include first-use/runtime overhead and are not pure compiler times or scores.
There is no reason to treat a large variant explosion as established evidence.

This branch isolates another optimization on the best parent. It does not
integrate the new RoPE dispatch or teammates' residual-RMSNorm/SwiGLU commits.

## What changes

The precomputed frozen-head training path previously projected ALL [B,T]
positions to the vocabulary for entropy. The actual trainer masks entropy by
the original two-dimensional attention_mask, including prompt tokens but
excluding padding. Loss, accuracy and Rewind use shifted labelled targets.

Now the head keeps the union of:

* EVERY non--100 shifted target, regardless of its attention_mask value;
* EVERY nonzero attention_mask position, including ignored prompt targets.

Only (attention_mask=0 AND shifted target=-100) is elided from head GEMMs and
CE statistics. Transformer forward/backward, attention inputs/masks, data and
token order remain unchanged. No labelled row can be lost at a left-padding
boundary or internal masked gap. Skipped token_loss/hit entries are zero;
unconsumed entropy entries are zero. All consumed entropy, counts, accuracy,
row losses and loss gradients must match within the existing numerical limits.

No mask means the original full-position behavior. The no-grad/evaluation and
exact recompute path is unchanged. Unsupported hook mask layouts use the full
path. No head or adapter weight, rank, probability, optimizer, step count,
batch size, sequence length, loss denominator or loss scale is changed. No new
gradient quantization or additional Triton kernel compilation is introduced.
Changing GEMM row counts can change FP16 rounding, so full quality is mandatory.

A single host-only message per training process records projected/total/labelled
row counts already computed by nonzero(). No extra GPU sync, delayed work or
measurement changes are used. The standalone launcher reads each train.log after
official checks to confirm this path ran, without modifying official artifacts.

## Validation

GPU tests cover right/left padding and internal gaps, ignored prompt entropy,
labelled targets at masked positions, all-padding and single-label cases,
actual projected row counts, real vocabulary/AMP scales through 65536,
TRL entropy/token/accuracy metrics and a real Qwen causal-LM hook/RNG check.
The Mac lacks CUDA/torch/triton; skipped GPU tests are not reported passed.

The component ABBA benchmark compares the SAME validated precomputed head,
chunk=128, with and without mask-aware projection. It tests no-padding and
variable-length batches on B=4,T=512,D=1536,V=151936, reporting actual projected
rows, time and local allocated peaks. Those are diagnostics, not whole-model
nvidia-smi memory, admission or scores.

On Kaggle T4, with pinned dependencies restored automatically before tests:

```bash
bash tools/run_kaggle_head_padding.sh
```

Keep a new pin only if all ten official checks pass, including unchanged
task/outputs within tolerance, loss-curve and eval limits, >=5% original kernels,
clean measurement and 79 Soup regressions. Kernel share may decline when useful
CE work is eliminated; it must still pass without artificially padding time.
No performance improvement is established until the actual full run is supplied.
