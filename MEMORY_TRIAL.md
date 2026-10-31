# Memory trial branch — full-task validation pending

This branch intentionally differs from intoxicated/nf4-lora. Do not merge or
submit it before the full official checks pass. The validated fallback is
85b03db58bf420b39be9759a2b892807d2c37372, 64.61s / 4315MiB.
No 40-second result is being claimed or predicted here.

Default apply() on THIS branch combines:

* All validated head/half-storage/NF4/RMSNorm/LoRA defaults, including K/V weights.
* Exact saved-data K/V compact snapshots and bool dropout-mask bitpacking.
* Approximate 4-bit saved MLP activations in place of the validated INT8 codec.

No env variables are required. Task flags, model weights, trainable adapter
parameters, optimizer, samples, masks, step count and native SDPA remain fixed.
Four-bit MLP forward uses the original gate/up values; only saved data for
backward is quantized. Its component gradient discrepancy is ~9--14%; this is
not a model-quality tolerance. Full eval and loss-curve admission is mandatory.
The first FP16 MLP batch is compressed too: already-configured FP16 frozen
projections initialize compute_type_is_set before can_bridge checks. This avoids
an initial uncompressed activation peak, and must be quality-validated as part
of this branch. Unsupported modules/dtypes keep the original fallback.

Component evidence: lossless codecs passed 5 tests. Attention live allocation
fell 10MiB/layer; dropout mask 2.62MiB/call. They added .093/.232ms respectively
in the supplied single-component measurements. NF4 saved MLP tests passed;
live allocation fell 17.5MiB/layer but MLP wall time rose ~2.2% at 2048 rows.

Approximate sum of saved-data opportunities at full length: ~280MiB KV,
~142MiB bool masks, ~490MiB MLP. Subtracting that sum from 4315 gives ~3400MiB,
but it is NOT a prediction of nvidia-smi peak: temporaries, first-batch behavior,
allocation lifetimes, native workspace and polling can change the actual result.
Runtime may increase. This branch tests the memory side of the time/memory
tradeoff; retain the faster fallback separately. Full-run measurements decide.

On the pinned Kaggle T4 environment, without experiment variables:

```bash
python -u -m pytest solution -q
python -u task/prepare_data.py
python -u harness/run_task.py --repeats 2 --out runs_memory_trial
python -u harness/check_rules.py --result runs_memory_trial/result.json
```

Accept only if all ten checks pass, including eval <=+0.5%, loss curve limits,
identical task/outputs within tolerance, >=5% original kernels and 79 Soup
regression tests. Derived example_* MLP kernels remain excluded. This file is
experiment documentation, not final README/SUBMISSION or proof of admission.
