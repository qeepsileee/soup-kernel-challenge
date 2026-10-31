# Memory trial branch — public-task validation passed

This branch intentionally differs from intoxicated/nf4-lora. All ten official
automatic checks passed on the supplied public-task run. The INT8 fallback is
85b03db58bf420b39be9759a2b892807d2c37372, 64.61s / 4315MiB.
No 40-second result is being claimed or predicted here.

## Full-task evidence, 2026-10-31

Tested code: c6f6096a58c29ef319e6d7ad40cfac01a561d369. Source: user's
Kaggle console output, runs_memory_trial; raw result.json/checks.json and
the saved notebook have not been independently downloaded into this repository.
All SOUP_EXPERIMENT_* variables were unset. All 66 solution tests passed.

| Run | Full command (s) | GPU peak (MiB) |
|---|---:|---:|
| base_1 | 89.67 | 14345 |
| solution_1 | 68.50 | 3279 |
| solution_2 | 71.17 | 3279 |
| base_2 | 86.65 | 14345 |

Mean time from printed per-run values: 88.16s / 69.835s (displayed 69.8s).
Reported speedup 1.262x; memory ratio 4.375x, saving 11066MiB / 77.1%.
Against the earlier 4315MiB configuration, the peak is 1036MiB lower (24.0%).
Timing across sessions is not a controlled ablation: the paired base also
changed, so this result alone does not establish codec overhead or speed gain.

Eval loss 2.04966 -> 2.05022, reported +0.027%, within the +0.5% limit.
The loss curve differs 0.04% at the first step and 0.06% on average.
Quality is not bit-identical or improved; public admission tolerances passed.
Own-kernel coverage is 8.6%, lowest of two runs, with example-derived MLP
kernels excluded. Clean T4 measurement, task/output checks and all 79 Soup
regression tests passed. Hidden rows/seed and manual review remain unverified.

Preserve the exact tested pin and runs_memory_trial outputs. Documentation-only
commits do not replace the tested pin. The 2500--3500MiB target range is reached
on this run; the 40-second whole-command target remains unmet.

## Implementation and reproduction

Default apply() on THIS branch combines:

* All validated head/half-storage/NF4/RMSNorm/LoRA defaults, including K/V weights.
* Exact saved-data K/V compact snapshots and bool dropout-mask bitpacking.
* Approximate 4-bit saved MLP activations in place of the validated INT8 codec.

No env variables are required. Task flags, model weights, trainable adapter
parameters, optimizer, samples, masks, step count and native SDPA remain fixed.
Four-bit MLP forward uses the original gate/up values; only saved data for
backward is quantized. Its component gradient discrepancy is ~9--14%; this is
not a model-quality tolerance. Full eval and loss-curve admission passed above.
The first FP16 MLP batch is compressed too: already-configured FP16 frozen
projections initialize compute_type_is_set before can_bridge checks. This avoids
an initial uncompressed activation peak, and was included in the validation
above. Unsupported modules/dtypes keep the original fallback.

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

Future changes must again pass all ten checks, including eval <=+0.5%, loss curve limits,
identical task/outputs within tolerance, >=5% original kernels and 79 Soup
regression tests. Derived example_* MLP kernels remain excluded. This file is
experiment documentation, not final README/SUBMISSION or proof of hidden-data admission.
