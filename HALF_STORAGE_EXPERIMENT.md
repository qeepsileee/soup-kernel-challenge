# Frozen FP16 storage candidate

Update 2026-10-31: full public-task validation passed at dee0170 with
HEAD_CHUNK=128, PRECOMPUTE_HEAD=1, HALF_STORAGE=1 (all prefixed
SOUP_EXPERIMENT_). Mean 67.66s / peak 4269MiB, all ten checks and 79 Soup
regressions passed. Eval loss 2.04967 -> 2.04944 (-0.011%). See
VALIDATED_RESULTS.md for exact runs and cross-session timing limits.
The candidate remains opt-in; original implementation notes follow.

Verified fallback: commit `2f7e59cd359d23245b34fd3244b3286f290010d3`,
71.8 s / 5363 MiB, all ten checks passed on the public task.

The supplied timing breakdown has mean training loop 57.79 s for base and
50.18 s for solution, versus 21.40 / 21.59 s outside the loop. First two steps
cost 1.90 / 5.74 s. The 7.42 s apply measurement overlaps the outside-loop
interval; do not add it again to total runtime. Typical step medians are
0.9168 / 0.7140 s (about 22% less time), not a whole-run speedup claim.

`SOUP_EXPERIMENT_HALF_STORAGE=1` enables a guarded post-model-setup hook.
Only frozen ordinary FP32 Parameters exactly equal to their FP16 round-trip
are converted. Parameter identity and tied-weight aliases survive; trainable
LoRA Parameters, Params4bit subclasses and all buffers are untouched.
The MLP bridge accepts FP16 inputs so INT8 saved-activation compression remains
active. The default without this environment variable keeps the verified path.

Exact parameter values do **not** imply exact forward/backward: the residual,
normalization and positional-embedding arithmetic can change dtype and rounding.
The full loss curves, metrics, adapter checks, own-kernel share and evaluation
must pass again before enabling this by default. No speed/memory result yet.

On the pinned Kaggle T4 environment, in fresh subprocesses:

```bash
python -m pytest solution -q
export SOUP_EXPERIMENT_HALF_STORAGE=1
python task/prepare_data.py
python harness/run_task.py --repeats 2 --out runs_half_storage
python harness/check_rules.py --result runs_half_storage/result.json
```

The training log must contain `[half_storage]` with nonzero `saved_bytes`.
If the tests fail, stop before the full run. Compilation, model conversion and
temporary allocations remain inside the ordinary harness measurement. Do not
run diagnostics in the notebook GPU process while measuring.

Local validation: Python AST and whitespace checks pass. PyTorch/CUDA tests
cannot run on this development environment and were explicitly skipped.
