# Time-first frozen-head experiments

Verified baseline remains `2f7e59c`: mean 71.77 s / 5363 MiB on public T4.
Below 50 s requires removing at least 21.77 s (30.3% of whole runtime).
With ~21.6 s outside the loop, the loop must drop from 50.18 to <28.4 s.
Removing the first-two-step excess alone cannot reach this target.

Two **off by default** controls are implemented:

* `SOUP_EXPERIMENT_HEAD_CHUNK=128` or `256`: larger head GEMMs, fewer chunk
  launches, higher temporary logits memory. Only the implementation chunk size
  changes; the task's batch size, sequences, steps and masks stay fixed.
* `SOUP_EXPERIMENT_PRECOMPUTE_HEAD=1`: group labelled rows first and compute
  dHidden immediately from forward logits. Restore original ordering for loss,
  entropy and accuracy. Backward multiplies the compact FP32 dHidden by its real
  incoming gradient; frozen LM head logits are never recomputed. No dWeight.

The second candidate uses anchored FP16 dLogits (scale <=65536) for cuBLAS,
then saves the rescaled FP32 dHidden, about 12 MiB for [4,512,1536].
This avoids small-gradient underflow but changes rounding relative to applying
the real upstream scale before the head GEMM. It is first-order only, and must
pass full quality, loss-curve, metrics and regression checks before adoption.
Tests cover real vocabulary, AMP, scale changes, ignored rows and sparse labels.
Neither performance nor CUDA correctness has yet been established here.

Run the pinned environment on an idle Kaggle T4:

```bash
python -m pytest solution -q
python -m solution.bench_head_speed
```

The diagnostic measures head forward **and backward**, pairs base/candidate in
ABBA order, excludes warmup, and uses B=4, T=512, D=1536, V=151936, 50% prompt.
Its peak_allocated is a local PyTorch diagnostic, not the official memory score.
The baseline is chunk=64 with recomputation. Benchmark figures do not establish
whole-run speedup; a different labelled-token proportion can change results.

Only after correctness passes and the diagnostic shows repeatable benefit:

```bash
export SOUP_EXPERIMENT_HEAD_CHUNK=256  # replace with measured winning chunk
export SOUP_EXPERIMENT_PRECOMPUTE_HEAD=1  # only if measured beneficial
python harness/run_task.py --repeats 2 --out runs_head_speed
python harness/check_rules.py --result runs_head_speed/result.json
```

Test frozen half storage separately first (`HALF_STORAGE_EXPERIMENT.md`), then
combine only candidates that passed independently. Do not use notebook GPU
allocations while timing the official harness. Set all experiment flags back
to their defaults to reproduce the verified implementation.

Local checks: AST parsing and whitespace checks pass. CUDA tests are explicitly
skipped on the development Mac because its Python environment lacks PyTorch.
