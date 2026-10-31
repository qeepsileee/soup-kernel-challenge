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

## First user-supplied T4 diagnostic

| Sequence chunk | Recompute | Precompute | Paired precompute speedup |
|---|---:|---:|---|
| 64 | 116.42 ms | 89.07 ms | 1.330, 1.343, 1.336, 1.339 |
| 128 | 109.58 ms | 81.18 ms | 1.494, 1.498, 1.494, 1.497 |
| 256 | 111.01 ms | 87.05 ms | 1.454, 1.479, 1.438, 1.440 |

Recompute and precompute rows were measured in separate ABBA sets, each with
its own chunk=64 recompute baseline (116.86--126.42 ms across the sweep).
The fastest candidate is chunk=128/precompute, not chunk=256. Its local
allocated peak was 1531 MiB vs about 1453 MiB for the chunk=64 baseline.
Do not infer 1.5x whole-training speedup from this component benchmark.

That notebook reported 13 passed / 36 skipped: the loss tests used the NF4
suite's availability guard, unnecessarily requiring bitsandbytes. Standalone
loss tests now use a PyTorch+Triton+CUDA guard. No CUDA correctness/quality
claim is made from the earlier skipped tests. Install the full pinned training
environment, rerun all solution tests, then measure the winning candidate with:

```bash
export SOUP_EXPERIMENT_HEAD_CHUNK=128
export SOUP_EXPERIMENT_PRECOMPUTE_HEAD=1
python task/prepare_data.py
python harness/run_task.py --repeats 2 --out runs_head128_precomputed
python harness/check_rules.py --result runs_head128_precomputed/result.json
```

Keep frozen half storage disabled for this comparison to isolate the change.
Only adopt if the official quality and all other checks pass. Matching baseline
quality within evaluation tolerances is the objective; no quality gain is promised.
