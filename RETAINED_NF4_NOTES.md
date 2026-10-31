# Forward-to-backward NF4 weight retention (opt-in)

CTO-directed profile of the validated half-storage configuration (user supplied):
16 active steps, about 7.52s total CUDA kernel work. Frozen projection GEMM
ranges: forward 2117.82ms (28.2%), backward 1971.18ms (26.2%). Raw NF4 decode
485.02ms (6.4%); head 565.38ms (7.5%); attention kernels 588.40+227.12ms
(about 10.8%). The earlier profile without half storage had substantial casts;
the new profile confirms [half_storage] and those large casts disappear.

Triton decode kernels are visible in raw CUDA events but are not attributed to
their CPU record_function ranges. Therefore the reported 0ms decode subranges
are NOT evidence of zero work. The diagnostic now says attribution unavailable
and prints raw decode separately. Do not add overlapping GEMM/range figures.

`SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT=1` keeps the exact FP16 weight decoded in
NF4Linear.forward until that invocation's backward. Backward uses that tensor
with the same torch.matmul rather than decoding it again. No persistent cache,
no shared scratch buffer, no weight changes, no task/optimizer changes. Tiny-M
custom bnb GEMM dispatch stays untouched and uses ordinary backward decode.
Default is off. This does not eliminate or accelerate the GEMM itself.

Tradeoff: potentially several GiB more live activation memory because dense
weights for all eligible layers survive the forward. Decode savings are bounded
by only part of the 6.4% CUDA decode share; alone this cannot remove >17s from
the official 67.66s command. Fewer allocations/launches may also help, but need
measurement. Own-kernel >=5% admission must be rechecked after removing launches.

Correctness test compares two live forward graphs consumed in reverse order,
exact outputs/input gradients, and verifies the number of decoder calls halves
on the supported dense route. A separate ABBA layer benchmark compares total
forward/backward (including decode), not isolated GEMM. The full model's retained
memory is not represented by the single-layer benchmark.

```bash
python -m pytest solution -q
python -m solution.bench_retained_nf4 --tokens 2048
```

Only if correctness passes and timings justify a full run:

```bash
export SOUP_EXPERIMENT_HEAD_CHUNK=128
export SOUP_EXPERIMENT_PRECOMPUTE_HEAD=1
export SOUP_EXPERIMENT_HALF_STORAGE=1
export SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT=1
python harness/run_task.py --repeats 2 --out runs_retained_nf4
python harness/check_rules.py --result runs_retained_nf4/result.json
```

Fallback: dee0170 with the first three flags enabled, 67.66s / 4269MiB, all
ten checks passed. No CUDA correctness/performance claim is made for retention
yet. Local AST/CLI/whitespace validation only; GPU tests skip on this Mac.

## First T4 results and selective policy

The user ran acf10cd: all 50 solution tests passed. At 2048 rows, retaining all
weights gave q/o/gate/up/down only about 1.03x, with some paired regressions.
K/V were stronger and consistent: k 0.617 -> 0.429ms (1.440x), v 0.604 ->
0.405ms (1.492x). No official whole-model retention result exists yet.

New `SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT=kv` selects only shape (256,1536),
the actual K/V shapes. Across 28 layers their retained weights total about
42MiB, instead of roughly 2.5GiB for all seven frozen projections. This is a
live-weight estimate, not a prediction of official whole-run peak memory.
The 1.49x V-projection result is NOT a 1.49x model speedup: these are small
projections. Retaining all weights is not being enabled by default.

## Separate cuBLAS dX layout experiment

`SOUP_EXPERIMENT_NF4_DX_LAYOUT=transposed` changes only the backward layout:
`grad @ W` becomes `F.linear(grad, W.T.contiguous())`. The mathematical product
is the same but cuBLAS may pick a different kernel, and rounding can differ.
The transpose's allocation and copy cost are included in the diagnostic timing.
It is off by default, independently of retention. Tiny-M forward dispatch stays
unchanged. No Triton GEMM is introduced. CUDA correctness/performance for this
layout has not yet been measured; full quality checks are still required.

Tests cover selective-retention identity and layout gradient tolerance with
noncontiguous upstream gradients. Benchmark layout alone before combining it:

```bash
python -m pytest solution -q
python -m solution.bench_retained_nf4 --retain 0 --dx-layout transposed --tokens 2048
python -m solution.bench_retained_nf4 --retain 0 --dx-layout transposed --tokens 512
```

The benchmark accepts --retain 0/1/kv and --dx-layout original/transposed,
always compares against retention off/original layout, and restores both
environment variables on exit. Only choose policies after measured benefit.

## Layout rejected after T4 measurement

User ran 8e44f13: all 51 solution tests passed. CUDA dX-layout microbenchmarks
with retention disabled consistently regressed. Speed ratios base/candidate:

| Projection | 2048 rows | 512 rows |
|---|---:|---:|
| q | 0.845 | 0.919 |
| k | 0.920 | 0.940 |
| v | 0.929 | 0.927 |
| o | 0.867 | 0.918 |
| gate | 0.824 | 0.592 |
| up | 0.825 | 0.554 |
| down | 0.858 | 0.614 |

Do not enable transposed dX in the solution. The experiment remains off by
default; mathematical correctness did not imply performance improvement.
There is no reason to spend a full official run on this rejected layout.

Only the small K/V retention policy is ready for whole-model evaluation. It
does not change GEMM layout or arithmetic; it removes a backward decode for
the small projections. Candidate may provide only a small total benefit.
No whole-run speed or peak-memory result exists for it yet. Run with pinned
8e44f13 code and these explicit settings to avoid environment ambiguity:

```bash
export SOUP_EXPERIMENT_HEAD_CHUNK=128
export SOUP_EXPERIMENT_PRECOMPUTE_HEAD=1
export SOUP_EXPERIMENT_HALF_STORAGE=1
export SOUP_EXPERIMENT_RETAIN_NF4_WEIGHT=kv
export SOUP_EXPERIMENT_NF4_DX_LAYOUT=original
python harness/run_task.py --repeats 2 --out runs_retained_kv
python harness/check_rules.py --result runs_retained_kv/result.json
```

Validated fallback remains dee0170 with the first three flags only:
67.66s / 4269MiB. The retention microbenchmarks do not demonstrate a path to
below 50s. That target still requires a substantially larger improvement.
