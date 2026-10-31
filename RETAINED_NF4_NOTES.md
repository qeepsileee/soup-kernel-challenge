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
