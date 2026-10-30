# Independent tiled vocabulary-statistics candidate

Status: CUDA validation and timings pending. NOT enabled by apply() and NOT wired
into participant 1's chunked loss. Changes are confined to new parallel_ce.py,
test_parallel_ce.py and bench_parallel_ce.py. Existing locally modified
chunked_loss.py, fused_ce.py and patch_chunked_loss.py belong to participant 1
and were left intact. The previous validated fallback is 8ff20b2.

## Why this direction

The validated profile showed expensive full-vocabulary copies and lm_head casts.
Participant 1 is already removing full logits and fixing AMP-safe recomputation.
The remaining CE implementation scans about 38 vocabulary tiles serially in one
CTA per token. This candidate instead assigns independent CTAs to (token, tile),
then merges small stable statistics. It targets reduction latency and parallelism
without replacing cuBLAS GEMM or changing tokens/metrics.

For each tile store maximum m, exp sum s, shifted moment t and first argmax.
Merge with global maximum M and factors exp(m-M):
S=sum(f*s); T=sum(f*(t+s*(m-M))); H=log(S)-T/S.
Loss is (M-z_target)+log(S), avoiding cancellation from large common offsets.
Optional gradient is a separate kernel launched after stats finish, supporting
in-place output, FP32 gradient buffers and a scalar AMP upstream scale applied
BEFORE rounding. Ignored targets have zero loss/gradient but retain entropy.

Scratch is 4 FP32 values per row/tile, plus two FP32 row statistics, not full
softmax/log-softmax buffers. LSE can be saved into the caller's FP32 buffer for
recomputation, compatible with participant 1's new lse_out argument. Full chunk
logits still exist; this module alone does not implement the output head.

Potential cost: two statistics launches instead of one; small vocabularies/row
counts can lose. Do not enable without measured improvement on actual chunks.
No padding time or unused repeated work to manipulate own-kernel share.

## Validation

```bash
python -m pytest solution/test_parallel_ce.py -q
python -m solution.bench_parallel_ce --iterations 20
```

Tests cover loss/entropy/accuracy, full vocabulary, tails, all ignored rows,
uniform argmax ties, FP16/FP32, strided input, in-place gradients, FP32 grad_out,
saved LSE, AMP scale before gradient rounding and large common logit offsets.
They compare to PyTorch, not only to the other candidate. Local CPU tests skip
CUDA explicitly. The benchmark compares all compact statistics, alternating
serial/tiled order in four rounds for 64/256 rows and FP16/FP32. If the serial
candidate's entropy is nonfinite, report that and time PyTorch as the baseline;
always validate tiled numerical results against PyTorch first.

## Integration contract for participant 1

If correctness and speed pass, candidate import is:

```python
from .parallel_ce import parallel_cross_entropy as fused_cross_entropy
```

The tuple matches (token_loss, token_entropy, hits, total_loss, optional_grad).
Existing keyword lse_out is supported. Recomputed gradients must retain the real
upstream AMP scaling before FP16 storage; do not precompute unscaled FP16 dH.
Coordinate the import change rather than installing competing trainer patches.
Full repeated harness, eval loss, per-row rewind, metrics, 79 Soup tests and
own-kernel >=5% checks are still required after integration. A microbenchmark
cannot establish whole-command speed or hidden-data admission.
