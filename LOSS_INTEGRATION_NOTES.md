# Combined chunked-head / parallel-CE candidate

Authorized by the user to continue participant 1's work on 2026-10-31.
Fetched origin before editing. Preserved their local recomputation, labelled-row
compaction, FP32-head conversion and constructor-hook changes, and now commit
them with the parallel CE integration so Kaggle receives a complete candidate.

## Implementation

- chunked_loss.py now imports parallel_cross_entropy under the existing CE alias.
- Forward runs cuBLAS head chunks and parallel loss/entropy/argmax statistics,
  saving compact LSE and hidden states, not full logits or unscaled FP16 dH.
- Backward recomputes labelled token rows only. Real grad_loss/AMP scaling is
  applied before dZ is stored in FP16 and multiplied by the frozen head.
- Denominator is a runtime Triton argument, not constexpr: different labelled
  token counts must not generate a new backward binary for each training step.
- Entropy retains prompt-token positions; trainer still uses attention_mask.
  Loss/accuracy/rewind use next-token shift and -100 labels.
- A frozen, bias-free head is required. Unsupported biased heads fall back at
  hook attachment instead of silently omitting bias.
- The previously local trainer-constructor hook is retained to actually attach
  the chunked path. Normal unlabeled generation continues through original forward.
- Legacy serial CE's bool/int branch mismatch is fixed and its empty running
  entropy accumulator avoids 0*(-inf). It is retained for tests/comparison.

## Tests and limits

Existing AMP test assumed exact proportionality after FP16 rounding; it now
compares each scale (1/1024/65536) against real PyTorch autograd independently.
Existing head in that test was trainable; it is now frozen as the task requires.
Added full-151936-vocab FP32 frozen head under FP16 autocast, ignored-row gradients,
all-ignore backward, and trainable-head rejection. Hook restoration compares
bound method function/receiver, because Python creates a fresh bound method on
each attribute lookup (identity of bound-method wrapper is not an invariant).
Numerical tolerances on existing loss/gradient comparisons were not loosened.

Parallel CE alone already passed 4 T4 tests and its statistics-only benchmark
was 10.6-24.9x faster than PyTorch on the supplied shapes. These are COMPONENT
results, not claims about this combined head or full training. New integrated
tests/quality/timing/peak/coverage remain unmeasured until Kaggle execution.
Local environment has no CUDA; local checks are syntax plus explicitly skipped
GPU cases, not a GPU pass.

## Run in Kaggle

```bash
python -m pytest solution -q
python task/prepare_data.py
python harness/run_task.py --repeats 2 --out runs_chunked_parallel
python harness/check_rules.py --result runs_chunked_parallel/result.json
```

Run tests before training and stop on failures. Use an idle T4, the unchanged
official harness and pinned dependencies. Keep old runs/ and runs_lora_selective/
results. The last fully validated fallback code is 8ff20b2 (1.126x speed,
1.387x memory ratio, all checks passed on public rows). Do not overwrite those
claims with new estimates or select this candidate without official validation.
