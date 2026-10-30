# Participant 3: first LoRA fusion candidate

Team synchronization: fetched and fast-forwarded shared intoxicated/nf4-lora to
450c916 before implementation. Preserved teammate CE/chunked-loss files and their
bench_candidate.py edits. apply() is unchanged: LoRA is deliberately opt-in until
GPU tests/benchmarks pass. No loss/CE code is modified by this workstream.

Files owned by this workstream:
- solution/lora_fusion.py: independent Triton scaled-add/cast forward and scaled
  adapter gradient; PEFT vanilla NF4 one-adapter rank-16 wrapper.
- solution/test_lora_fusion.py: real PEFT q/v tests and scalar epilogue tests.
- solution/bench_lora.py: actual PEFT full layer forward/backward comparison,
  with the same existing NF4 optimization on both sides.

The candidate leaves dropout's exact module/call order and cuBLAS A/B GEMMs in
place. No base/adapters are mutated or added. A/B/x gradients flow through their
original autograd operations; the epilogue saves no input tensors. The separate
PEFT defensive clone is omitted because the replacement writes a fresh output
and never modifies base. Multiply rounding to adapter dtype occurs before the
cast/add; enable_fp_fusion=False prevents replacing mul+add with an FMA.
Unsupported adapters, variants, mixed-batch kwargs, merged/disabled adapters,
other ranks/dtypes and non-CUDA use the original PEFT forward.

Status: Python syntax checked locally; GPU tests are SKIPPED on this Mac (no
CUDA/torch/Triton). No speed, memory or training-quality improvement claimed yet.

Kaggle, isolated from teammate tests that may still be under development:

```bash
python -m pytest solution/test_lora_fusion.py -q
python -m solution.bench_lora --tokens 512 --iterations 50
python -m solution.bench_lora --tokens 2048 --iterations 50
```

Tests require bitwise forward epilogue agreement, scaled backward, both output
widths 256/1536, FP16/FP32 input, autocast on/off, nonzero A and B, identical
CUDA RNG state after dropout and unchanged parameter identities/shapes.
After GPU validation, coordinate the single patch_lora() addition to the team's
apply(); then run the full official repeated harness and every rule check.

API semantics/reference, no kernel code copied:
https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py

## Notes for participant 1 (review only; their files were not edited)

At 450c916, CE/chunked loss are present but not called by apply(). In fused_ce.py,
the initial running m is -inf. The entropy moment update contains
alpha_old * (t + s_old * (m - new_m)); on the first tile this evaluates 0*(-inf)
inside the expression and may produce NaN. Initialize the first tile explicitly
or guard the empty previous accumulator, and test entropy finiteness.

In chunked_loss.py the unscaled dZ/dH can be FP16 before grad_loss applies AMP
scaling in backward. Check full-vocabulary (151936) gradients for underflow and
compare against a reference where the loss scale is applied BEFORE FP16 gradient
rounding. Small-vocabulary proportionality tests alone do not establish this.
Keep dtype compatibility with the actual frozen FP32 tied head and autocast.
These are inspection findings awaiting GPU confirmation, not measured failures.
