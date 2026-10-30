# Opt-in INT8 saved-activation candidate

Validated fallback: 983074d (1.173x full-command speedup, 6263MiB peak, all checks
passed on public rows). This new experiment is NOT enabled by solution.apply().
No claimed end-to-end improvement until official GPU validation.

## Hypothesis and implementation

MLP bridge still saves gate/up activations in FP16 for 28 layers. Save each
group of 64 values as signed INT8 + one FP32 scale instead. Forward evaluates
the same example SwiGLU from original FP16 values and stores the same output.
Backward reconstructs approximate gate/up directly inside its derivative kernel,
without a dequantized intermediate tensor or recomputing any matrix multiply.

Saved gate/up bytes change from 4N to 2N + 8*ceil(N/64), approximately 53.1% of
the original for large N (46.9% saving in these activations, NOT total GPU memory).
Quantization uses per-group max/127 and floor(value/scale+0.5). Scales are FP32;
zero groups use scale=1 and codes=0. Tail groups are masked.

Activation math derives from the provided SwiGLU example. Both kernel names
start with example_ and MUST NOT count as own kernels, under the challenge's
rule about rewritten example kernels. Existing useful NF4/CE/LoRA/RMSNorm kernels
must still provide >=5% in the final measured runs. Source attribution is explicit.

Gradients are APPROXIMATE due to quantization. No tolerances in the official
harness, eval, trainer, optimizer, batches, LoRA or task are changed. Acceptance
still requires eval loss <=+0.5%, loss trajectory/metrics within original limits,
speed and memory limits, clean measurement and manual jury review. Reject or
disable this experiment if it fails. Do not call this lossless compression.

The change to bridged_mlp adds only an optional activation callback; its default
remains the original supplied swiglu. patch_compressed_mlp() is opt-in and must
run after the existing bridge patch. Evaluation/no-grad uses the original path;
only saved tensors for training backward need compression. No RNG is used.

## Validation gates

1. CPU numpy sanity of the formula (not a GPU pass): observed relative L2 gradient
   errors ~0.0070/0.0061 on 8192 random groups. This does not establish model quality.
2. GPU tests: exact forward, finite gradients, <2% L2 derivative error on random
   probes, zero/tail groups, saved-byte reduction and the existing bridge tests.
   The 2% criterion is only an initial experiment screen, not a replacement for
   the official training/eval quality gates.
3. Full MLP microbench: alternating baseline/compressed runs, no patch to apply().
   Report both temporary single-layer peak and resident memory after forward.
   Single-layer peak can INCREASE while resident saved memory decreases, because
   original and compressed values coexist briefly during the forward kernel.
4. Only after these gates, enable temporarily via a separate code commit and run
   the full repeated official harness. Keep old result directories untouched.

```bash
python -m pytest solution/test_compressed_mlp.py solution/test_mlp_bridge.py -q
python -m solution.bench_compressed_mlp --tokens 2048 --iterations 20
```

Locally: syntax checked; CUDA tests explicitly skipped (no torch/CUDA/Triton).
No full model quality, time or memory result exists for this candidate yet.
