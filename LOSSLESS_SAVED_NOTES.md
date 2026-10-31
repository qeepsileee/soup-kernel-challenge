# Exact saved-data codecs — not enabled by production apply()

Validated fallback remains 85b03db: 64.61s / 4315MiB, all ten checks passed.
Target 40s / 2500--3500MiB is not achieved by the diagnostics below.

## Resource audit supplied by the user (b118a42)

Four diagnostic steps, not a score. Peak unique non-parameter saved storage:
2666.6MiB. Major groups: INT8 MLP values 980MiB plus 61.2MiB scales; native
attention data 672MiB; LoRA A inputs 336MiB; RMSNorm inputs about 330MiB;
LoRA dropout bool masks 162MiB. Shared allocations were counted once.
Imports were inclusive/overlapping: torch 1.56s, transformers 3.04s, TRL trainer
3.54s; model setup 2.90s, first CUDA allocation .37s; apply 7.69s. Do not sum
these or interpret already-loaded import durations near zero as free loading.

NF4 saved MLP: component tests passed, approximate gradient L2 errors ~9--14%.
Full MLP micro: 16.153 -> 16.525ms (0.978x); live after forward 44.5 -> 27MiB.
It saves space but did not speed this benchmark up and has no full-task quality
validation. cuBLASLt micro preference gave ~0.97--1.03x, no substantial gain.
Neither is enabled in production. Full-run quality for NF4 activations remains
unknown; component gradient sanity is not task-quality admission.

## Exact attention K/V storage

The pinned HF SDPA integration explicitly repeats KV heads for masked GQA.
Source: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/sdpa_attention.py

The candidate wraps the existing registry function for Qwen2 training only.
It observes the ORIGINAL repeat_kv calls, snapshots their compact source values,
and substitutes those snapshots only in autograd saved data. Forward still
uses the original repeated tensors and original native SDPA/backend/mask.
Unpacking repeats the compact snapshot and restores the original saved view's
shape, strides and offset. Bounds and storage-layout guards reject unsupported
views. No numerical quantization, model changes, attention math rewrite or RNG
changes are introduced. Native-GQA/no-repeat paths simply retain their original
saved data. Snapshot clones ensure later source mutation cannot change backward.
Only first-order training is in scope.

At B=4,T=512,QH=12,KVH=2,D=128, expanded K/V store 12MiB per layer versus
2MiB for compact snapshots: 10MiB/layer, ~280MiB across 28 layers. This is a
storage estimate, not a measured whole-run peak. Unpacking costs copying time
and needs transient expanded storage for the current backward layer.

## Exact dropout-mask bitpacking

Only nn.Dropout modules under PEFT lora_dropout are wrapped. Their original
forward/RNG call remains unchanged. Bool masks saved for backward are packed
eight per byte by two original Triton pack/unpack kernels. Non-bool, CPU,
noncontiguous and small saved tensors pass through unchanged. Model parameters,
adapter identities, dropout probability and train/eval behavior remain intact.
For the observed 162MiB masks, storage arithmetic suggests ~142MiB reduction.
Extra pack/unpack launches can cost time; benefit needs measurement.

Combined exact-codec saved-storage opportunity is ~422MiB, not a promise of
meeting 3500MiB or improving wall time. These codecs alone cannot establish
40s. The original mask kernels perform useful compression and are original
kernel contributions; their actual share must be measured if ever integrated.

## Verification workflow

Tests check bool/tail round trips, exact dropout output/gradient/RNG, real PEFT
x/A/B gradients and parameter identities, K/V snapshot independence after source
mutation, saved views, native SDPA forward/backward, and actual codec use (not
silent fallback). Benchmark includes native attention and dropout forward PLUS
backward in ABBA order, and prints local peak/live allocated memory. Single-layer
peak can rise while saved data falls; whole-model admission remains essential.

```bash
python -u -m pytest solution/test_lossless_saved.py -q
python -u -m solution.bench_lossless_saved --tokens 512
```

Neither codec is enabled in apply(). Explicit patch_lossless_saved() is for
later diagnostic integration only after the tests and timing justify it. The
existing saved-tensor inventory does not compose with inner codec hooks: do not
compare its saved-storage accounting directly after enabling these codecs.
Use actual allocated memory and official nvidia-smi peaks instead.

Local checks are AST/whitespace and CLI-independent CPU tests only; CUDA
correctness, performance and full-task quality/admission have not been measured
for this candidate. Preserve the validated fallback artifacts.

## First T4 component results (d0c6c13)

All five tests passed after restoring the pinned bitsandbytes dependency.
Attention: 2.615 -> 2.708ms (0.966x), live allocations 18.59 -> 8.59MiB;
incremental local peak 48.69 -> 50.69MiB. Dropout: .275 -> .507ms (0.542x),
live allocations 9.00 -> 6.38MiB; incremental peak 15.00 -> 15.38MiB.
These results verify the expected saved-data reductions and a real time cost.
They do NOT demonstrate a faster whole command or a lower whole-model peak.
Production apply() remains unchanged. A separate memory-oriented branch is
needed for any full-task combination with the approximate NF4 activation codec.
