# Runtime extents instead of compile-time tensor sizes

UPDATE: 2f7e59c passed all 45 solution tests on T4. Variant diagnostic confirmed
one NF4 binary across three matrices and one binary per compressed activation
direction across seq=64/256/512. Official repeated run passed all ten checks:
79.2s base / 71.8s solution, 14345MiB / 5363MiB, eval +0.001%, own share 8.0%,
79 Soup tests passed. See VALIDATED_RESULTS.md for exact per-run numbers.

Competitive target supplied by the user: beat 54.8 seconds and 4683MiB on the
same Kaggle T4 fixed task, preserving every admission check. This is a target,
not an achieved result. Last fully validated fallback: 983074d, 71.7s/6263MiB.

The INT8 candidate at 5c67852 has component tests but full-run quality/time/peak
are still pending. It used constexpr N in two rewritten-example kernels even
though the sequence length changes between batches. That can create separate
compiler cache entries for different token extents. NF4 decoder's constexpr N
also created size-specific entries for different matrix geometries; LoRA's N
was similarly static although not required for the elementwise computation.

N is now a runtime scalar for these five kernels. Tile sizes, quant group sizes,
quantization format and scalar LoRA scale remain constexpr. All masks and launch
grids remain based on actual input extents; math and data layout are unchanged.
No prebuilt cache is shipped, no official cache clearing/timing is changed.

## GPU checks

```bash
python -m pytest solution -q
python -m solution.bench_jit_reuse
```

The diagnostic observes actual CompiledKernel return values in its own process
and counts unique variants across three NF4 shapes and three MLP sequence lengths.
It restores run methods afterward and is not imported in solution.apply().
Expected: one training-format decode variant and one per compression direction
across aligned lengths. Compiler version/alignment/dtype specialization may still
create entries, so inspect the output rather than infer success from annotations.

Only after GPU verification run the official repeated comparison with idle GPUs,
preserving prior output directories. Compilation is INCLUDED in the official
full-command result. INT8 saved activations still imply approximate backward and
must pass eval loss, loss trajectory and metric tolerances. Runtime extents do
not remove that quality requirement. No performance gain is yet measured here.
