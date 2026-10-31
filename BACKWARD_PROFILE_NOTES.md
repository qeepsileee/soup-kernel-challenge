# CTO-directed backward diagnostic

`python -m solution.profile_backward` runs the existing short harness profiler
with extra attribution ranges. It does not edit harness/, task/ or tests/.
It is never imported by solution.apply() and is not a scored measurement.
The script runs in its own process; do not run a scored test concurrently.

Default configuration matches the latest quality-validated head candidate:
chunk=128, precomputed head gradient enabled, frozen half storage disabled.
Flags and profiler active-step count are saved in the diagnostic report.
Other options: --head-chunk N, --recompute-head, --half-storage, --out DIR.
Use separate output directories when comparing configurations.

The script wraps the existing NF4Linear forward/backward without reimplementing
their math, and uses record_function around their decoder and Python matmul.
A context variable scopes matmul labels to NF4 backward. Packed weight pointers
map each context to its original projection and weight shape. Both head-loss
Functions receive separate forward/backward ranges. Patches are restored in
finally, including on a training error. Profiler record_shapes is enabled.

Output `runs_backward_profile/backward_breakdown.json` contains:

* Frozen projection backward CUDA time, including nested decode and dX GEMM.
* Backward decode and dX GEMM separately, also per projection shape.
* Forward projection/decode CUDA time.
* Head forward/backward time, including recomputation/precomputation as selected.
* Copy/cast operations attributed to their nearest instrumented range.
  Unlabelled transformer/attention copies remain explicitly unlabelled.
* Raw top CUDA kernels, including attention if it is significant.

Categories overlap: do NOT sum backward with its decode/GEMM children. GPU
shares use the sum of CUDA kernel durations, not whole-process wall time.
CPU inclusive times overlap too. Copy operation self times are reported rather
than adding overlapping aten::to/_to_copy/copy_ durations. CUDA allocations and
operator overhead can still matter even when a copy's device time is small.
If CUDA range attribution is empty, a warning is emitted: do not interpret zero
as absence of work. Inspect profiler linkage before deciding what to optimize.

Decision order from the team's CTO/TL:

1. Frozen NF4 backward: quantify decode vs dX GEMM. Eliminating a small decode
   share cannot remove the much larger matrix multiplication share. Retaining
   decoded weights trades memory for time; earlier direct Triton GEMM candidates
   regressed on this T4/toolchain and remain disabled.
2. Repeated dtype conversions and the head. Compare half storage independently
   against the same labelled-token/task configuration, with quality checks.
3. Attention only if the current profile establishes a material share.

Timing differences under this instrumentation are not official speedup evidence.
After selecting a candidate, use the unchanged full run_task/check_rules flow
with repeats=2 and idle GPUs. Target below 50s has not been reached.

Local verification: CLI help/argument validation, Python AST and diff whitespace
checks. This Mac has no CUDA/PyTorch test environment; attribution must be
verified from the first T4 output before it is used for decisions.
