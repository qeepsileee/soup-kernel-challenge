# Validated public-data run on Kaggle T4

## Runtime branch: fa12d01, 3119MiB, all ten public checks passed

Branch: intoxicated/runtime-trial. Code tested:
fa12d016036918faf37ef30a7a86f23e089b6be3, recorded 2026-10-31.
Source: user's official Kaggle console output, runs_runtime_trial. The standalone
launcher restored the pinned environment before measurement. No packages changed
during the scored run; no experiment environment variables were required.

| Run | Full time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 88.39 | 14345 |
| solution_1 | 66.84 | 3119 |
| solution_2 | 68.04 | 3119 |
| base_2 | 80.63 | 14345 |

Means: base 84.51s, solution 67.44s. Reported speedup 1.253x; memory ratio
4.599x. Peak memory fell 11226MiB / 78.3% against this base, and another
160MiB / 4.9% against the preceding 3279MiB candidate. The previous paired
base was 88.16s, so the cross-session wall-time difference does not establish
an isolated speed gain; normalized speedup is slightly lower than 1.262x.

Eval loss 2.04966 -> 2.05031, +0.032% within +0.5%. First-step loss differs
0.04%, mean 0.06%. Own-kernel share 9.8%, lowest of two runs. All 75 solution
tests, all ten automatic checks and all 79 Soup regressions passed. Measurement
was clean, task/outputs matched within tolerance and starter-derived MLP
kernels remained excluded. Hidden data and final manual review remain open.

RoPE kernels are listed as not launched in every measured run, so the 2.487x
component RoPE result must NOT be credited as an active whole-task optimization
at this pin. The strict guard rejected batch-broadcast positional embeddings;
Qwen2's default position_ids have batch dimension 1. A follow-up must verify
actual dispatch with the normal Qwen2Model path and measured launch counters.

Component evidence at B=4,T=512: RoPE 1.528 -> 0.614ms; packed dropout adjoint
0.455 -> 0.450ms (near noise); shared q/v adapters 2.092 -> 2.404ms (slower),
with live allocation 20.02 -> 14.02MiB. At T=128 the q/v path was slower too,
1.728 -> 2.264ms, with live allocation 5.12 -> 3.62MiB. Retain the memory
tradeoff transparently; no isolated LoRA speedup is supported by these numbers.

Preserve runs_runtime_trial/result.json, checks.json, per-run outputs and the
saved notebook with the tested full SHA. They have not been independently
downloaded into this repository. No sub-40s or sub-3000MiB result is established.

## Memory branch: c6f6096, 3279MiB, all ten public checks passed

Branch: intoxicated/memory-trial. Code tested:
c6f6096a58c29ef319e6d7ad40cfac01a561d369, recorded 2026-10-31.
Source: user-supplied official Kaggle console output for runs_memory_trial.
No experiment environment variables were required. The normal apply() on this
branch adds compact saved K/V, bitpacked dropout masks and NF4 saved MLP
activations. The separate intoxicated/nf4-lora branch retains the INT8 defaults.

| Run | Full time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 89.67 | 14345 |
| solution_1 | 68.50 | 3279 |
| solution_2 | 71.17 | 3279 |
| base_2 | 86.65 | 14345 |

Means from printed values: base 88.16s, solution 69.835s (displayed 69.8s).
Reported speedup 1.262x, memory ratio 4.375x; 77.1% less peak memory,
11066MiB saved against its paired base. Peak is 1036MiB / 24.0% below the
previous 4315MiB configuration. Cross-session time differences do not isolate
the effect of these codecs because the paired base times also changed.

Eval loss 2.04966 -> 2.05022, reported +0.027%, below the +0.5% limit.
First-step loss differs 0.04%; mean loss differs 0.06%. Own kernels account
for 8.6%, lowest of two runs; example-derived MLP kernels do not count.
All 66 solution tests, all ten automatic checks and all 79 Soup regression
tests passed, including clean T4 measurement and task/outputs within tolerance.
This is a small quality deviation within public admission limits, not equal
gradients or a quality improvement. Hidden-data and manual review remain open.

Raw official JSON and saved notebook have not been independently downloaded
here. Preserve runs_memory_trial and pin reproduction to the full code SHA
above. See MEMORY_TRIAL.md for reproduction and implementation details.
No final README/SUBMISSION changes or merge into the INT8 branch were made.
The 2500--3500MiB target range was reached; 40 seconds was not reached.

## INT8 branch validated defaults: 85b03db, no environment flags

The current apply() uses chunk=128, precomputed head gradient, frozen half
storage and small K/V weight retention unconditionally, with no
SOUP_EXPERIMENT_* variables required. Other NF4 weight-retention/layout trials
are isolated in nf4_experiments.py and are not imported by apply().
Code tested: 85b03db58bf420b39be9759a2b892807d2c37372, recorded 2026-10-31.
The user supplied full runs_fixed_defaults_kv output: all SOUP_EXPERIMENT_*
variables were explicitly unset before pytest/run_task/check_rules. All 56
solution tests passed (no skips); all ten admission checks and 79 Soup
regression tests passed. This closes the environment-dependent reproduction
gap. README/SUBMISSION finalization remains deferred at the team's request.

| Run | Full time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 75.75 | 14345 |
| solution_1 | 63.44 | 4315 |
| solution_2 | 65.78 | 4315 |
| base_2 | 82.18 | 14345 |

Mean/median of two runs: base 78.965s, solution 64.61s (displayed 79.0 / 64.6s).
Reported speedup 1.222x, memory ratio 3.324x; about 18.2% less wall time and
69.9% less peak memory (10030MiB saved). Eval loss 2.04965 -> 2.04946
(-0.009%), with no public-quality regression detected. This tiny loss decrease
is not evidence of general quality improvement. Loss curve first-step deviation
0.04%, mean 0.04--0.05%. Own kernels 8.5%, lowest of two solution runs; example
and paired-example derivatives are excluded as required. Clean measurement passed.

Source is the user's console output, not independently downloaded raw JSON.
Preserve the runs_fixed_defaults_kv directory and the saved notebook version
with COMMIT pinned to the full SHA above. Later documentation-only commits do
not replace that tested pin. No improvement is inferred solely from the time
difference against 66.66s: paired base times also changed between sessions.
The under-50s target has NOT been achieved; gap from this mean is >14.61s.

Reproduction on the pinned T4 environment, without experiment variables:

```bash
python -m pytest solution -q
python harness/run_task.py --repeats 2 --out runs_fixed_defaults_kv
python harness/check_rules.py --result runs_fixed_defaults_kv/result.json
```

## Prior validated candidate: small K/V retention via environment

Code tested: 8e44f13b68528626dd919fcb192fa127615fde8b, recorded 2026-10-31.
Source: user supplied runs_retained_kv/result.json summary and all ten checks.
Historical flags: HEAD_CHUNK=128, PRECOMPUTE_HEAD=1, HALF_STORAGE=1,
RETAIN_NF4_WEIGHT=kv, NF4_DX_LAYOUT=original (all prefixed SOUP_EXPERIMENT_).
The newer production code expresses the same policy without these variables.

| Run | Full time (s) | Training loop (s) | Peak GPU memory (MiB) |
|---|---:|---:|---:|
| base_1 | 81.53 | 59.84 | 14345 |
| solution_1 | 66.38 | 44.54 | 4315 |
| solution_2 | 66.95 | 44.61 | 4315 |
| base_2 | 82.33 | 60.70 | 14345 |

Official summary: base 81.93s, solution 66.66s, speedup 1.2291x (18.6% less
wall time); memory ratio 3.3244x (10030MiB / 69.9% less memory). Both solution
runs measured the same peak. Eval loss 2.04968 -> 2.04944 (-0.012%); do not
claim statistically significant quality improvement. Own kernels 8.4%, lowest
of two runs. All ten checks passed, including clean measurement and 79 Soup
regression tests. First training loss differs 0.04%, mean differs 0.04--0.05%.

Median loop time 60.27 -> 44.58s; time outside the loop stays about 22s.
The earlier reader falsely printed first_2_steps=0 because run_task removes
step_s from embedded metrics. That field is available in each run's own
metrics.json; solution.summarize_run reads those files and prints null rather
than zero when missing. No zero-startup-cost conclusion is supported.

Compared with the preceding 67.66s/4269MiB result, this mean is about 1s lower
and memory 46MiB higher. Sessions had different paired bases, so this is not an
isolated causal estimate of retention's benefit. Under 50s has not been achieved.

## Prior validated candidate: half storage + precomputed head, chunk 128

Code tested: dee0170041604400b1ddcc24eeab766b4a569ea0. Recorded 2026-10-31.
Source: user's official Kaggle output, runs_head128_half_storage/result.json.
Required settings (all three): SOUP_EXPERIMENT_HEAD_CHUNK=128,
SOUP_EXPERIMENT_PRECOMPUTE_HEAD=1, SOUP_EXPERIMENT_HALF_STORAGE=1.
Defaults alone do not reproduce this configuration.

| Run | Wall time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 76.92 | 14345 |
| solution_1 | 66.53 | 4269 |
| solution_2 | 68.79 | 4269 |
| base_2 | 84.14 | 14345 |

Means 80.53s / 67.66s; reported speedup 1.190x (16.0% less wall time).
Memory ratio 3.360x; reduction 10076MiB (70.2%). Eval losses display 2.04967
base / 2.04944 solution (-0.011%); this small decrease is not evidence of
general quality improvement. Loss curves differ by 0.04% first step and mean.
Own kernels: 8.1%, lowest of two runs. ALL TEN CHECKS PASSED, 79 Soup tests
passed. The same code's complete solution suite previously passed 49 tests.
The half-storage conversion and changed activation rounding now pass the
public-task quality/metric/adapter checks; hidden-data review remains pending.

Compared with the prior configuration's 5423MiB peak, this run uses 1154MiB
less (21.3%). Whole-run timing changes cannot be attributed solely to half
storage: paired base time also fell from 87.19s to 80.53s and the speed ratio
remained 1.190x. Keep the exact pin, flags and Kaggle artifacts as fallback.
Target below 50s has NOT been achieved: remaining gap from this mean is >17.66s.
The user-reported competitor's 4683MiB is exceeded on memory (4269MiB uses
414MiB less), but their reported 54.8s remains faster; measurement equivalence
has not been independently verified.

Profile this current configuration using `python -m solution.profile_backward
--half-storage` at diagnostic commit 8a275cd or later. That instrumentation is
not part of the scored measurement and its elapsed time must not be scored.

## Prior validated candidate: precomputed head gradient, chunk 128

Code tested: dee0170041604400b1ddcc24eeab766b4a569ea0. Recorded 2026-10-31.
Source: user's official Kaggle output, runs_head128_precomputed/result.json.
Required configuration: SOUP_EXPERIMENT_HEAD_CHUNK=128,
SOUP_EXPERIMENT_PRECOMPUTE_HEAD=1, half storage disabled. These are opt-in
implementation settings, not changes to the task batch size or token sequence.

| Run | Wall time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 90.00 | 14345 |
| solution_1 | 73.65 | 5423 |
| solution_2 | 72.85 | 5423 |
| base_2 | 84.38 | 14345 |

Means 87.19s / 73.25s; reported speedup 1.190x (16.0% less wall time).
Memory ratio 2.645x; reduction 8922MiB (62.2%). Both eval losses display
2.04962, relative difference +0.000%. Loss curves differ by about 0.01% mean.
Own kernels: 8.1%, lowest of two runs. ALL TEN CHECKS PASSED, 79 Soup tests
passed, complete solution suite 49 passed (no skips). Quality is confirmed
within public-task tolerances; it is not established to be improved or identical
on arbitrary inputs. The head gradient uses a rounding approximation.

The previous configuration's 71.77s cannot be directly ranked against 73.25s:
its paired base was 79.185s rather than 87.19s. The new paired speed ratio is
higher, but an interleaved comparison of both solution configurations is still
needed to isolate their whole-run difference. Below 50s has NOT been achieved.
Keep this exact pin/settings and its Kaggle artifacts as a validated fallback.

Next candidate: combine this head configuration with
SOUP_EXPERIMENT_HALF_STORAGE=1. It remains unvalidated; recheck the complete
task rather than assuming exact stored weight values imply unchanged arithmetic.

## Prior validated candidate: INT8 activations with runtime extents

Code tested: 2f7e59cd359d23245b34fd3244b3286f290010d3. Recorded 2026-10-31.
Source: user's official Kaggle output; raw artifacts are in runs_int8_runtime/.

| Run | Wall time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 77.08 | 14345 |
| solution_1 | 71.92 | 5363 |
| solution_2 | 71.62 | 5363 |
| base_2 | 81.29 | 14345 |

Means 79.185s / 71.77s; reported speedup 1.103x, about 9.4% less wall time.
Memory ratio 2.675x, reduction 8982MiB (62.6%). Eval loss displayed as
2.04963 / 2.04966 (+0.001%). Own kernels: 8.0%, lowest of two runs.
ALL TEN CHECKS PASSED, 79 Soup tests passed. Preceding solution suite: 45 passed.
INT8 saved activations imply approximate backward, but the public task's loss,
metrics and eval-quality checks all passed; hidden-data review is still pending.

Earlier INT8 code 5c67852 also passed all checks, but measured 81.56/81.09s
solution vs 78.19/81.19s base, same 5363MiB peak (speed ratio 0.980x).
Runtime-size reuse diagnostic at 2f7e59c confirmed one compiled training-format
NF4 variant across three weight shapes and one compression forward/backward
variant across seq lengths 64/256/512. The subsequent full run was about 9.6s
faster than the earlier INT8 run; this cross-session difference is not an isolated
on/off ablation of compilation cost.

User-reported competitor target: 54.8s / 4683MiB. This candidate has NOT beaten
those values; same-task/harness equivalence of competitor numbers is unverified.
Remaining raw gap about 17s and 680MiB. Retain both this pin and 983074d as
validated alternatives; score ratios across sessions have different baselines.

## Prior validated candidate: chunked head + parallel CE

Code tested: 983074da7f131839065723c821736e2d57a19c1d. Recorded: 2026-10-31
(Asia/Almaty). Source: official harness console supplied by the Kaggle operator.
Raw artifacts live in runs_chunked_parallel/ in that Kaggle session; preserve
result.json, checks.json, per-run logs and the saved notebook. No raw JSON was
independently downloaded or fabricated in this repository.

| Run | Wall time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 86.97 | 14345 |
| solution_1 | 71.34 | 6263 |
| solution_2 | 72.12 | 6263 |
| base_2 | 81.31 | 14345 |

Mean full-command times: 84.14s vs 71.73s. Official speedup 1.173x, about
14.7% less time. Memory ratio 2.290x; reduction 8082MiB, about 56.3%.
Eval losses displayed as 2.04967 vs 2.04965, reported relative change -0.001%.
Own-kernel share: 8.1%, lowest of two solution runs. ALL TEN CHECKS PASSED,
including clean single-T4 measurement, matching task/outputs within tolerance,
loss trajectory and 79 Soup regression tests. The complete solution GPU suite
passed 43 tests before this run.

Active own kernels: tiled CE statistics/merge, AMP-scaled chunk gradient,
selective LoRA epilogue/gradient, NF4 decoding and RMSNorm forward/backward.
The starter SwiGLU remains excluded from original-kernel accounting. Unused
experimental/reference kernels being listed as unlaunched is not a failed check.

Compared with the prior solution peak of 10343MiB, the combined chunked-head
version uses 4080MiB less. This is cross-run evidence, not an isolated ablation;
all speed/memory claims describe the full combined configuration. Component CE
statistics speedup is not the full-training speedup.

Current validated fallback for reproduction/submission: pin EXACTLY 983074d,
not an untested later code revision. Docs-only commits do not change the tested
implementation. Open-data/seed checks do not replace hidden-data testing or jury
manual review. Final README, platform repository, SUBMISSION.md, public saved
Kaggle notebook with this pin, presentation and demo still need completion.

## Prior validated candidate: selective LoRA

Code tested: 8ff20b2a417b7155eb136692f6d750b07270552c. Date: 2026-10-30.
Source: team member's official Kaggle console output. Preserve original JSON,
logs and saved notebook from runs_lora_selective; those files are not copied here.

| Run | Wall time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 87.33 | 14345 |
| solution_1 | 74.39 | 10343 |
| solution_2 | 76.59 | 10343 |
| base_2 | 82.71 | 14345 |

Mean times 85.02s / 75.49s; reported speedup 1.126x (about 11.2% less wall time).
Memory ratio 1.387x, reduction 27.9% or 4002MiB. Eval loss displayed as
2.04966 / 2.04965; reported relative change -0.000%. Own-kernel share 6.9%,
lowest of the two solution runs. All ten checks passed, including clean T4
measurement, identical task/outputs within tolerance and 79 Soup regression tests.
LoRA's isolated three-test suite passed on T4 before this full run.

The combined solution includes NF4, RMSNorm, entropy, the MLP bridge, selective
LoRA and the starter SwiGLU. These total gains are NOT an isolated LoRA ablation.
Separate repeated component benchmarks established about 1.10x for 2048-token
q-like projections with AMP. Other shapes retain the original PEFT path.

IMPORTANT: teammate commit 549d1c5 subsequently wired chunked loss into apply().
That code is not included in this validated run. New integration must independently
pass the GPU suite and every official rule before inheriting any score claims.
Keep the exact 8ff20b2 pin as a validated fallback and the current benchmark source.
Hidden rows/seed and final manual jury review remain outstanding.

## Latest validated candidate: MLP bridge

Code tested: 968e327172552a97023f4f0aef91fa9f337f5f2e. Date: 2026-10-30.
Source: user-supplied official harness console output; raw JSON lives in the
Kaggle runs_mlp_bridge directory and has not been independently downloaded here.

| Run | Wall time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 78.94 | 14345 |
| solution_1 | 76.83 | 10343 |
| solution_2 | 77.59 | 10343 |
| base_2 | 84.41 | 14345 |

Official mean: base 81.7s, solution 77.2s. Reported speedup 1.058x, above the
1.05 scoring threshold in this run. Peak memory ratio 1.387x; reduction 4002MiB,
about 27.9%. Eval losses displayed as 2.04966 / 2.04964; reported -0.001%.
Own-kernel coverage 6.2%, lowest of two runs. All ten automatic checks passed;
79 Soup tests passed. Component suite previously passed 20 tests on T4.

Against the prior validated solution peak (12383MiB), the bridge reduced peak
by another 2040MiB. This comparison is between runs, not a controlled ablation;
the full baseline/solution metrics describe the combined solution including the
starter SwiGLU. Example kernels are NOT counted as original team work.

Keep 968e327 as the tested code revision for reproduction. Subsequent docs-only
commits do not require replacing its pin. Preserve the saved Kaggle version and
runs_mlp_bridge/result.json, checks.json and per-run logs for team handoff.
Hidden-data quality, coverage and timing still need the jury's validation;
automatic checks do not replace manual review. Speed is only modestly above
the 5% threshold, with visible variation in the two base runs.

## Prior validated candidate: decoder + RMSNorm + entropy

Code tested: d863fedf70cbcf6c8794e0dfb27bbe2cc69e8a6d.
Date: 2026-10-30. Source: console output supplied by the team member running Kaggle;
the original runs/result.json and runs/checks.json should also be preserved in
the saved Kaggle notebook output. This document does not fabricate those files.

| Run | Wall time (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| base_1 | 83.81 | 14345 |
| solution_1 | 81.95 | 12383 |
| solution_2 | 81.64 | 12383 |
| base_2 | 84.16 | 14345 |

Reported ratios: speedup 1.027x, memory base/solution 1.158x. Speed counts as
1.000 for scoring because the improvement is below the 1.05 threshold.
Memory reduction is about 13.7%, or 1962 MiB. Eval loss displayed as 2.04963 for
both sides; reported relative change +0.000%. Useful own-kernel share was 5.6%
in the lowest of the two solution runs. All ten automatic checks passed,
including a clean measurement and 79 Soup regression tests.

The current combination is the starter SwiGLU, own NF4 decoder, bnb-compatible
linear dispatch with cuBLAS on the unfused branch, frozen RMSNorm forward/backward,
and two-stage entropy. The starter SwiGLU is NOT an own kernel. Memory and speed
numbers above describe the combination, not the isolated contribution of NF4.
Experimental dense/inline-decode Triton GEMMs are disabled in normal training.

Limitations: public rows and this seed only. The jury's hidden rows/seed may
change the result. Coverage has only a 0.6 percentage-point margin over 5%; the
harness warns about this. Full-command speed has not reached the scoring
threshold. Final manual code review and submission artifacts are still required.

Next diagnostic, without overwriting the validated runs directory:

```bash
python harness/run_task.py --only solution --profile --out runs_profile
```

Inspect runs_profile/solution_profile/profile.json. This diagnostic includes an
early-stopped profiled run and is NOT a submission-quality comparison. Do not
replace the validated result or report its profiled timings as a speed score.

Team integration: coordinate rmsnorm.py with participant 2 and entropy.py with
participant 1, avoiding competing patches to the same function. The final code
must be in the repository created for the team by the hackathon platform.
