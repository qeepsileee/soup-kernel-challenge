# Validated public-data run on Kaggle T4

## Latest validated candidate: selective LoRA

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
