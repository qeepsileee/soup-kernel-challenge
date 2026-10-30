# Validated public-data run on Kaggle T4

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
