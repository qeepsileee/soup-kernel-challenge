#!/usr/bin/env bash
# Standalone setup, component comparison and official validation of this branch.
set -euo pipefail
trial_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$trial_repo_root"
bash tools/run_kaggle_runtime_trial.sh --prepare-only
for experiment_var in ${!SOUP_EXPERIMENT_@}; do
    unset "$experiment_var"
done

trial_out_dir="${1:-runs_head_padding}"
if [[ -e "$trial_out_dir" ]]; then
    trial_out_dir="${trial_out_dir}_$(date -u +%Y%m%d_%H%M%S)"
fi
printf '[head-padding] Code revision: '
git rev-parse HEAD
printf '[head-padding] Output directory: %s\n' "$trial_out_dir"
python -u -m pytest solution -q
python -u -m solution.bench_head_padding
python -u task/prepare_data.py
nvidia-smi
python -u harness/run_task.py --repeats 2 --out "$trial_out_dir"
python -u harness/check_rules.py --result "$trial_out_dir/result.json"

printf '[head-padding] Additional artifact audit: mask-aware head path must run\n'
python -u - "$trial_out_dir" <<'PY'
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1])
result = json.loads((folder / "result.json").read_text())
pattern = re.compile(r"\[head_padding\] projected_rows=(\d+) total_rows=(\d+) labelled_rows=(\d+)")
runs = result.get("runs", {}).get("solution", [])
assert runs, "No solution runs to audit"
for run in runs:
    text = (folder / run["run"] / "train.log").read_text(errors="replace")
    match = pattern.search(text)
    assert match, f"Mask-aware head dispatch was not recorded in {run['run']}"
    projected, total, labelled = map(int, match.groups())
    assert 0 <= labelled <= projected <= total
    print(f"[head-padding] {run['run']}: first batch projected {projected}/{total}, "
          f"skipped {total-projected} unconsumed rows, retained {labelled} labelled rows", flush=True)
print("Additional audit only; official result/checks files were read without modification.")
PY
