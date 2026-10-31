#!/usr/bin/env bash
# Standalone Kaggle preparation/tests. Package installation is outside scoring.
set -euo pipefail

trial_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$trial_repo_root"

printf '[environment] Checking the T4 session before installing packages\n'
python -u - <<'PY'
import torch
assert torch.cuda.is_available(), "Enable Kaggle GPU T4 before running this script"
name = torch.cuda.get_device_name()
assert "T4" in name, f"The official task needs a T4, got {name}"
print("GPU:", name, "torch:", torch.__version__, flush=True)
PY

printf '[environment] Checking the pinned training dependencies\n'
if ! python -u - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

required = {
    "soup-cli": "0.77.0", "transformers": "5.16.1", "peft": "0.20.0",
    "trl": "0.29.1", "bitsandbytes": "0.50.2", "accelerate": "1.14.0",
    "pytest": None,
}
missing = []
for package, expected in required.items():
    try:
        actual = version(package)
    except PackageNotFoundError:
        actual = None
    if actual is None or expected is not None and actual != expected:
        missing.append(f"{package}: installed={actual}, required={expected or 'present'}")
if missing:
    print("Restoring training dependencies:", *missing, sep="\n", flush=True)
else:
    print("Pinned packages already present; skipping pip installation", flush=True)
sys.exit(bool(missing))
PY
then
    timeout 600 python -m pip install --disable-pip-version-check \
        'soup-cli[train]==0.77.0' 'transformers==5.16.1' \
        'peft==0.20.0' 'trl==0.29.1' 'bitsandbytes==0.50.2' \
        'accelerate==1.14.0' pytest
fi

printf '[environment] Verifying real imports before pytest\n'
python -u - <<'PY'
from importlib.metadata import version
import torch
import triton
import bitsandbytes
import peft
import transformers
import trl
import soup_cli
from peft.tuners.lora.bnb import Linear4bit
from bitsandbytes.functional import quantize_4bit

for package in ("soup-cli", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "pytest"):
    print(f"{package}={version(package)}", flush=True)
print("triton=" + triton.__version__, flush=True)
assert torch.cuda.is_available()
assert "T4" in torch.cuda.get_device_name()
print("Training environment import check passed", flush=True)
PY

for experiment_var in ${!SOUP_EXPERIMENT_@}; do
    unset "$experiment_var"
done

trial_out_dir="${1:-runs_runtime_trial}"
if [[ -e "$trial_out_dir" ]]; then
    trial_out_dir="${trial_out_dir}_$(date -u +%Y%m%d_%H%M%S)"
fi
printf '[trial] Code revision: '
git rev-parse HEAD
printf '[trial] Output directory: %s\n' "$trial_out_dir"

printf '[trial] Running the complete solution test suite\n'
python -u -m pytest solution -q
printf '[trial] Component comparisons, sequence lengths 512 and 128\n'
python -u -m solution.bench_runtime_trial --tokens 512
python -u -m solution.bench_runtime_trial --tokens 128
printf '[trial] Preparing the fixed task data\n'
python -u task/prepare_data.py
nvidia-smi
printf '[trial] Official measurement: two base and two solution runs, then eval\n'
python -u harness/run_task.py --repeats 2 --out "$trial_out_dir"
python -u harness/check_rules.py --result "$trial_out_dir/result.json"
