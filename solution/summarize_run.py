"""Read existing harness artifacts without loading CUDA or repeating training."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    if not args.result.is_file():
        parser.error(f"result not found: {args.result}")
    result = json.loads(args.result.read_text())
    print(json.dumps(result.get("summary", {}), indent=2))
    for runs in result.get("runs", {}).values():
        for run in runs:
            path = args.result.parent / run["run"] / "metrics.json"
            # run_task deliberately removes step_s from the embedded metrics.
            metrics = json.loads(path.read_text()) if path.is_file() else run.get("metrics", {})
            steps = metrics.get("step_s")
            first_two = round(sum(steps[:2]), 4) if isinstance(steps, list) and len(steps) >= 2 else None
            loop, wall = metrics.get("train_loop_s"), run.get("wall_s")
            print(run["run"], json.dumps({
                "total_s": wall, "train_loop_s": loop, "apply_s": metrics.get("apply_s"),
                "first_two_steps_s": first_two,
                "steady_loop_s": metrics.get("timed_loop_s"),
                "median_step_s": metrics.get("median_step_s"),
                "outside_loop_s": round(wall-loop, 2) if wall is not None and loop is not None else None,
            }))
    checks_path = args.result.parent / "checks.json"
    if checks_path.is_file():
        checks = json.loads(checks_path.read_text())
        recorded_hash = checks.get("result_sha256")
        if recorded_hash and recorded_hash != hashlib.sha256(args.result.read_bytes()).hexdigest():
            raise SystemExit("checks.json belongs to different result bytes; do not use it as admission evidence")
        for check in checks.get("checks", []):
            print("PASS" if check["ok"] else "FAIL", check["check"], check["detail"])
    else:
        print("checks.json not found; admission has not been established by this report")
    print("null means unavailable, not zero. apply_s overlaps outside_loop_s; do not add it again.")


if __name__ == "__main__":
    main()
