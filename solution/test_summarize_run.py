import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class ArtifactSummaryTests(unittest.TestCase):
    def test_required_kernel_must_launch_in_every_solution_raw_metrics(self):
        from solution.summarize_run import main
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "result.json"
            path.write_text(json.dumps({"summary": {}, "runs": {"solution": [
                {"run": "solution_1", "metrics": {"triton_launches": {"rotary_forward_kernel": 99}}},
                {"run": "solution_2", "metrics": {"triton_launches": {"rotary_forward_kernel": 99}}},
            ]}}))
            for name, count in (("solution_1", 10), ("solution_2", 0)):
                (folder / name).mkdir()
                (folder / name / "metrics.json").write_text(json.dumps({
                    "triton_launches": {"rotary_forward_kernel": count}}))

            def report():
                output = io.StringIO()
                argv = ["summarize_run", str(path), "--require-kernel", "rotary_forward_kernel"]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
                    main()
                return output.getvalue()

            with self.assertRaisesRegex(SystemExit, "solution_2: rotary_forward_kernel"):
                report()
            (folder / "solution_2/metrics.json").write_text(json.dumps({
                "triton_launches": {"rotary_forward_kernel": 12}}))
            self.assertEqual(report().count("[dispatch]"), 2)

    def test_required_kernel_rejects_missing_solution_runs(self):
        from solution.summarize_run import main
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps({"summary": {}, "runs": {}}))
            argv = ["summarize_run", str(path), "--require-kernel", "rotary_forward_kernel"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "no solution runs"):
                    main()

    def test_steps_are_read_from_raw_metrics_and_missing_is_not_zero(self):
        from solution.summarize_run import main
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "result.json"
            path.write_text(json.dumps({"summary": {}, "runs": {"solution": [
                {"run": "solution_1", "wall_s": 12., "metrics": {"train_loop_s": 5.}}
            ]}}))
            (folder / "solution_1").mkdir()
            raw = folder / "solution_1/metrics.json"
            raw.write_text(json.dumps({"train_loop_s": 5., "step_s": [1.2, 2.3, 1.5]}))

            def report():
                output = io.StringIO()
                with patch.object(sys, "argv", ["summarize_run", str(path)]), contextlib.redirect_stdout(output):
                    main()
                line = next(line for line in output.getvalue().splitlines() if line.startswith("solution_1 "))
                return json.loads(line.split(" ", 1)[1])

            self.assertEqual(report()["first_two_steps_s"], 3.5)
            raw.unlink()
            missing = report()
            self.assertIsNone(missing["first_two_steps_s"])
            self.assertEqual(missing["outside_loop_s"], 7.)
