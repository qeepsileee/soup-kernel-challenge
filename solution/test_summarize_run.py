import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class ArtifactSummaryTests(unittest.TestCase):
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
