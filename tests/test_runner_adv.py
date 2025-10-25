from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from snn_py.pipeline.chaos import ChaosConfig
from snn_py.pipeline.runner_adv import PipelineRunner


class RunnerAdvTests(unittest.TestCase):
    def test_runner_no_chaos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "run.jsonl"
            runner = PipelineRunner(out_jsonl=out, theta=0.5, max_events=100, seed=123)
            with self.assertLogs("snn_py.pipeline.runner_adv", level="INFO") as captured:
                res = runner.run()
        log = "\n".join(captured.output)
        self.assertIn('"pipeline_start"', log)
        self.assertIn('"pipeline_complete"', log)
        self.assertEqual(res.produced_segments, 100)
        self.assertEqual(res.proposals, res.audited)

    def test_runner_with_chaos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "run2.jsonl"
            chaos = ChaosConfig(prob_exception=0.2, prob_timeout=0.2, timeout_s=0.01)
            runner = PipelineRunner(out_jsonl=out, theta=0.7, max_events=80, seed=7, chaos=chaos)
            with self.assertLogs("snn_py.pipeline.runner_adv", level="INFO") as captured:
                res = runner.run()
        log = "\n".join(captured.output)
        self.assertIn('"pipeline_complete"', log)
        if '"pipeline_error"' in log:
            self.assertGreaterEqual(res.errors, 1)


if __name__ == "__main__":
    unittest.main()
