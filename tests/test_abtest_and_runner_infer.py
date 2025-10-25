from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from snn_py.cli.abtest import main as ab_main
from snn_py.cli.train_gate import main as train_main
from snn_py.pipeline.runner_infer import RunnerInfer


def _emit(dir_path: Path, theta_true: float = 0.6, n: int = 200, seed: int = 17) -> None:
    rng = random.Random(seed)
    path = dir_path / "d.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(n):
            q = rng.random()
            handle.write(json.dumps({"event": "segment", "meta": {"q": q}}, ensure_ascii=False) + "\n")
            if q > theta_true:
                handle.write(json.dumps({"event": "proposal"}, ensure_ascii=False) + "\n")


class ABTestAndRunnerInferTests(unittest.TestCase):
    def test_ab_and_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data = tmp_path / "data"
            data.mkdir()
            _emit(data, theta_true=0.6, n=300, seed=3)

            model_a = tmp_path / "A.json"
            model_b = tmp_path / "B.json"
            train_main(["--jsonl-dir", str(data), "--out", str(model_a), "--grid", "0.5,0.55,0.6"])
            train_main(["--jsonl-dir", str(data), "--out", str(model_b), "--grid", "0.65,0.7,0.75"])

            with self.assertLogs("snn_py.cli.abtest", level="INFO") as captured:
                rc = ab_main(
                    ["--jsonl-dir", str(data), "--model-a", str(model_a), "--model-b", str(model_b)]
                )
            self.assertEqual(rc, 0)
            combined_logs = "\n".join(captured.output)
            self.assertIn('"abtest_complete"', combined_logs)

            out_file = tmp_path / "run.jsonl"
            runner = RunnerInfer(out_jsonl=out_file, model_path=model_a, max_events=100)
            res = runner.run()
            self.assertEqual(res.proposals, res.audited)
            self.assertEqual(res.segments, 100)


if __name__ == "__main__":
    unittest.main()
