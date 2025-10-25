from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from snn_py.cli.eval_gate import main as eval_main
from snn_py.cli.train_gate import main as train_main
from snn_py.pipeline.infer_gate import GateInfer


def _emit(dir_path: Path, theta_true: float = 0.6, n: int = 200, seed: int = 11) -> None:
    rng = random.Random(seed)
    path = dir_path / "d.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(n):
            q = rng.random()
            handle.write(json.dumps({"event": "segment", "meta": {"q": q}}, ensure_ascii=False) + "\n")
            if q > theta_true:
                handle.write(json.dumps({"event": "proposal"}, ensure_ascii=False) + "\n")


class EvalAndInferTests(unittest.TestCase):
    def test_eval_and_infer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_dir = tmp_path / "train"
            val_dir = tmp_path / "val"
            train_dir.mkdir()
            val_dir.mkdir()
            _emit(train_dir, theta_true=0.55, n=300, seed=1)
            _emit(val_dir, theta_true=0.55, n=200, seed=2)

            model_path = tmp_path / "model.json"
            rc_train = train_main(
                [
                    "--jsonl-dir",
                    str(train_dir),
                    "--out",
                    str(model_path),
                    "--grid",
                    "0.4,0.5,0.55,0.6,0.65",
                ]
            )
            self.assertEqual(rc_train, 0)

            with self.assertLogs("snn_py.cli.eval_gate", level="INFO") as captured:
                rc_eval = eval_main(["--jsonl-dir", str(val_dir), "--model", str(model_path)])
            self.assertEqual(rc_eval, 0)
            combined_logs = "\n".join(captured.output)
            self.assertIn('"eval_complete"', combined_logs)

            infer = GateInfer(model_path)
            preds = infer.batch([0.1, 0.56, 0.9])
            self.assertEqual(preds, [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
