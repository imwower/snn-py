from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from snn_py.cli.train_gate import main as train_main
from snn_py.train.threshold_model import ThresholdModel, f1_score


def _emit(dir_path: Path, segments: int = 200, theta_true: float = 0.6, seed: int = 7) -> None:
    rng = random.Random(seed)
    a = dir_path / "a.jsonl"
    b = dir_path / "b.jsonl.gz"

    def write(fp: Path, rows, gz: bool = False) -> None:
        if gz:
            import gzip

            with gzip.open(fp, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            with fp.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    segs = []
    props = []
    for i in range(segments):
        q = rng.random()
        segs.append({"event": "segment", "meta": {"q": q}})
        if q > theta_true:
            props.append({"event": "proposal", "meta": {"from": "segment"}})

    write(a, segs[: segments // 2] + props[: len(props) // 2], gz=False)
    write(b, segs[segments // 2 :] + props[len(props) // 2 :], gz=True)


class TrainingPipelineTests(unittest.TestCase):
    def test_train_gate_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_dir = tmp_path / "episodes"
            data_dir.mkdir()
            _emit(data_dir, segments=300, theta_true=0.65, seed=123)

            out = tmp_path / "model.json"
            rc = train_main(
                ["--jsonl-dir", str(data_dir), "--out", str(out), "--grid", "0.4,0.5,0.6,0.65,0.7"]
            )
            self.assertEqual(rc, 0)
            model = ThresholdModel.load(out)
            self.assertLessEqual(abs(model.theta - 0.65), 0.05)

            qs = [0.2, 0.6, 0.7, 0.9]
            ys = [0, 0, 1, 1]
            preds = [model.predict(q) for q in qs]
            f1 = f1_score(ys, preds)
            self.assertGreaterEqual(f1, 0.8)


if __name__ == "__main__":
    unittest.main()
