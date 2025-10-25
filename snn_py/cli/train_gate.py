from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

try:
    from snn_py.logging_config import get_logger as _get_logger
except Exception:

    def _get_logger(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger(name)


from snn_py.train.dataset import load_dataset, train_val_split
from snn_py.train.threshold_model import ThresholdModel, f1_score, grid_search_theta

log = _get_logger("snn_py.cli.train_gate")


def _j(event: str, **meta) -> None:
    rec = {"event": event, "ts": time.time(), "meta": meta}
    log.info(json.dumps(rec, ensure_ascii=False))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser("train gate threshold from jsonl")
    parser.add_argument("--jsonl-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.jsonl_dir)
    xs = load_dataset(dataset_dir)
    if not xs:
        _j("train_failed", reason="empty_dataset")
        return 1

    train_set, val_set = train_val_split(xs, args.val_ratio)
    if not val_set:
        val_set = train_set

    grid = [float(x.strip()) for x in args.grid.split(",") if x.strip()]
    if not grid:
        raise ValueError("grid must contain at least one value")

    model, best_f1 = grid_search_theta(train_set, val_set, grid)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    y_val = [sample.y for sample in val_set]
    preds = [model.predict(sample.q) for sample in val_set]
    eval_f1 = f1_score(y_val, preds) if val_set else 0.0
    _j("train_complete", samples=len(xs), theta=model.theta, f1=eval_f1, best_f1=best_f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
