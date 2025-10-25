from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

try:
    from snn_py import logging_config as _logging_config
except Exception:
    _logging_config = None  # type: ignore[assignment]

from snn_py.train.dataset import load_dataset
from snn_py.train.threshold_model import ThresholdModel

_BASIC_LOG_FORMAT = "%(message)s"


def _ensure_logging() -> None:
    if _logging_config is not None:
        try:
            _logging_config.setup()
            return
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format=_BASIC_LOG_FORMAT)


def _get_logger(name: str) -> logging.Logger:
    _ensure_logging()
    if _logging_config is not None:
        try:
            return _logging_config.get_logger(name)
        except Exception:
            pass
    return logging.getLogger(name)


log = _get_logger("snn_py.cli.abtest")


def _j(event: str, **meta: Dict[str, object]) -> None:
    rec = {"event": event, "ts": time.time(), "meta": meta}
    log.info(json.dumps(rec, ensure_ascii=False))


def bootstrap_diff(a: List[int], b: List[int], iters: int = 200, seed: int = 0) -> float:
    rng = random.Random(seed)

    def mean(xs: List[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    base = abs(mean(a) - mean(b))
    more = 0
    for _ in range(iters):
        ra = [rng.choice(a) for _ in range(len(a))]
        rb = [rng.choice(b) for _ in range(len(b))]
        if abs(mean(ra) - mean(rb)) >= base:
            more += 1
    return (more + 1) / (iters + 1)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser("AB test between two models on same dataset")
    parser.add_argument("--jsonl-dir", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    samples = load_dataset(Path(args.jsonl_dir))
    if not samples:
        _j("abtest_failed", reason="empty_dataset")
        return 1

    model_a = ThresholdModel.load(Path(args.model_a))
    model_b = ThresholdModel.load(Path(args.model_b))
    y_true = [sample.y for sample in samples]
    y_pred_a = [model_a.predict(sample.q) for sample in samples]
    y_pred_b = [model_b.predict(sample.q) for sample in samples]

    hits_a = [int(pred == true) for pred, true in zip(y_pred_a, y_true)]
    hits_b = [int(pred == true) for pred, true in zip(y_pred_b, y_true)]

    acc_a = sum(hits_a) / len(hits_a) if hits_a else 0.0
    acc_b = sum(hits_b) / len(hits_b) if hits_b else 0.0
    p_value = bootstrap_diff(hits_a, hits_b, iters=args.bootstrap_iters, seed=args.seed)
    _j("abtest_complete", acc_a=acc_a, acc_b=acc_b, p_value=p_value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
