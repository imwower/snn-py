from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

try:
    from snn_py import logging_config as _logging_config
except Exception:
    _logging_config = None  # type: ignore[assignment]

from snn_py.train.dataset import load_dataset
from snn_py.train.threshold_model import ThresholdModel, f1_score

_BASIC_LOG_FORMAT = "%(message)s"
_LOGGER: logging.Logger | None = None


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
    logging.basicConfig(level=logging.INFO, format=_BASIC_LOG_FORMAT)
    return logging.getLogger(name)


def _logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = _get_logger("snn_py.cli.eval_gate")
    return _LOGGER


def _j(event: str, **meta):
    rec = {"event": event, "ts": time.time(), "meta": meta}
    _logger().info(json.dumps(rec, ensure_ascii=False))


def _confusion(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return tp, tn, fp, fn


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser("evaluate gate model")
    parser.add_argument("--jsonl-dir", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    samples = load_dataset(Path(args.jsonl_dir))
    if not samples:
        _j("eval_failed", reason="empty_dataset")
        return 1

    model = ThresholdModel.load(Path(args.model))
    y_true = [sample.y for sample in samples]
    y_pred = [model.predict(sample.q) for sample in samples]
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    _j(
        "eval_complete",
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        theta=model.theta,
        samples=len(samples),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
