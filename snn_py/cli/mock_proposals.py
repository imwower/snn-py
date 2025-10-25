"""Emit synthetic proposal events into a JSONL stream."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Optional

try:
    from snn_py import logging_config as _logging_config
except Exception:  # pragma: no cover - fallback
    _logging_config = None  # type: ignore[assignment]

_BASIC_LOG_FORMAT = "%(message)s"


def _ensure_logging() -> None:
    if _logging_config is not None:
        try:
            _logging_config.setup()
            return
        except Exception:  # pragma: no cover - fallback
            pass
    logging.basicConfig(level=logging.INFO, format=_BASIC_LOG_FORMAT)


def _get_logger(name: str) -> logging.Logger:
    _ensure_logging()
    if _logging_config is not None:
        try:
            return _logging_config.get_logger(name)
        except Exception:  # pragma: no cover - fallback
            pass
    return logging.getLogger(name)


log = _get_logger("snn_py.cli.mock_proposals")


def _j(event: str, **meta: object) -> None:
    rec = {"event": event, "ts": time.time(), "meta": meta}
    log.info(json.dumps(rec, ensure_ascii=False))


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate synthetic proposal events.")
    parser.add_argument("--out", required=True, help="Output proposals JSONL file.")
    parser.add_argument("--count", type=int, default=10, help="Number of events to emit.")
    parser.add_argument("--interval-ms", type=int, default=0, help="Sleep between events.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for q sampling.")
    parser.add_argument("--q-min", type=float, default=0.1, help="Minimum q value.")
    parser.add_argument("--q-max", type=float, default=0.95, help="Maximum q value.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    count = max(0, args.count)
    if count == 0:
        _j("mock_proposals_skip", reason="zero_count")
        return 0

    q_min = min(args.q_min, args.q_max)
    q_max = max(args.q_min, args.q_max)
    interval = max(0, args.interval_ms) / 1000.0
    rng = random.Random(args.seed)

    _j("mock_proposals_start", out=str(out_path), count=count)
    for idx in range(count):
        q = rng.uniform(q_min, q_max)
        record = {"event": "proposal", "ts": time.time(), "meta": {"q": round(q, 3)}}
        _append(out_path, record)
        _j("proposal_written", index=idx, q=record["meta"]["q"])
        if interval and idx + 1 < count:
            time.sleep(interval)

    _j("mock_proposals_complete", out=str(out_path), count=count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

