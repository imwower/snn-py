"""Replay++ CLI for expectation diffing and time-travel recompute."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    from snn_py import logging_config as _logging_config
    _get_logger = _logging_config.get_logger
except Exception:
    _logging_config = None

    def _get_logger(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger(name)

from snn_py.util.jsonl import iter_jsonl, scan_jsonl_dir

log = _get_logger("snn_py.cli.replay_plus")


def _j(event: str, **meta: Any) -> None:
    rec = {"event": event, "ts": time.time(), "meta": meta}
    try:
        log.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        print(json.dumps(rec, ensure_ascii=False))


def load_events(dir_path: Path) -> List[Dict[str, Any]]:
    files = list(scan_jsonl_dir(dir_path))
    total = 0
    events: List[Dict[str, Any]] = []
    for fp in files:
        for data in iter_jsonl(fp):
            events.append(data)
            total += 1
    _j("replay_loaded", files=len(files), events=total)
    return events


def summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    cnt = {"proposal": 0, "audit_decision": 0, "segment": 0}
    by_status: Dict[str, int] = {}
    for event in events:
        ev = event.get("event")
        if ev in cnt:
            cnt[ev] += 1
        if ev == "audit_decision":
            status = (event.get("meta") or {}).get("status", "NA")
            by_status[status] = by_status.get(status, 0) + 1
    meta = {"counts": cnt, "by_status": by_status}
    _j("replay_summary", **meta)
    return meta


def recompute_proposals(events: List[Dict[str, Any]], theta: float) -> Tuple[int, int, int]:
    """
    基于 segment 事件中的 q 值和阈值 theta 复算 proposal 数。
    返回: (segments, proposals_pred, proposals_actual)
    """
    seg_q: List[float] = []
    actual = 0
    for event in events:
        if event.get("event") == "segment":
            q = (event.get("meta") or {}).get("q")
            if isinstance(q, (int, float)):
                seg_q.append(float(q))
        elif event.get("event") == "proposal":
            actual += 1
    predicted = sum(1 for q in seg_q if q is not None and q > theta)
    _j("replay_recomputed", segments=len(seg_q), predicted=predicted, actual=actual, theta=theta)
    return len(seg_q), predicted, actual


def diff_expect(expect_path: Path, summary_meta: Dict[str, Any]) -> Dict[str, Any]:
    exp = json.loads(expect_path.read_text(encoding="utf-8"))
    mismatch: Dict[str, Dict[str, Any]] = {}
    exp_counts = exp.get("counts", {})
    got_counts = summary_meta.get("counts", {})
    for key, expect_value in exp_counts.items():
        if got_counts.get(key) != expect_value:
            mismatch[key] = {"expect": expect_value, "actual": got_counts.get(key)}
    _j("replay_diff", mismatch=mismatch)
    return mismatch


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay++: expectation diff & time-travel recompute")
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--expect", default=None, help="path to expect.json")
    ap.add_argument("--time-travel", action="store_true", help="recompute proposals by segments & theta")
    ap.add_argument("--theta", type=float, default=None, help="threshold for time-travel")
    args = ap.parse_args(argv)

    if _logging_config is not None:
        _logging_config.setup()

    events = load_events(Path(args.jsonl_dir))
    sm = summary(events)
    rc = 0

    if args.expect:
        mismatch = diff_expect(Path(args.expect), sm)
        if mismatch:
            rc = 1

    if args.time_travel:
        if args.theta is None:
            _j("error", msg="theta required for time-travel")
            return 2
        segments, predicted, actual = recompute_proposals(events, args.theta)
        match_rate = (predicted == 0 and actual == 0) and 1.0 or (min(predicted, actual) / max(predicted, actual))
        _j("replay_match_rate", segments=segments, predicted=predicted, actual=actual, match_rate=match_rate)
        if match_rate < 0.9:
            rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
