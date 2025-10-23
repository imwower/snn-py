"""多线程管线：片段生产→意向→审计持久化。"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from snn_py import logging_config
from snn_py.memory.episodic import Episode, EpisodicStore
from snn_py.policy.auditor import Auditor


@dataclass
class Segment:
    index: int
    mean_rate: float
    is_up: bool


def _parse_args(args: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="snn_py pipeline runner")
    parser.add_argument("--policy", required=True, help="策略 JSON 文件路径")
    parser.add_argument("--jsonl-dir", default=None, help="情景 JSONL 输出目录")
    parser.add_argument("--max-events", type=int, default=100, help="最大片段数量")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="队列超时时间（秒）")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args(args=args)


def _generate_segments(max_events: int) -> Sequence[Segment]:
    segments = []
    for idx in range(max_events):
        t = idx / max(1, max_events)
        mean_rate = 1.0 + 0.8 * math.sin(2 * math.pi * t) + random.uniform(-0.1, 0.1)
        is_up = mean_rate >= 1.0
        segments.append(Segment(index=idx, mean_rate=mean_rate, is_up=is_up))
    return segments


def run_pipeline(
    policy_path: str,
    jsonl_dir: Optional[Path] = None,
    max_events: int = 100,
    timeout_s: float = 5.0,
) -> Dict[str, int]:
    logger = logging_config.get_logger("snn_py.pipeline.runner")
    store = EpisodicStore(jsonl_dir=jsonl_dir)
    auditor = Auditor(policy_path)

    segments_q: "queue.Queue[Optional[Segment]]" = queue.Queue()
    proposals_q: "queue.Queue[Optional[Dict[str, object]]]" = queue.Queue()

    produced = {"segments": 0}
    consumed = {"proposals": 0}

    stop_event = threading.Event()

    def producer() -> None:
        try:
            for seg in _generate_segments(max_events):
                if stop_event.is_set():
                    break
                segments_q.put(seg)
                produced["segments"] += 1
                logger.info(
                    "",
                    extra={"event": "segment_produced", "meta": {"index": seg.index, "rate": seg.mean_rate}},
                )
        finally:
            segments_q.put(None)

    def intent_worker() -> None:
        while not stop_event.is_set():
            try:
                item = segments_q.get(timeout=timeout_s)
            except queue.Empty:
                stop_event.set()
                break
            if item is None:
                proposals_q.put(None)
                segments_q.task_done()
                break
            score = abs(item.mean_rate - 1.0)
            proposals_q.put({"segment": item.index, "score": score, "rate": item.mean_rate})
            logger.info(
                "",
                extra={
                    "event": "intent_fired",
                    "meta": {"segment": item.index, "score": round(score, 4)},
                },
            )
            segments_q.task_done()

    def consumer() -> None:
        nonlocal stop_event
        while not stop_event.is_set():
            try:
                item = proposals_q.get(timeout=timeout_s)
            except queue.Empty:
                stop_event.set()
                break
            if item is None:
                proposals_q.task_done()
                break
            consumed["proposals"] += 1
            auditor.check("ProposeAction", {"mode": "simulate"})
            episode = Episode(
                t0=time.time(),
                t1=time.time(),
                kind="proposal",
                meta=item,
                payload={"tool": "ProposeAction"},
            )
            store.append(episode)
            proposals_q.task_done()
        store.close()

    threads = [
        threading.Thread(target=producer, name="producer"),
        threading.Thread(target=intent_worker, name="intent"),
        threading.Thread(target=consumer, name="consumer"),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout_s + 5)
    stop_event.set()

    meta = {"produced": produced["segments"], "consumed": consumed["proposals"]}
    logger.info("", extra={"event": "pipeline_complete", "meta": meta})
    return meta


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(args)
    prev_level = os.environ.get("SNN_PY_LOGLEVEL")
    os.environ["SNN_PY_LOGLEVEL"] = parsed.loglevel
    logging_config.setup()
    jsonl_dir = Path(parsed.jsonl_dir) if parsed.jsonl_dir else None
    run_pipeline(
        policy_path=parsed.policy,
        jsonl_dir=jsonl_dir,
        max_events=parsed.max_events,
        timeout_s=parsed.timeout_s,
    )
    if prev_level is None:
        os.environ.pop("SNN_PY_LOGLEVEL", None)
    else:
        os.environ["SNN_PY_LOGLEVEL"] = prev_level
    logging_config.setup()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
