"""演示 CLI：合成群体率→意向门→审计→情景记忆。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from snn_py import logging_config
from snn_py.intent.scoring import NoveltyScorer
from snn_py.memory.episodic import Episode, EpisodicStore
from snn_py.policy.auditor import Auditor


def _parse_args(args: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="snn_py demo run")
    parser.add_argument("--T", type=float, default=8.0, help="模拟时长（秒）")
    parser.add_argument("--policy", required=True, help="策略 JSON 文件")
    parser.add_argument("--jsonl-dir", default=None, help="情景落盘目录")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args(args=args)


def _synth_rate(duration: float, dt: float) -> List[float]:
    steps = max(1, int(duration / dt))
    rates: List[float] = []
    for i in range(steps):
        t = i * dt
        value = 1.0 + 0.6 * math.sin(2.0 * math.pi * t / max(duration, dt))
        value += 0.2 * math.sin(6.0 * math.pi * t / max(duration, dt))
        value += random.uniform(-0.05, 0.05)
        rates.append(max(0.0, value))
    return rates


def _segments(rates: Sequence[float], thr_low: float, thr_high: float) -> List[Tuple[int, int, bool]]:
    logger = logging_config.get_logger("snn_py.cli.demo")
    segments: List[Tuple[int, int, bool]] = []
    start = 0
    state_up = False
    for idx, value in enumerate(rates):
        if not state_up and value >= thr_high:
            if idx > start:
                segments.append((start, idx, False))
            state_up = True
            start = idx
            logger.info("", extra={"event": "up_start", "meta": {"index": idx, "rate": value}})
        elif state_up and value <= thr_low:
            if idx > start:
                segments.append((start, idx, True))
            state_up = False
            start = idx
            logger.info("", extra={"event": "down_start", "meta": {"index": idx, "rate": value}})
    end_idx = len(rates)
    if end_idx > start:
        segments.append((start, end_idx, state_up))
    return segments


class SimpleIntentGate:
    def __init__(self, dt: float, threshold: float = 0.9, decay: float = 0.7, refractory: float = 0.15) -> None:
        self._dt = dt
        self._threshold = threshold
        self._decay = decay
        self._refrac_steps = max(1, int(refractory / dt))
        self._refrac = 0
        self._potential = 0.0
        self._time = 0.0

    def step(self, drive: float) -> Tuple[bool, float]:
        fired = False
        if self._refrac > 0:
            self._refrac -= 1
        else:
            self._potential = self._potential * self._decay + drive
            if self._potential >= self._threshold:
                fired = True
                self._potential = 0.0
                self._refrac = self._refrac_steps
        self._time += self._dt
        return fired, self._time


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(args)

    previous_level = os.environ.get("SNN_PY_LOGLEVEL")
    os.environ["SNN_PY_LOGLEVEL"] = parsed.loglevel
    logging_config.setup()
    logger = logging_config.get_logger("snn_py.cli.demo")

    jsonl_dir = Path(parsed.jsonl_dir) if parsed.jsonl_dir else None
    store = EpisodicStore(jsonl_dir=jsonl_dir)
    auditor = Auditor(parsed.policy)
    scorer = NoveltyScorer()

    dt = 0.05
    rates = _synth_rate(parsed.T, dt)
    global_mean = sum(rates) / len(rates)
    thr_low = global_mean - 0.1
    thr_high = global_mean + 0.1
    segs = _segments(rates, thr_low, thr_high)

    gate = SimpleIntentGate(dt=dt, threshold=0.85, decay=0.65, refractory=0.1)
    episode_count = 0

    for idx, (start, end, is_up) in enumerate(segs):
        segment_rates = rates[start:end]
        if not segment_rates:
            continue
        mean_rate = sum(segment_rates) / len(segment_rates)
        q = scorer.score(mean_rate)
        drive = q if is_up else 0.5 * q

        for _ in segment_rates:
            fired, timestamp = gate.step(drive)
            if not fired:
                continue
            logger.info(
                "",
                extra={
                    "event": "intent_fired",
                    "meta": {"t": round(timestamp, 3), "q": drive, "segment": idx, "up": is_up},
                },
            )
            proposal = {"tool": "ProposeAction", "args": {"note": "spontaneous"}}
            episode = Episode(
                t0=timestamp,
                t1=timestamp,
                kind="proposal",
                meta={"segment": idx, "up": is_up, "q": drive},
                payload=proposal,
            )
            store.append(episode)
            episode_count += 1
            auditor.check(proposal["tool"], proposal["args"])

    store.close()
    logger.info("", extra={"event": "run_complete", "meta": {"episodes": episode_count}})

    if previous_level is None:
        os.environ.pop("SNN_PY_LOGLEVEL", None)
    else:
        os.environ["SNN_PY_LOGLEVEL"] = previous_level
    logging_config.setup()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

