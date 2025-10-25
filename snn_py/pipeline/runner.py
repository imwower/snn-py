"""多线程意图管线：采样→意图→审计持久化。"""

from __future__ import annotations

import argparse
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from snn_py import logging_config
from snn_py.intent.gate import GateConfig, IntentGate
from snn_py.intent.scoring import NoveltyScorer
from snn_py.memory.episodic import Episode, EpisodicStore
from snn_py.policy.auditor import Auditor
from snn_py.seed import SeedManager

logger = logging_config.get_logger("snn_py.pipeline.runner")


@dataclass(frozen=True)
class Segment:
    """上/下状态片段。"""

    index: int
    t0: float
    t1: float
    mean_rate: float
    is_up: bool


def _parse_args(args: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="snn_py pipeline runner")
    parser.add_argument("--policy", required=True, help="策略 JSON 文件路径")
    parser.add_argument("--jsonl-dir", default=None, help="情景 JSONL 输出目录")
    parser.add_argument("--max-events", type=int, default=100, help="最大片段数量")
    parser.add_argument("--timeout-s", type=float, default=2.0, help="队列超时时间（秒）")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args(args=args)


def _seed_base(run_id: str, env: Mapping[str, str]) -> int:
    seed_str = env.get("SNN_PY_SEED")
    if seed_str is not None:
        try:
            return int(seed_str)
        except ValueError:
            pass
    try:
        return int(run_id[:8], 16) & 0xFFFFFFFF
    except ValueError:
        return hash(run_id) & 0xFFFFFFFF


def _synth_rate(num_steps: int, dt: float, rng: Random) -> List[float]:
    """生成简单的上下交替速率轨迹。"""
    rates: List[float] = []
    for idx in range(num_steps):
        phase = math.sin(2.0 * math.pi * idx / max(1, num_steps))
        base = 1.0 + 0.25 * phase
        if idx % 2 == 0:
            value = base + 0.2
        else:
            value = base - 0.2
        value += rng.uniform(-0.05, 0.05)
        rates.append(value)
    return rates


def _segments(rates: Sequence[float], thr_low: float, thr_high: float) -> List[Tuple[int, int, bool]]:
    """借鉴 demo._segments 的分段逻辑。"""
    segments: List[Tuple[int, int, bool]] = []
    start = 0
    state_up = False
    for idx, value in enumerate(rates):
        if not state_up and value >= thr_high:
            if idx > start:
                segments.append((start, idx, False))
            state_up = True
            start = idx
        elif state_up and value <= thr_low:
            if idx > start:
                segments.append((start, idx, True))
            state_up = False
            start = idx
    end_idx = len(rates)
    if end_idx > start:
        segments.append((start, end_idx, state_up))
    return segments


class PipelineRunner:
    """基于 threading + queue 的最小并发流水线。"""

    def __init__(self, policy_path: str, jsonl_dir: Optional[Path] = None, max_events: int = 100, timeout_s: float = 2.0) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self._policy_path = policy_path
        self._jsonl_dir = jsonl_dir
        self._max_events = max_events
        self._timeout_s = timeout_s
        self._dt = 0.05
        self._run_id = uuid4().hex
        seed_base = _seed_base(self._run_id, os.environ)
        self._seed_manager = SeedManager(seed_base)
        self._segments_rng = self._seed_manager.rng("segments")
        gate_seed = self._seed_manager.seed_for("gate")
        logger.info("", extra={"event": "seed_streams_ready", "meta": {"streams": sorted(self._seed_manager.describe().keys())}})

        self._segment_queue: "queue.Queue[object]" = queue.Queue(maxsize=16)
        self._proposal_queue: "queue.Queue[object]" = queue.Queue(maxsize=16)
        self._result_queue: "queue.Queue[object]" = queue.Queue(maxsize=4)
        self._sentinel = object()

        self._produced = 0
        self._consumed = 0

        self._store = EpisodicStore(jsonl_dir=self._jsonl_dir)
        self._auditor = Auditor(self._policy_path)
        self._scorer = NoveltyScorer()
        gate_cfg = GateConfig(
            dt=self._dt,
            lam=0.0,
            alpha=1.0,
            sigma=0.0,
            theta=0.0,
            refractory=0.0,
            max_rate_hz=0.0,
        )
        self._gate = IntentGate(gate_cfg, seed=gate_seed)

    def run(self) -> Dict[str, int]:
        logger.info("", extra={"event": "pipeline_start"})

        threads = [
            threading.Thread(target=self._producer, name="pipeline-producer", daemon=True),
            threading.Thread(target=self._intent_worker, name="pipeline-intent", daemon=True),
            threading.Thread(target=self._audit_worker, name="pipeline-audit", daemon=True),
        ]

        for worker in threads:
            worker.start()

        self._wait_for_completion()

        for worker in threads:
            worker.join(timeout=self._timeout_s + 1.0)

        meta = {"produced": self._produced, "consumed": self._consumed}
        logger.info("", extra={"event": "pipeline_complete", "meta": meta})
        return meta

    def _wait_for_completion(self) -> None:
        while True:
            try:
                item = self._result_queue.get(timeout=self._timeout_s)
            except queue.Empty:
                continue
            if item is self._sentinel:
                self._result_queue.task_done()
                break
            self._result_queue.task_done()

    def _producer(self) -> None:
        rates = _synth_rate(self._max_events, self._dt, self._segments_rng)
        if not rates:
            self._put(self._segment_queue, self._sentinel)
            return

        global_mean = sum(rates) / len(rates)
        thr_low = global_mean - 0.05
        thr_high = global_mean + 0.05
        segments = _segments(rates, thr_low, thr_high)

        produced = 0
        for start, end, is_up in segments:
            if produced >= self._max_events:
                break
            segment_rates = rates[start:end]
            if not segment_rates:
                continue
            mean_rate = sum(segment_rates) / len(segment_rates)
            segment = Segment(
                index=produced,
                t0=start * self._dt,
                t1=end * self._dt,
                mean_rate=mean_rate,
                is_up=is_up,
            )
            self._put(self._segment_queue, segment)
            produced += 1

        while produced < self._max_events:
            segment = Segment(
                index=produced,
                t0=produced * self._dt,
                t1=(produced + 1) * self._dt,
                mean_rate=rates[-1],
                is_up=rates[-1] >= thr_high,
            )
            self._put(self._segment_queue, segment)
            produced += 1

        self._put(self._segment_queue, self._sentinel)

    def _intent_worker(self) -> None:
        gate = self._gate
        scorer = self._scorer
        events_sent = 0

        while True:
            item = self._get(self._segment_queue)
            if item is self._sentinel:
                self._segment_queue.task_done()
                self._put(self._proposal_queue, self._sentinel)
                break

            assert isinstance(item, Segment)
            score = scorer.score(item.mean_rate)
            fired, _ = gate.step(score)
            if fired:
                proposal = {
                    "segment": item.index,
                    "mean_rate": item.mean_rate,
                    "score": score,
                    "is_up": item.is_up,
                    "t0": item.t0,
                    "t1": item.t1,
                }
                self._put(self._proposal_queue, proposal)
                events_sent += 1
                self._produced = events_sent
            self._segment_queue.task_done()

    def _audit_worker(self) -> None:
        consumed = 0
        while True:
            item = self._get(self._proposal_queue)
            if item is self._sentinel:
                self._proposal_queue.task_done()
                self._store.close()
                self._put(self._result_queue, {"consumed": consumed})
                self._put(self._result_queue, self._sentinel)
                break

            assert isinstance(item, dict)
            args = {"mode": "autonomous", "score": item["score"]}
            self._auditor.check("ProposeAction", args)
            timestamp = time.time()
            episode = Episode(
                t0=item["t0"],
                t1=item["t1"],
                kind="proposal",
                meta={"segment": item["segment"], "is_up": item["is_up"], "score": item["score"]},
                payload={"tool": "ProposeAction", "args": args, "ts": timestamp},
            )
            self._store.append(episode)
            consumed += 1
            self._consumed = consumed
            self._proposal_queue.task_done()

    def _put(self, q: "queue.Queue[object]", item: object) -> None:
        while True:
            try:
                q.put(item, timeout=self._timeout_s)
                return
            except queue.Full:
                continue

    def _get(self, q: "queue.Queue[object]") -> object:
        while True:
            try:
                return q.get(timeout=self._timeout_s)
            except queue.Empty:
                continue


def run_pipeline(
    policy_path: str,
    jsonl_dir: Optional[Path] = None,
    max_events: int = 100,
    timeout_s: float = 2.0,
) -> Dict[str, int]:
    runner = PipelineRunner(policy_path=policy_path, jsonl_dir=jsonl_dir, max_events=max_events, timeout_s=timeout_s)
    return runner.run()


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(args)
    previous_level = os.environ.get("SNN_PY_LOGLEVEL")
    os.environ["SNN_PY_LOGLEVEL"] = parsed.loglevel
    logging_config.setup()

    jsonl_dir = Path(parsed.jsonl_dir) if parsed.jsonl_dir else None

    run_pipeline(
        policy_path=parsed.policy,
        jsonl_dir=jsonl_dir,
        max_events=parsed.max_events,
        timeout_s=parsed.timeout_s,
    )

    if previous_level is None:
        os.environ.pop("SNN_PY_LOGLEVEL", None)
    else:
        os.environ["SNN_PY_LOGLEVEL"] = previous_level
    logging_config.setup()
    return 0


__all__ = ["PipelineRunner", "run_pipeline", "main"]
