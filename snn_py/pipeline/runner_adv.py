"""A Chaos++ friendly concurrent pipeline runner using only the stdlib."""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from snn_py.logging_config import get_logger as _get_logger
except Exception:

    def _get_logger(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger(name)


from snn_py.pipeline.chaos import ChaosConfig, maybe_chaos
from snn_py.util.jsonl import write_jsonl

log = _get_logger("snn_py.pipeline.runner_adv")


def _j(event: str, **meta):
    rec = {"event": event, "ts": time.time(), "meta": meta}
    try:
        log.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        print(json.dumps(rec, ensure_ascii=False))


_SENTINEL = object()


@dataclass
class PipelineResult:
    produced_segments: int = 0
    proposals: int = 0
    audited: int = 0
    errors: int = 0


class PipelineRunner:
    def __init__(
        self,
        out_jsonl: Path,
        theta: float = 0.6,
        max_events: int = 200,
        chaos: Optional[ChaosConfig] = None,
        seed: int = 7,
    ):
        self.out = out_jsonl
        self.theta = theta
        self.max_events = max_events
        self.chaos = chaos or ChaosConfig()
        self.rng = random.Random(seed)
        self.q_seg: queue.Queue[object] = queue.Queue(maxsize=256)
        self.q_prop: queue.Queue[object] = queue.Queue(maxsize=256)
        self.stop = threading.Event()
        self.res = PipelineResult()

    def _producer(self):
        try:
            for i in range(self.max_events):
                if self.stop.is_set():
                    break
                maybe_chaos(self.rng, self.chaos, "producer")
                q = self.rng.random()
                write_jsonl(self.out, {"event": "segment", "meta": {"q": q, "i": i}})
                self.q_seg.put(q, timeout=0.5)
                self.res.produced_segments += 1
            self.q_seg.put(_SENTINEL)
        except Exception as e:
            self.res.errors += 1
            _j("pipeline_error", where="producer", msg=str(e))
            self.stop.set()
            self.q_seg.put(_SENTINEL)

    def _intent(self):
        try:
            while not self.stop.is_set():
                v = self.q_seg.get(timeout=0.5)
                if v is _SENTINEL:
                    self.q_prop.put(_SENTINEL)
                    break
                maybe_chaos(self.rng, self.chaos, "intent")
                if v > self.theta:
                    write_jsonl(self.out, {"event": "proposal", "meta": {"q": v}})
                    self.q_prop.put(True, timeout=0.5)
                    self.res.proposals += 1
                else:
                    self.q_prop.put(False, timeout=0.5)
        except Exception as e:
            self.res.errors += 1
            _j("pipeline_error", where="intent", msg=str(e))
            self.stop.set()
            self.q_prop.put(_SENTINEL)

    def _audit(self):
        try:
            while not self.stop.is_set():
                v = self.q_prop.get(timeout=0.5)
                if v is _SENTINEL:
                    break
                maybe_chaos(self.rng, self.chaos, "audit")
                if v:
                    write_jsonl(self.out, {"event": "audit_decision", "meta": {"status": "APPROVED"}})
                    self.res.audited += 1
        except Exception as e:
            self.res.errors += 1
            _j("pipeline_error", where="audit", msg=str(e))
            self.stop.set()

    def run(self, join_timeout: float = 5.0) -> PipelineResult:
        _j("pipeline_start", theta=self.theta, max_events=self.max_events)
        th1 = threading.Thread(target=self._producer, name="producer", daemon=True)
        th2 = threading.Thread(target=self._intent, name="intent", daemon=True)
        th3 = threading.Thread(target=self._audit, name="audit", daemon=True)
        for t in (th1, th2, th3):
            t.start()
        for t in (th1, th2, th3):
            t.join(join_timeout)
        _j(
            "pipeline_complete",
            produced=self.res.produced_segments,
            proposals=self.res.proposals,
            audited=self.res.audited,
            errors=self.res.errors,
        )
        return self.res
