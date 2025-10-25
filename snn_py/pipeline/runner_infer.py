from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from snn_py import logging_config as _logging_config
except Exception:
    _logging_config = None  # type: ignore[assignment]

from snn_py.pipeline.infer_gate import GateInfer
from snn_py.util.jsonl import write_jsonl

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


log = _get_logger("snn_py.pipeline.runner_infer")
_SENTINEL = object()


@dataclass
class Result:
    segments: int = 0
    proposals: int = 0
    audited: int = 0


class RunnerInfer:
    def __init__(self, out_jsonl: Path, model_path: Path, max_events: int = 200, seed: int = 7):
        self.out = out_jsonl
        self.infer = GateInfer(model_path)
        self.max_events = max_events
        self.rng = random.Random(seed)
        self.q_seg: queue.Queue[object] = queue.Queue(maxsize=256)
        self.q_prop: queue.Queue[object] = queue.Queue(maxsize=256)

    def _j(self, event: str, **meta) -> None:
        rec = {"event": event, "ts": time.time(), "meta": meta}
        log.info(json.dumps(rec, ensure_ascii=False))

    def _producer(self, res: Result) -> None:
        for _ in range(self.max_events):
            q = self.rng.random()
            write_jsonl(self.out, {"event": "segment", "meta": {"q": q}})
            self.q_seg.put(q)
            res.segments += 1
        self.q_seg.put(_SENTINEL)

    def _intent(self, res: Result) -> None:
        while True:
            value = self.q_seg.get()
            if value is _SENTINEL:
                self.q_prop.put(_SENTINEL)
                break
            decision = self.infer.model.predict(value)
            if decision == 1:
                write_jsonl(self.out, {"event": "proposal", "meta": {"q": value}})
                self.q_prop.put(True)
                res.proposals += 1
            else:
                self.q_prop.put(False)

    def _audit(self, res: Result) -> None:
        while True:
            value = self.q_prop.get()
            if value is _SENTINEL:
                break
            if value:
                write_jsonl(self.out, {"event": "audit_decision", "meta": {"status": "APPROVED"}})
                res.audited += 1

    def run(self) -> Result:
        self._j("pipeline_start", mode="infer")
        res = Result()
        threads = [
            threading.Thread(target=self._producer, args=(res,), daemon=True),
            threading.Thread(target=self._intent, args=(res,), daemon=True),
            threading.Thread(target=self._audit, args=(res,), daemon=True),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._j("pipeline_complete", segments=res.segments, proposals=res.proposals, audited=res.audited)
        return res


__all__ = ["RunnerInfer", "Result"]
