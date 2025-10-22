import json
import logging
import os
import unittest

from snn_py import logging_config
from snn_py.intent import GateConfig, IntentGate


def _reset_logging_state() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)
    manager = logging.Logger.manager
    for name in list(manager.loggerDict.keys()):
        if name.startswith("snn_py"):
            manager.loggerDict.pop(name, None)


class IntentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging_state()
        os.environ.pop("SNN_PY_LOGLEVEL", None)
        logging_config.setup()
        self.addCleanup(_reset_logging_state)
        self.addCleanup(lambda: os.environ.pop("SNN_PY_LOGLEVEL", None))

    def test_spontaneous_fire(self) -> None:
        cfg = GateConfig(
            dt=0.05,
            lam=0.2,
            alpha=0.0,
            sigma=2.5,
            theta=0.8,
            refractory=0.1,
            max_rate_hz=50.0,
        )
        gate = IntentGate(cfg, seed=13)
        fired = False
        steps = int(10.0 / cfg.dt)
        with self.assertLogs("snn_py.intent.gate", level="INFO") as captured:
            for _ in range(steps):
                fired, _ = gate.step()
                if fired:
                    break
        self.assertTrue(fired, "Gate should spontaneously fire with high noise.")
        events = [json.loads(rec.getMessage()) for rec in captured.records]
        self.assertTrue(any(evt["event"] == "意图触发" for evt in events))

    def test_rate_limit_and_refractory(self) -> None:
        cfg = GateConfig(
            dt=0.05,
            lam=0.1,
            alpha=4.0,
            sigma=0.5,
            theta=0.6,
            refractory=0.5,
            max_rate_hz=2.0,
        )
        gate = IntentGate(cfg, seed=5)
        events = []
        times = []
        current_time = 0.0
        with self.assertLogs("snn_py.intent.gate", level="INFO") as captured:
            for _ in range(int(5.0 / cfg.dt)):
                fired, _ = gate.step(q_t=5.0)
                current_time += cfg.dt
                if fired:
                    events.append("意图触发")
                    times.append(current_time)
        self.assertGreaterEqual(len(times), 2, "Expect multiple firings under sustained drive.")
        for prev, nxt in zip(times, times[1:]):
            self.assertGreaterEqual(nxt - prev, 0.5 - 1e-6)
        logged = [json.loads(rec.getMessage()) for rec in captured.records]
        limiter_events = {evt["event"] for evt in logged}
        self.assertTrue({"速率限制", "不应期"} & limiter_events)
