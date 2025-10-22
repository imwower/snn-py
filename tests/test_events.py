import json
import logging
import random
import unittest

from snn_py import logging_config
from snn_py.core import detect_avalanches, detect_up_down


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)


class EventDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()
        logging_config.setup()
        self.addCleanup(_reset_logging)

    def test_updown_synthetic(self) -> None:
        rates = [0.5] * 50 + [2.0] * 60 + [0.4] * 40
        with self.assertLogs("snn_py.core.events", level="INFO") as captured:
            segments = detect_up_down(rates, thr_low=0.8, thr_high=1.2)
        self.assertGreaterEqual(len(segments), 3)
        _, _, middle_state = segments[1]
        self.assertTrue(middle_state)
        entries = [json.loads(rec.getMessage()) for rec in captured.records]
        self.assertTrue(any(entry["event"] == "上状态开始" for entry in entries))

    def test_avalanche_bins(self) -> None:
        rng = random.Random(91)
        time_steps = 120
        neurons = 8
        spikes = [[1 if rng.random() < 0.05 else 0 for _ in range(neurons)] for _ in range(time_steps)]
        for t in range(40, 45):
            for n in range(3):
                spikes[t][n] = 1
        with self.assertLogs("snn_py.core.events", level="INFO") as captured:
            avalanches = detect_avalanches(spikes)
        for avalanche in avalanches:
            self.assertGreater(avalanche["size"], 0)
            self.assertGreater(avalanche["duration_bins"], 0)
        if avalanches:
            entries = [json.loads(rec.getMessage()) for rec in captured.records]
            self.assertTrue(any(entry["event"] == "神经雪崩" for entry in entries))
