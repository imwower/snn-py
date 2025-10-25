import time
import unittest

from snn_py import logging_config
from snn_py.telemetry import Telemetry
from tests.utils import record_payload


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_emits_periodic_events(self) -> None:
        telemetry = Telemetry(interval_s=0.05)
        with self.assertLogs("snn_py.telemetry", level="INFO") as captured:
            telemetry.start()
            time.sleep(0.12)
            telemetry.stop()
        events = [record_payload(record) for record in captured.records]
        telemetry_events = [entry for entry in events if entry.get("event") == "telemetry"]
        self.assertTrue(telemetry_events)
        sample = telemetry_events[0]["meta"]
        self.assertIn("py_alloc_kb", sample)
        self.assertIn("cpu_time_s", sample)


if __name__ == "__main__":
    unittest.main()
