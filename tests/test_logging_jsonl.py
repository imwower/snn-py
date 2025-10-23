import importlib
import json
import logging
import unittest

from snn_py import logging_config


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.NOTSET)


class JSONLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()
        importlib.reload(logging_config)

    def test_json_log_line(self) -> None:
        logging_config.setup()
        logger = logging_config.get_logger("tests.json")
        with self.assertLogs("snn_py.tests.json", level="INFO") as captured:
            logger.info("test message", extra={"event": "unit_event", "meta": {"k": 1}})

        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        formatter = logging_config.JSONLineFormatter(getattr(record, "run"))
        payload = json.loads(formatter.format(record))
        self.assertIn("run", payload)
        self.assertIn("run_id", payload["run"])
        self.assertEqual(payload["event"], "unit_event")

    def test_no_duplicate_handlers(self) -> None:
        logging_config.setup()
        root = logging.getLogger()
        initial_handlers = len(root.handlers)
        logging_config.setup()
        self.assertEqual(len(root.handlers), initial_handlers)
        logger = logging_config.get_logger("tests.dup")
        with self.assertLogs("snn_py.tests.dup", level="INFO") as captured:
            logger.info("only once", extra={"event": "dup", "meta": {}})
        self.assertEqual(len(captured.records), 1)

