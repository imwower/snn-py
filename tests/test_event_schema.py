import logging
import unittest

from snn_py import logging_config
from snn_py.event_schema import EVENT_CODES, validate_log_dict
from tests.utils import record_payload


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)


class EventSchemaTests(unittest.TestCase):
    def tearDown(self) -> None:
        _reset_logging()

    def test_validate_log_dict_success(self) -> None:
        payload = {"event": "metrics_done", "code": EVENT_CODES["metrics_done"], "ts": 123.4, "meta": {"len": 1}}
        self.assertTrue(validate_log_dict(payload))

    def test_validate_log_dict_failure(self) -> None:
        payload = {"event": "metrics_done", "code": "WRONG", "ts": "bad", "meta": {}}
        message = validate_log_dict(payload)
        self.assertIsInstance(message, str)
        self.assertIn("code mismatch", message)

    def test_logging_setup_contains_code(self) -> None:
        _reset_logging()
        with self.assertLogs("snn_py", level="INFO") as captured:
            logging_config.setup()
        self.assertTrue(captured.records)
        payload = record_payload(captured.records[0])
        self.assertEqual(payload["event"], "logging_setup")
        self.assertEqual(payload["code"], EVENT_CODES["logging_setup"])
        self.assertTrue(validate_log_dict(payload))
