import json
import logging
import os
import unittest

from snn_py import logging_config


def _record_payload(record: logging.LogRecord) -> dict:
    formatter = logging_config.JSONLineFormatter(getattr(record, "run", {}))
    return json.loads(formatter.format(record))


def _reset_logging_state() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)
    manager = logging.Logger.manager
    for name in list(manager.loggerDict.keys()):
        if name == "snn_py" or name.startswith("snn_py."):
            manager.loggerDict.pop(name, None)


class LoggingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging_state()
        os.environ.pop("SNN_PY_LOGLEVEL", None)
        self.addCleanup(_reset_logging_state)
        self.addCleanup(lambda: os.environ.pop("SNN_PY_LOGLEVEL", None))

    def test_setup_default_info(self) -> None:
        with self.assertLogs("snn_py", level="INFO") as captured:
            logging_config.setup()
            self.assertEqual(logging.getLogger().level, logging.INFO)
            self.assertEqual(logging.getLogger("snn_py").level, logging.INFO)
        self.assertGreaterEqual(len(captured.records), 1)
        payload = _record_payload(captured.records[-1])
        self.assertEqual(payload["event"], "logging_setup")
        self.assertEqual(payload["meta"], {"level": "INFO"})
        self.assertIsInstance(payload["ts"], float)
        self.assertEqual(logging.getLogger("snn_py").getEffectiveLevel(), logging.INFO)

    def test_setup_debug_from_env(self) -> None:
        os.environ["SNN_PY_LOGLEVEL"] = "DEBUG"
        with self.assertLogs("snn_py", level="DEBUG") as captured:
            logging_config.setup()
            self.assertEqual(logging.getLogger().level, logging.DEBUG)
            self.assertEqual(logging.getLogger("snn_py").level, logging.DEBUG)
        payload = _record_payload(captured.records[-1])
        self.assertEqual(payload["meta"]["level"], "DEBUG")
        self.assertEqual(logging.getLogger().level, logging.DEBUG)
        self.assertEqual(logging.getLogger("snn_py").getEffectiveLevel(), logging.DEBUG)

    def test_setup_is_idempotent(self) -> None:
        logging_config.setup()
        root = logging.getLogger()
        handler_ids = [id(handler) for handler in root.handlers]

        with self.assertLogs("snn_py", level="INFO"):
            logging_config.setup()

        self.assertEqual(handler_ids, [id(handler) for handler in logging.getLogger().handlers])
