import logging
import unittest

from snn_py.errors import ConfigError, IOError_, PolicyError, SnnPyError, log_error, to_exit_code


class ErrorTests(unittest.TestCase):
    def test_exit_code_mapping(self) -> None:
        self.assertEqual(to_exit_code(PolicyError("policy")), 2)
        self.assertEqual(to_exit_code(ConfigError("config")), 3)
        self.assertEqual(to_exit_code(IOError_("io")), 4)
        self.assertEqual(to_exit_code(ValueError("other")), 1)

    def test_log_error(self) -> None:
        with self.assertLogs("snn_py", level="ERROR") as captured:
            log_error(PolicyError("bad policy"))
        record = captured.records[-1]
        self.assertEqual(record.levelno, logging.ERROR)
        self.assertIn("error", record.__dict__.get("event", "error"))

