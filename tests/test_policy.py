import json
import logging
import os
import tempfile
import unittest

from snn_py import logging_config
from snn_py.policy import AuditResult, Auditor
from tests.utils import record_payload


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)


class AuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()
        logging_config.setup()
        self.addCleanup(_reset_logging)

    def _write_policy(self) -> str:
        policy = {
            "tools": {
                "allowlist": ["SaveParam", "WriteMemory", "ProposeAction"],
                "guarded": ["ControlHardware"],
                "rules": {
                    "ControlHardware": {"modes": ["simulate"], "require_human_confirm": True}
                },
            }
        }
        fp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        json.dump(policy, fp)
        fp.flush()
        fp.close()
        self.addCleanup(lambda: os.remove(fp.name))
        return fp.name

    def test_allowlist_approved(self) -> None:
        path = self._write_policy()
        auditor = Auditor(path)
        with self.assertLogs("snn_py.policy.auditor", level="INFO") as captured:
            result = auditor.check("SaveParam", {})
        self.assertEqual(result.status, "APPROVED")
        events = [record_payload(record) for record in captured.records]
        self.assertTrue(any(entry["event"] == "审计决策" for entry in events))

    def test_guarded_requires_human(self) -> None:
        path = self._write_policy()
        auditor = Auditor(path)
        result = auditor.check("ControlHardware", {"mode": "simulate"})
        self.assertEqual(result.status, "REQUIRES_HUMAN")

    def test_denied_mode(self) -> None:
        path = self._write_policy()
        auditor = Auditor(path)
        result = auditor.check("ControlHardware", {"mode": "real"})
        self.assertEqual(result.status, "DENIED")
