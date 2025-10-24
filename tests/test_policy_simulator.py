import json
import os
import tempfile
import unittest

from snn_py import logging_config
from snn_py.policy.auditor import Auditor, simulate


POLICY = {
    "tools": {
        "allowlist": ["AllowedTool"],
        "guarded": ["GuardedTool"],
        "rules": {
            "GuardedTool": {
                "modes": ["simulate"],
                "require_human_confirm": True,
            }
        },
    }
}


class PolicySimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def _create_auditor(self) -> Auditor:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp.write(json.dumps(POLICY).encode("utf-8"))
        temp.flush()
        temp.close()
        self.addCleanup(lambda: os.remove(temp.name))
        return Auditor(temp.name)

    def test_allowlist(self) -> None:
        auditor = self._create_auditor()
        result = simulate("AllowedTool", {"param": 1}, auditor)
        self.assertEqual(result["status"], "APPROVED")
        self.assertIn("允许列表", result["explain"])

    def test_guarded_requires_human(self) -> None:
        auditor = self._create_auditor()
        result = simulate("GuardedTool", {"mode": "simulate"}, auditor)
        self.assertEqual(result["status"], "REQUIRES_HUMAN")
        self.assertIn("需要人工确认", result["explain"])

    def test_denied(self) -> None:
        auditor = self._create_auditor()
        result = simulate("UnknownTool", {}, auditor)
        self.assertEqual(result["status"], "DENIED")
        self.assertIn("未被授权", result["explain"])


if __name__ == "__main__":
    unittest.main()
