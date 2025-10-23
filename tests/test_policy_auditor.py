import json
import os
import tempfile
import unittest

from snn_py.policy.auditor import AuditResult, Auditor


VALID_POLICY = {
    "tools": {
        "allowlist": ["ProposeAction", "WriteMemory"],
        "guarded": ["ControlHardware"],
        "rules": {
            "ControlHardware": {
                "modes": ["simulate"],
                "require_human_confirm": True,
            }
        },
    }
}


class AuditorTests(unittest.TestCase):
    def _write_policy(self, doc) -> str:
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        json.dump(doc, tmp)
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: os.remove(tmp.name))
        return tmp.name

    def test_allowlist_and_guarded(self) -> None:
        path = self._write_policy(VALID_POLICY)
        auditor = Auditor(path)

        result = auditor.check("ProposeAction", {})
        self.assertEqual(result.status, "APPROVED")

        result = auditor.check("ControlHardware", {"mode": "simulate"})
        self.assertEqual(result.status, "REQUIRES_HUMAN")

        result = auditor.check("ControlHardware", {"mode": "real"})
        self.assertEqual(result.status, "DENIED")

    def test_invalid_policy_structure(self) -> None:
        bad_policy = {"tools": {"allowlist": "oops"}}
        path = self._write_policy(bad_policy)
        with self.assertRaises(ValueError):
            Auditor(path)

