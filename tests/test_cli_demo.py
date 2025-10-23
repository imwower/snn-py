import json
import logging
import os
import tempfile
import unittest

from snn_py import logging_config
from snn_py.cli import demo
from tests.utils import record_payload


POLICY_TEMPLATE = {
    "tools": {
        "allowlist": ["SaveParam", "WriteMemory", "ProposeAction"],
        "guarded": ["ControlHardware"],
        "rules": {
            "ControlHardware": {"modes": ["simulate"], "require_human_confirm": True}
        },
    }
}


class DemoCLITests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_demo_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, "policy.json")
            jsonl_dir = os.path.join(tmpdir, "jsonl")
            os.makedirs(jsonl_dir, exist_ok=True)
            with open(policy_path, "w", encoding="utf-8") as fh:
                json.dump(POLICY_TEMPLATE, fh)

            with self.assertLogs("snn_py", level="INFO") as captured:
                exit_code = demo.main([
                    "--T",
                    "3.0",
                    "--policy",
                    policy_path,
                    "--jsonl-dir",
                    jsonl_dir,
                    "--loglevel",
                    "INFO",
                ])

        self.assertEqual(exit_code, 0)
        events = [record_payload(record) for record in captured.records]
        self.assertTrue(any(entry.get("event") == "run_complete" for entry in events))
        if any(entry.get("event") == "episode_appended" for entry in events):
            self.assertTrue(any(entry.get("event") == "audit_decision" for entry in events))
