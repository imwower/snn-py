import json
import logging
import os
import tempfile
import unittest

from snn_py import logging_config
from snn_py.cli import demo


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
            with open(policy_path, "w", encoding="utf-8") as fh:
                json.dump(POLICY_TEMPLATE, fh)

            with self.assertLogs("snn_py.cli.demo", level="INFO") as captured:
                exit_code = demo.main([
                    "--T",
                    "3.0",
                    "--policy",
                    policy_path,
                    "--loglevel",
                    "WARNING",
                ])

        self.assertEqual(exit_code, 0)
        events = [json.loads(record.getMessage()) for record in captured.records]
        self.assertTrue(any(entry.get("event") == "run_complete" for entry in events))

