import json
import os
import tempfile
import unittest

from snn_py.cli import demo
from snn_py.policy.auditor import Auditor


POLICY = {
    "tools": {
        "allowlist": ["ProposeAction"],
        "guarded": [],
        "rules": {},
    }
}


class ProfileTraceTests(unittest.TestCase):
    def test_profile_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, "policy.json")
            with open(policy_path, "w", encoding="utf-8") as fh:
                json.dump(POLICY, fh)

            profile_path = os.path.join(tmpdir, "out.prof")
            trace_dir = os.path.join(tmpdir, "trace")

            exit_code = demo.main([
                "--T",
                "1.0",
                "--policy",
                policy_path,
                "--profile",
                profile_path,
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(profile_path))

            exit_code = demo.main([
                "--T",
                "1.0",
                "--policy",
                policy_path,
                "--trace-coverage",
                trace_dir,
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isdir(trace_dir))

