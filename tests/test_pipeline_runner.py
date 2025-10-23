import json
import os
import tempfile
import unittest
from pathlib import Path

from snn_py import logging_config
from snn_py.pipeline.runner import main, run_pipeline


POLICY = {
    "tools": {
        "allowlist": ["ProposeAction"],
        "guarded": [],
        "rules": {},
    }
}


class PipelineRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def _write_policy(self, tmpdir: str) -> str:
        path = os.path.join(tmpdir, "policy.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(POLICY, fh)
        return path

    def test_pipeline_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self._write_policy(tmpdir)
            jsonl_dir = os.path.join(tmpdir, "episodes")
            os.makedirs(jsonl_dir, exist_ok=True)
            meta = run_pipeline(policy_path=policy_path, jsonl_dir=Path(jsonl_dir), max_events=50, timeout_s=1.0)
            self.assertEqual(meta["produced"], 50)
            self.assertEqual(meta["consumed"], 50)

    def test_cli_timeout_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self._write_policy(tmpdir)
            exit_code = main([
                "--policy",
                policy_path,
                "--max-events",
                "5",
                "--timeout-s",
                "0.1",
                "--loglevel",
                "INFO",
            ])
            self.assertEqual(exit_code, 0)
