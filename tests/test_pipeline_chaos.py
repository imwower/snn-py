import json
import os
import tempfile
import unittest
from pathlib import Path

from snn_py import logging_config
from snn_py.pipeline.runner import PipelineRunner, run_pipeline
from tests.utils import record_payload

POLICY = {
    "tools": {
        "allowlist": ["ProposeAction"],
        "guarded": [],
        "rules": {},
    }
}


class PipelineChaosTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def _write_policy(self, tmpdir: str) -> str:
        path = Path(tmpdir) / "policy.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(POLICY, handle)
        return str(path)

    def test_pipeline_reports_errors_under_chaos(self) -> None:
        previous_seed = os.environ.get("SNN_PY_SEED")
        os.environ["SNN_PY_SEED"] = "123"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                policy_path = self._write_policy(tmpdir)
                jsonl_dir = Path(tmpdir) / "episodes"
                jsonl_dir.mkdir(parents=True, exist_ok=True)
                runner = PipelineRunner(
                    policy_path=policy_path,
                    jsonl_dir=jsonl_dir,
                    max_events=20,
                    timeout_s=0.05,
                    chaos_prob=0.2,
                )
                with self.assertLogs("snn_py.pipeline.runner", level="INFO") as captured:
                    meta = runner.run()
        finally:
            if previous_seed is None:
                os.environ.pop("SNN_PY_SEED", None)
            else:
                os.environ["SNN_PY_SEED"] = previous_seed

        events = [record_payload(record) for record in captured.records]
        errors = [entry for entry in events if entry.get("event") == "pipeline_error"]
        self.assertTrue(errors)
        self.assertIn("pipeline_complete", [entry.get("event") for entry in events])
        self.assertGreaterEqual(meta["produced"], 0)
        self.assertGreaterEqual(meta["consumed"], 0)

    def test_pipeline_balances_without_chaos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self._write_policy(tmpdir)
            jsonl_dir = Path(tmpdir) / "episodes"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            meta = run_pipeline(
                policy_path=policy_path,
                jsonl_dir=jsonl_dir,
                max_events=10,
                timeout_s=0.05,
            )
        self.assertEqual(meta["produced"], meta["consumed"])


if __name__ == "__main__":
    unittest.main()
