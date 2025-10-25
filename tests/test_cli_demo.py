import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

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
        seed_events = [entry for entry in events if entry.get("event") == "seed_streams_ready"]
        self.assertTrue(seed_events)
        seed_meta = seed_events[0].get("meta") or {}
        self.assertIn("gate", seed_meta.get("streams", []))
        metrics_index = next(idx for idx, entry in enumerate(events) if entry.get("event") == "metrics_done")
        run_complete_index = next(idx for idx, entry in enumerate(events) if entry.get("event") == "run_complete")
        seed_index = next(idx for idx, entry in enumerate(events) if entry.get("event") == "seed_streams_ready")
        self.assertLess(seed_index, metrics_index)
        self.assertLess(metrics_index, run_complete_index)
        metrics_events = [entry for entry in events if entry.get("event") == "metrics_done"]
        self.assertTrue(metrics_events)
        metrics_meta = metrics_events[0].get("meta") or {}
        self.assertIn("len", metrics_meta)
        self.assertIn("fano", metrics_meta)
        manifest_events = [entry for entry in events if entry.get("event") == "manifest_saved"]
        self.assertTrue(manifest_events)
        manifest_meta = manifest_events[0].get("meta") or {}
        manifest_path = Path(manifest_meta.get("path", ""))
        self.assertTrue(manifest_path.is_file())
        self.assertTrue(manifest_path.parent.is_dir())
        self.assertEqual(manifest_path.parent.parent.name, "episodes")
        manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn("seeds", manifest_doc)
        self.assertIn("gate", manifest_doc["seeds"])
        self.assertIn("segments", manifest_doc["seeds"])

        run_dir = manifest_path.parent
        try:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            episodes_root = run_dir.parent
            if episodes_root.exists() and not any(episodes_root.iterdir()):
                episodes_root.rmdir()
        except OSError:
            pass
