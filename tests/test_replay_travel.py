import json
import tempfile
import unittest
from pathlib import Path

from snn_py import logging_config
from snn_py.cli import replay
from tests.utils import record_payload


class ReplayTimeTravelTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_time_travel_and_expect_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_id = "travel_case"
            manifest_dir = base / run_id
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": run_id,
                "ts_start": 0.0,
                "git_sha": None,
                "policy_sha256": None,
                "policy_path": None,
                "argv": ["--T", "2.0"],
                "env_whitelist": {},
                "artifacts": {},
                "seeds": {"segments": 1234, "gate": 5678},
            }
            frames = replay._recompute_frames(manifest)
            self.assertIsNotNone(frames)
            assert frames is not None
            self.assertGreater(len(frames), 0)

            jsonl_path = base / f"{run_id}.part0.jsonl"
            dt = 0.05
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for frame in frames:
                    t = frame * dt
                    payload = {
                        "t0": t,
                        "t1": t,
                        "kind": "proposal",
                        "meta": {"frame": frame},
                        "payload": {"tool": "ProposeAction", "args": {"frame": frame}},
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            manifest_path = manifest_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            expect_path = base / "expected.json"
            with expect_path.open("w", encoding="utf-8") as handle:
                json.dump({"runs": 1, "proposals": len(frames)}, handle)

            with self.assertLogs("snn_py.cli.replay", level="INFO") as captured:
                exit_code = replay.main(
                    [
                        "--jsonl-dir",
                        str(base),
                        "--time-travel",
                        "--expect",
                        str(expect_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        events = [record_payload(record) for record in captured.records]
        recomputed = next(entry for entry in events if entry.get("event") == "replay_recomputed")
        self.assertGreaterEqual(recomputed["meta"]["match_rate"], 0.95)
        diff = next(entry for entry in events if entry.get("event") == "replay_diff")
        self.assertEqual(diff["meta"]["mismatch"], {})
