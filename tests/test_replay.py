import tempfile
import unittest

from snn_py import logging_config
from snn_py.cli import replay
from snn_py.memory.episodic import Episode, EpisodicStore
from tests.utils import record_payload


class ReplayCLITests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_replay_counts_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EpisodicStore(capacity=64, jsonl_dir=tmpdir)
            for idx in range(30):
                episode = Episode(
                    t0=float(idx),
                    t1=float(idx) + 0.1,
                    kind="proposal",
                    meta={"index": idx},
                    payload={"value": idx},
                )
                store.append(episode)
            store.close()

            with self.assertLogs("snn_py", level="INFO") as captured:
                exit_code = replay.main(["--jsonl-dir", tmpdir])

        self.assertEqual(exit_code, 0)
        events = [record_payload(record) for record in captured.records]
        self.assertTrue(any(entry.get("event") == "replay_loaded" for entry in events))

        summary = next((entry for entry in events if entry.get("event") == "replay_summary"), None)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(30, summary["meta"].get("proposals"))
        self.assertEqual(1, summary["meta"].get("runs"))


if __name__ == "__main__":
    unittest.main()
