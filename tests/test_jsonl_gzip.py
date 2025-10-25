import gzip
import json
import tempfile
import unittest
from pathlib import Path

from snn_py import logging_config
from snn_py.cli import replay
from snn_py.memory.episodic import Episode, EpisodicStore
from tests.utils import record_payload


class JsonlGzipTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_store_rotates_and_replay_reads_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            total = 40
            store = EpisodicStore(
                capacity=16,
                jsonl_dir=dir_path,
                roll_bytes=200,
                compress=True,
            )
            for idx in range(total):
                episode = Episode(
                    t0=float(idx),
                    t1=float(idx) + 0.01,
                    kind="proposal",
                    meta={"idx": idx},
                    payload={"value": idx},
                )
                store.append(episode)
            store.close()

            gz_files = sorted(dir_path.glob("*.jsonl.gz"))
            self.assertGreaterEqual(len(gz_files), 2)

            decoded = 0
            for gz_path in gz_files:
                with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        json.loads(line)
                        decoded += 1
            self.assertEqual(decoded, total)

            with self.assertLogs("snn_py", level="INFO") as captured:
                exit_code = replay.main(["--jsonl-dir", tmpdir])

        self.assertEqual(exit_code, 0)
        events = [record_payload(record) for record in captured.records]
        summary = next(entry for entry in events if entry.get("event") == "replay_summary")
        self.assertEqual(summary["meta"].get("proposals"), total)


if __name__ == "__main__":
    unittest.main()
