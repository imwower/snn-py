import json
import os
import tempfile
import unittest

from pathlib import Path

from snn_py.memory.episodic import Episode, EpisodicStore


class EpisodicStoreTests(unittest.TestCase):
    def test_ring_buffer_overwrite(self) -> None:
        store = EpisodicStore(capacity=3)
        for i in range(5):
            store.append(Episode(t0=i, t1=i + 0.1, kind=f"k{i}", meta={}, payload={"v": i}))
        recent = store.query_last(5)
        self.assertEqual(len(recent), 3)
        self.assertEqual([ep.kind for ep in recent], ["k4", "k3", "k2"])

    def test_jsonl_logging_and_roll(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            store = EpisodicStore(capacity=10, jsonl_dir=dir_path, roll_bytes=1_000, compress=False)
            for i in range(100):
                store.append(Episode(t0=i, t1=i + 0.1, kind="demo", meta={"i": i}, payload={"v": i}))
            store.close()

            files = sorted(dir_path.glob("*.jsonl"))
            self.assertTrue(files)
            entries = []
            for file in files:
                with open(file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        entries.append(json.loads(line))
            self.assertEqual(len(entries), 100)
            self.assertEqual(entries[0]["meta"]["i"], 0)
            self.assertEqual(entries[-1]["meta"]["i"], 99)
