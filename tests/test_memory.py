import json
import logging
import unittest

from snn_py import logging_config
from snn_py.memory import Episode, EpisodicStore


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)


class EpisodicStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()
        logging_config.setup()
        self.addCleanup(_reset_logging)

    def _episode(self, index: int) -> Episode:
        return Episode(
            t0=float(index),
            t1=float(index) + 0.5,
            kind=f"kind_{index}",
            meta={"idx": index},
            payload={"value": index},
        )

    def test_ring_overwrite(self) -> None:
        store = EpisodicStore(capacity=2)
        store.append(self._episode(1))
        store.append(self._episode(2))
        store.append(self._episode(3))
        recent = store.query_last(10)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].kind, "kind_3")
        self.assertEqual(recent[1].kind, "kind_2")

    def test_append_logs(self) -> None:
        store = EpisodicStore(capacity=3)
        episode = self._episode(5)
        with self.assertLogs("snn_py.memory.episodic", level="INFO") as captured:
            store.append(episode)
        entries = [json.loads(record.getMessage()) for record in captured.records]
        self.assertTrue(any(entry["event"] == "episode_appended" for entry in entries))
