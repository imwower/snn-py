import unittest

from snn_py.seed import SeedManager


class SeedManagerTests(unittest.TestCase):
    def _intent_series(self, base: int) -> list[int]:
        manager = SeedManager(base)
        gate_rng = manager.rng("gate")
        return [1 if gate_rng.random() > 0.7 else 0 for _ in range(32)]

    def test_named_streams_reproducible(self) -> None:
        seq1 = self._intent_series(1234)
        seq2 = self._intent_series(1234)
        self.assertEqual(seq1, seq2)

    def test_different_seed_changes_sequence(self) -> None:
        seq_a = self._intent_series(4321)
        seq_b = self._intent_series(9876)
        self.assertNotEqual(seq_a, seq_b)
