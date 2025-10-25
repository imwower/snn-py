import random
import unittest

from snn_py.core import metrics


def _fake_spikes(T=500, N=20, p=0.01, seed=7):
    rng = random.Random(seed)
    mat = []
    for _ in range(T):
        row = [1 if rng.random() < p else 0 for _ in range(N)]
        mat.append(row)
    return mat


class MetricsTests(unittest.TestCase):
    def test_population_rate_length_and_nonneg(self) -> None:
        spk = _fake_spikes(T=200, N=10, p=0.05)
        rate = metrics.population_rate(spk, dt=1e-3, win=10)
        self.assertEqual(len(rate), 200)
        self.assertTrue(all(r >= 0 for r in rate))

    def test_fano_factor_contrast(self) -> None:
        low = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        high = [0, 3, 0, 5, 0, 2, 0, 6, 0, 4]
        f_low = metrics.fano_factor(metrics.window_counts(low, 2))
        f_high = metrics.fano_factor(metrics.window_counts(high, 2))
        self.assertIsNotNone(f_low)
        self.assertIsNotNone(f_high)
        assert f_high is not None and f_low is not None
        self.assertGreater(f_high, f_low)

    def test_stability_cv_relation(self) -> None:
        rate = [1.0, 2.0, 3.0, 4.0]
        s = metrics.stability(rate)
        self.assertAlmostEqual(abs(s["cv"] - (s["std"] / s["mean"])), 0.0, places=6)
