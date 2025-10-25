import random
import unittest

from snn_py.core import metrics


class MetricsTests(unittest.TestCase):
    def test_population_rate_poisson_spikes(self) -> None:
        rng = random.Random(42)
        steps = 64
        neurons = 3
        dt = 0.01
        spikes = []
        for _ in range(neurons):
            train = [1 if rng.random() < 0.2 else 0 for _ in range(steps)]
            spikes.append(train)
        win = 5
        rates = metrics.population_rate(spikes, dt=dt, win=win)
        self.assertEqual(len(rates), steps - win + 1)
        self.assertTrue(all(rate >= 0.0 for rate in rates))

    def test_fano_factor_distinguishes_variance(self) -> None:
        low_series = [10] * 40
        rng = random.Random(7)
        high_series = [rng.randint(0, 25) for _ in range(40)]
        win = 4
        low_counts = metrics.window_counts(low_series, win)
        high_counts = metrics.window_counts(high_series, win)
        low_fano = metrics.fano_factor(low_counts)
        high_fano = metrics.fano_factor(high_counts)
        self.assertIsNotNone(low_fano)
        self.assertIsNotNone(high_fano)
        assert low_fano is not None and high_fano is not None
        self.assertGreater(high_fano, low_fano)

    def test_stability_cv_matches_std_div_mean(self) -> None:
        rate = [1.0, 1.5, 2.0, 2.5, 3.0]
        stats = metrics.stability(rate)
        self.assertAlmostEqual(stats["mean"], sum(rate) / len(rate))
        self.assertAlmostEqual(stats["std"] ** 2, sum((x - stats["mean"]) ** 2 for x in rate) / len(rate))
        expected_cv = stats["std"] / stats["mean"]
        self.assertAlmostEqual(stats["cv"], expected_cv)

    def test_reliability_counts_events(self) -> None:
        events = ["intent_fired", "intent_fired", "audit_decision", "denied", "denied"]
        counts = metrics.reliability(events)
        self.assertEqual(counts["intent_fired"], 2)
        self.assertEqual(counts["denied"], 2)
        self.assertEqual(counts["audit_decision"], 1)
