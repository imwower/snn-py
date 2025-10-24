import random
import unittest

from snn_py.core.metrics import fano_factor, population_rate, run_stability


class MetricsTests(unittest.TestCase):
    def test_population_rate_shape_and_non_negative(self) -> None:
        rng = random.Random(123)
        neuron_count = 20
        total_steps = 500
        dt = 0.001
        win = 25
        firing_rate_hz = 15.0
        p_spike = firing_rate_hz * dt

        spikes = [
            [1 if rng.random() < p_spike else 0 for _ in range(neuron_count)]
            for _ in range(total_steps)
        ]

        rates = population_rate(spikes, dt=dt, win=win)

        expected_length = total_steps - win + 1
        self.assertEqual(expected_length, len(rates))
        self.assertTrue(all(value >= 0.0 for value in rates))

    def test_fano_factor_distinguishes_variance_levels(self) -> None:
        win = 5
        low_variance_counts = [10, 11, 9, 10, 10, 11, 10, 9, 10, 10]
        high_variance_counts = [2, 18, 4, 22, 1, 19, 3, 25, 2, 20]

        low_factor = fano_factor(low_variance_counts, win=win)
        high_factor = fano_factor(high_variance_counts, win=win)

        self.assertIsNotNone(low_factor)
        self.assertIsNotNone(high_factor)
        assert low_factor is not None
        assert high_factor is not None
        self.assertGreater(high_factor, low_factor * 2)

    def test_run_stability_cv_matches_std_over_mean(self) -> None:
        trace = [1.0, 3.0, 5.0, 7.0]

        stats = run_stability(trace)

        self.assertAlmostEqual(stats["cv"], stats["std"] / stats["mean"], places=9)


if __name__ == "__main__":
    unittest.main()
