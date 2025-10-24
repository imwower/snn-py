import tempfile
import unittest
from pathlib import Path

from snn_py import logging_config
from snn_py.core.clusters import ClusterConfig, ClusterWLC
from snn_py.core.lif import LIF, LIFConfig
from tests.utils import record_payload


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_lif_checkpoint_roundtrip(self) -> None:
        cfg = LIFConfig(
            n=16,
            frac_inh=0.25,
            p_conn=0.2,
            dt=0.001,
            tau_m=0.02,
            v_rest=0.0,
            v_reset=-0.05,
            v_th=0.6,
            w_e=0.08,
            w_i=-0.1,
            refrac_steps=3,
            ext_noise=0.01,
        )
        baseline = LIF(cfg, seed=7)
        baseline_series = [baseline.step() for _ in range(300)]

        model = LIF(cfg, seed=7)
        first_segment = [model.step() for _ in range(200)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lif.json"
            with self.assertLogs("snn_py.core.lif", level="INFO") as saved_logs:
                model.save_json(path)
            continued_segment = [model.step() for _ in range(100)]
            with self.assertLogs("snn_py.core.lif", level="INFO") as loaded_logs:
                restored = LIF.load_json(path)
            restored_segment = [restored.step() for _ in range(100)]

        self.assertEqual(first_segment, baseline_series[:200])
        self.assertEqual(restored_segment, baseline_series[200:300])
        self.assertEqual(restored_segment, continued_segment)

        saved_events = [record_payload(record) for record in saved_logs.records]
        self.assertTrue(
            any(event.get("event") == "checkpoint_saved" for event in saved_events)
        )
        loaded_events = [record_payload(record) for record in loaded_logs.records]
        self.assertTrue(
            any(event.get("event") == "checkpoint_loaded" for event in loaded_events)
        )

    def test_clusterwlc_checkpoint_roundtrip(self) -> None:
        cfg = ClusterConfig(
            n=32,
            n_clusters=2,
            frac_inh=0.25,
            p_intra=0.3,
            p_inter=0.05,
            w_e_intra=0.12,
            w_e_inter=0.04,
            w_i=-0.3,
            adapt_tau_steps=50,
            adapt_inc=0.01,
            dt=0.001,
            tau_m=0.02,
            v_rest=0.0,
            v_reset=-0.04,
            v_th=0.6,
            refrac_steps=4,
            ext_noise=0.02,
        )

        baseline = ClusterWLC(cfg, seed=11)
        baseline_series = [baseline.step() for _ in range(300)]

        model = ClusterWLC(cfg, seed=11)
        first_segment = [model.step() for _ in range(200)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cluster.json"
            with self.assertLogs("snn_py.core.clusters", level="INFO") as saved_logs:
                model.save_json(path)
            continued_segment = [model.step() for _ in range(100)]
            with self.assertLogs("snn_py.core.clusters", level="INFO") as loaded_logs:
                restored = ClusterWLC.load_json(path)
            restored_segment = [restored.step() for _ in range(100)]

        self.assertEqual(first_segment, baseline_series[:200])
        self.assertEqual(restored_segment, baseline_series[200:300])
        self.assertEqual(restored_segment, continued_segment)

        saved_events = [record_payload(record) for record in saved_logs.records]
        self.assertTrue(
            any(event.get("event") == "checkpoint_saved" for event in saved_events)
        )
        loaded_events = [record_payload(record) for record in loaded_logs.records]
        self.assertTrue(
            any(event.get("event") == "checkpoint_loaded" for event in loaded_events)
        )


if __name__ == "__main__":
    unittest.main()
