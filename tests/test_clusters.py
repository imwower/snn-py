import json
import logging
import unittest

from snn_py import logging_config
from snn_py.core import ClusterConfig, ClusterWLC


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)


class ClusterWLCTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()
        logging_config.setup()
        self.addCleanup(_reset_logging)

    def test_sequence_nontrivial(self) -> None:
        cfg = ClusterConfig(
            n=120,
            n_clusters=4,
            frac_inh=0.25,
            p_intra=0.35,
            p_inter=0.06,
            w_e_intra=1.6,
            w_e_inter=0.45,
            w_i=-1.4,
            adapt_tau_steps=120,
            adapt_inc=0.5,
            dt=0.001,
            tau_m=0.02,
            v_rest=0.0,
            v_reset=0.0,
            v_th=1.0,
            refrac_steps=4,
            ext_noise=0.35,
        )
        net = ClusterWLC(cfg, seed=23)
        spikes = net.run(2.8)
        with self.assertLogs("snn_py.core.clusters", level="INFO") as captured:
            series = net.dominant_cluster_series(spikes, win_steps=60)
        self.assertGreaterEqual(len(series), 3)
        self.assertGreaterEqual(len(set(series)), 2)
        entries = [json.loads(record.getMessage()) for record in captured.records]
        events = [entry["event"] for entry in entries]
        self.assertIn("主导簇变更", events)
