import unittest

from snn_py import logging_config
from snn_py.core.clusters import ClusterConfig, ClusterWLC
from snn_py.core.events import detect_avalanches, detect_up_down
from snn_py.core.lif import LIF, LIFConfig
from snn_py.intent.gate import GateConfig, IntentGate
from tests.utils import record_payload


class EventCodeTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_config.setup()

    def test_intent_gate_codes(self) -> None:
        cfg = GateConfig(dt=0.01, lam=0.0, alpha=1.0, sigma=0.0, theta=0.5, refractory=0.05, max_rate_hz=10.0)
        gate = IntentGate(cfg, seed=1)
        with self.assertLogs("snn_py.intent.gate", level="INFO") as captured:
            gate._emit("意图触发", 0.2, 0.1)  # type: ignore[attr-defined]
            gate._emit("不应期", 0.0, 0.01)  # type: ignore[attr-defined]
            gate._emit("速率限制", 0.0, 0.01)  # type: ignore[attr-defined]

        codes = {record_payload(record).get("code") for record in captured.records}
        self.assertTrue({"INTENT_FIRED", "REFRACTORY", "RATE_LIMITED"}.issubset(codes))

    def test_events_codes(self) -> None:
        pop_rate = [0.1, 0.15, 0.7, 0.8, 0.15, 0.1]
        spikes = [[0], [1], [1], [0], [0], [1], [1], [0]]
        with self.assertLogs("snn_py.core.events", level="INFO") as captured:
            detect_up_down(pop_rate, thr_low=0.2, thr_high=0.6)
            detect_avalanches(spikes, bin_width=2)

        codes = {record_payload(record).get("code") for record in captured.records}
        self.assertTrue({"UP_START", "DOWN_START", "AVALANCHE"}.issubset(codes))

    def test_lif_codes(self) -> None:
        cfg = LIFConfig(
            n=4,
            frac_inh=0.25,
            p_conn=0.0,
            dt=0.001,
            tau_m=0.02,
            v_rest=0.0,
            v_reset=0.0,
            v_th=1.0,
            w_e=0.1,
            w_i=-0.1,
            refrac_steps=1,
            ext_noise=0.0,
        )
        with self.assertLogs("snn_py.core.lif", level="INFO") as captured:
            lif = LIF(cfg, seed=3)
            for _ in range(50):
                lif.step()

        codes = {record_payload(record).get("code") for record in captured.records}
        self.assertTrue({"LIF_BUILT", "LIF_STEP_SUMMARY"}.issubset(codes))

    def test_cluster_codes(self) -> None:
        cfg = ClusterConfig(n=8, n_clusters=2, frac_inh=0.25, p_intra=0.0, p_inter=0.0, w_e_intra=0.1, w_e_inter=0.1, w_i=-0.2, adapt_tau_steps=10, adapt_inc=0.01, dt=0.001, tau_m=0.02, v_rest=0.0, v_reset=0.0, v_th=1.0, refrac_steps=2, ext_noise=0.0)
        cluster = ClusterWLC(cfg, seed=5)
        members_a = cluster._clusters[0][:2]
        members_b = cluster._clusters[1][:2]
        spike_a = [0] * cfg.n
        spike_b = [0] * cfg.n
        for idx in members_a:
            spike_a[idx] = 1
        for idx in members_b:
            spike_b[idx] = 1
        spikes = [spike_a, spike_b]

        with self.assertLogs("snn_py.core.clusters", level="INFO") as captured:
            cluster.dominant_cluster_series(spikes, win_steps=1)

        codes = {record_payload(record).get("code") for record in captured.records}
        self.assertIn("CLUSTER_DOMINANT_CHANGE", codes)


if __name__ == "__main__":
    unittest.main()
