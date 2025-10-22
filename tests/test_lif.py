import json
import logging
import unittest

from snn_py import logging_config
from snn_py.core.lif import LIF, LIFConfig, isi_cv


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)


class LIFNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()
        logging_config.setup()
        self.addCleanup(_reset_logging)

    def _build_cfg(self) -> LIFConfig:
        return LIFConfig(
            n=30,
            frac_inh=0.3,
            p_conn=0.2,
            dt=0.001,
            tau_m=0.02,
            v_rest=-65.0,
            v_reset=-68.0,
            v_th=-50.0,
            w_e=1.2,
            w_i=-1.8,
            refrac_steps=2,
            ext_noise=3.5,
        )

    def test_run_nonzero(self) -> None:
        cfg = self._build_cfg()
        net = LIF(cfg, seed=42)
        spikes = net.run(1.2)
        total_spikes = sum(sum(step) for step in spikes)
        self.assertGreater(total_spikes, 0)

    def test_cv_sanity(self) -> None:
        cfg = self._build_cfg()
        net = LIF(cfg, seed=7)
        spikes = net.run(1.5)
        profiles = []
        for neuron in range(cfg.n):
            train = [step[neuron] for step in spikes]
            cv = isi_cv(train, cfg.dt)
            if cv is not None:
                profiles.append(cv)
        self.assertTrue(profiles)
        bounded = [cv for cv in profiles if 0.4 <= cv <= 1.8]
        self.assertTrue(bounded)

    def test_log_step_summary(self) -> None:
        cfg = self._build_cfg()
        with self.assertLogs("snn_py.core.lif", level="INFO") as captured:
            net = LIF(cfg, seed=1)
            net.run(0.5)
        entries = [json.loads(record.getMessage()) for record in captured.records]
        events = {entry["event"] for entry in entries}
        self.assertIn("lif_build", events)
        self.assertTrue(any(entry["event"] == "lif_step_summary" for entry in entries))
        entries = [json.loads(record.getMessage()) for record in captured.records]
        events = {entry["event"] for entry in entries}
        self.assertIn("lif_build", events)
        self.assertTrue(any(entry["event"] == "lif_step_summary" for entry in entries))
