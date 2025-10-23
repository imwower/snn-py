import logging
import os
import random
import unittest

from snn_py import logging_config
from snn_py.config import AppConfig, apply_seed, from_env
from tests.utils import record_payload


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.NOTSET)


class ConfigSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_logging()
        logging_config.setup()
        self._env_backup = {}

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            os.environ[key] = value
        for key in ["SNN_PY_SEED", "SNN_PY_DT"]:
            if key not in self._env_backup and key in os.environ:
                os.environ.pop(key)
        _reset_logging()

    def _set_env(self, **pairs: str) -> None:
        for key, value in pairs.items():
            if key not in self._env_backup and key in os.environ:
                self._env_backup[key] = os.environ[key]
            os.environ[key] = value

    def test_from_env_with_logging(self) -> None:
        self._set_env(SNN_PY_SEED="123", SNN_PY_DT="0.005")
        with self.assertLogs("snn_py.config", level="INFO") as captured:
            cfg = from_env()
        self.assertEqual(cfg.seed, 123)
        self.assertAlmostEqual(cfg.dt, 0.005)
        payload = record_payload(captured.records[-1])
        self.assertEqual(payload["event"], "config_loaded")
        self.assertEqual(payload["meta"], {"seed": 123})

    def test_apply_seed_deterministic(self) -> None:
        cfg = AppConfig(seed=999)
        state = random.getstate()
        apply_seed(cfg)
        seq1 = [random.random() for _ in range(3)]
        random.setstate(state)
        apply_seed(cfg)
        seq2 = [random.random() for _ in range(3)]
        self.assertEqual(seq1, seq2)

