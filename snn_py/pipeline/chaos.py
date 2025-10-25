"""Chaos injection helpers for runner_adv."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class ChaosConfig:
    prob_exception: float = 0.0
    prob_timeout: float = 0.0
    timeout_s: float = 0.02


def maybe_chaos(rng: random.Random, cfg: ChaosConfig, where: str) -> None:
    """按概率在关键点制造异常或短暂停顿。"""
    r = rng.random()
    if r < cfg.prob_exception:
        raise RuntimeError(f"chaos@{where}")
    r2 = rng.random()
    if r2 < cfg.prob_timeout:
        time.sleep(cfg.timeout_s)
