"""Minimal leaky integrate-and-fire network simulator."""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

from snn_py import logging_config

_LOGGER = logging_config.get_logger("snn_py.core.lif")


@dataclass(frozen=True)
class LIFConfig:
    """Configuration for a simple LIF network."""

    n: int
    frac_inh: float
    p_conn: float
    dt: float
    tau_m: float
    v_rest: float
    v_reset: float
    v_th: float
    w_e: float
    w_i: float
    refrac_steps: int
    ext_noise: float


class LIF:
    """Sparse, delayed LIF network with refractory handling."""

    def __init__(self, cfg: LIFConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._dt_over_tau = cfg.dt / cfg.tau_m if cfg.tau_m > 0 else 0.0
        self._adj: List[List[int]] = [[] for _ in range(cfg.n)]
        for pre in range(cfg.n):
            for post in range(cfg.n):
                if pre == post:
                    continue
                if self._rng.random() < cfg.p_conn:
                    self._adj[pre].append(post)
        inh_cutoff = int(cfg.n * cfg.frac_inh)
        self._is_inhibitory = [idx < inh_cutoff for idx in range(cfg.n)]
        self._v = [cfg.v_rest for _ in range(cfg.n)]
        self._refrac = [0 for _ in range(cfg.n)]
        self._pending: List[List[int]] = [[] for _ in range(cfg.n)]
        self._time_step = 0
        _LOGGER.info(
            json.dumps(
                {
                    "event": "lif_build",
                    "meta": {
                        "n": cfg.n,
                        "p_conn": cfg.p_conn,
                        "frac_inh": cfg.frac_inh,
                    },
                },
                separators=(",", ":"),
            )
        )

    def step(self) -> List[int]:
        """Advance the network dynamics by one time step."""
        spikes = [0 for _ in range(self.cfg.n)]
        inputs = [0.0 for _ in range(self.cfg.n)]

        for src, targets in enumerate(self._adj):
            if self._pending[src]:
                for dst in self._pending[src]:
                    weight = self.cfg.w_i if self._is_inhibitory[src] else self.cfg.w_e
                    inputs[dst] += weight
                self._pending[src].clear()

        for idx in range(self.cfg.n):
            if self._refrac[idx] > 0:
                self._refrac[idx] -= 1
                self._v[idx] = self.cfg.v_reset
                continue

            dv = (- (self._v[idx] - self.cfg.v_rest) + inputs[idx]) * self._dt_over_tau
            dv += self._rng.gauss(0.0, self.cfg.ext_noise)
            self._v[idx] += dv

            if self._v[idx] >= self.cfg.v_th:
                spikes[idx] = 1
                self._v[idx] = self.cfg.v_reset
                self._refrac[idx] = self.cfg.refrac_steps
                self._pending[idx] = self._adj[idx][:]

        self._time_step += 1
        if self._time_step % 50 == 0:
            _LOGGER.info(
                json.dumps(
                    {
                        "event": "lif_step_summary",
                        "meta": {"t": self._time_step * self.cfg.dt, "spike_count": sum(spikes)},
                    },
                    separators=(",", ":"),
                )
            )
        return spikes

    def run(self, T: float) -> List[List[int]]:
        """Simulate the network for a duration T seconds."""
        steps = int(T / self.cfg.dt)
        return [self.step() for _ in range(steps)]


def isi_cv(train: Sequence[int], dt: float) -> Optional[float]:
    """Compute coefficient of variation of inter-spike intervals."""
    spikes = [idx for idx, value in enumerate(train) if value]
    if len(spikes) < 3:
        return None
    intervals = [b - a for a, b in zip(spikes, spikes[1:])]
    mean_isi = statistics.mean(intervals)
    if mean_isi == 0:
        return None
    std_isi = statistics.pstdev(intervals)
    return std_isi / mean_isi
