"""Cluster-based winnerless competition built atop an LIF network."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from snn_py import logging_config

_LOGGER = logging_config.get_logger("snn_py.core.clusters")


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for clustered LIF winnerless competition."""

    n: int = 150
    n_clusters: int = 4
    frac_inh: float = 0.2
    p_intra: float = 0.15
    p_inter: float = 0.02
    w_e_intra: float = 0.12
    w_e_inter: float = 0.04
    w_i: float = -0.35
    adapt_tau_steps: int = 200
    adapt_inc: float = 0.02
    dt: float = 0.001
    tau_m: float = 0.02
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_th: float = 1.0
    refrac_steps: int = 5
    ext_noise: float = 0.05


class ClusterWLC:
    """Sparse clustered LIF network with adaptation-driven switching."""

    def __init__(self, cfg: ClusterConfig, seed: int = 0) -> None:
        if cfg.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if cfg.n <= 0:
            raise ValueError("n must be positive")
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._logger = _LOGGER
        self._time_step = 0

        self._n_inh = max(1, int(round(cfg.n * cfg.frac_inh)))
        self._n_exc = cfg.n - self._n_inh
        if self._n_exc <= 0:
            raise ValueError("Configuration yields no excitatory neurons.")

        self._adapt_decay = (
            math.exp(-1.0 / cfg.adapt_tau_steps) if cfg.adapt_tau_steps > 0 else 0.0
        )

        self._cluster_of: List[int] = [-1] * cfg.n
        exc_indices = list(range(self._n_exc))
        self._rng.shuffle(exc_indices)
        clusters: List[List[int]] = [[] for _ in range(cfg.n_clusters)]
        for idx, neuron in enumerate(exc_indices):
            cluster_id = idx % cfg.n_clusters
            self._cluster_of[neuron] = cluster_id
            clusters[cluster_id].append(neuron)
        self._clusters = clusters

        self._edges: List[List[Tuple[int, float]]] = [[] for _ in range(cfg.n)]
        self._build_connections()

        self._v = [cfg.v_rest for _ in range(cfg.n)]
        self._refrac = [0 for _ in range(cfg.n)]
        self._adapt = [0.0 for _ in range(cfg.n)]
        self._incoming = [self._rng.random() * cfg.w_e_inter for _ in range(cfg.n)]

    def _build_connections(self) -> None:
        cfg = self.cfg
        for pre in range(cfg.n):
            if pre < self._n_exc:
                pre_cluster = self._cluster_of[pre]
                for post in range(cfg.n):
                    if pre == post:
                        continue
                    if post < self._n_exc:
                        post_cluster = self._cluster_of[post]
                        same_cluster = pre_cluster == post_cluster
                        prob = cfg.p_intra if same_cluster else cfg.p_inter
                        weight = cfg.w_e_intra if same_cluster else cfg.w_e_inter
                    else:
                        prob = cfg.p_inter
                        weight = cfg.w_e_inter
                    if prob > 0 and self._rng.random() < prob:
                        self._edges[pre].append((post, weight))
            else:
                for post in range(cfg.n):
                    if pre == post:
                        continue
                    prob = self.cfg.p_inter if post < self._n_exc else self.cfg.p_inter
                    if prob > 0 and self._rng.random() < prob:
                        self._edges[pre].append((post, self.cfg.w_i))

    def step(self) -> List[int]:
        """Advance the clustered network by one time step."""
        cfg = self.cfg
        spikes = [0 for _ in range(cfg.n)]
        inputs = self._incoming
        next_incoming = [0.0 for _ in range(cfg.n)]

        for idx in range(cfg.n):
            if self._refrac[idx] > 0:
                self._refrac[idx] -= 1
                self._v[idx] = cfg.v_reset
                continue

            if idx < self._n_exc:
                self._adapt[idx] *= self._adapt_decay
                total_input = inputs[idx] - self._adapt[idx]
            else:
                total_input = inputs[idx]

            dv = (-(self._v[idx] - cfg.v_rest) + total_input) * (cfg.dt / cfg.tau_m)
            dv += self._rng.gauss(0.0, cfg.ext_noise)
            self._v[idx] += dv

            if self._v[idx] >= cfg.v_th:
                spikes[idx] = 1
                self._v[idx] = cfg.v_reset
                self._refrac[idx] = cfg.refrac_steps
                for dst, weight in self._edges[idx]:
                    next_incoming[dst] += weight
                if idx < self._n_exc:
                    self._adapt[idx] += cfg.adapt_inc

        self._incoming = next_incoming
        self._time_step += 1
        return spikes

    def run(self, T: float) -> List[List[int]]:
        """Simulate the clustered network for duration T seconds."""
        steps = max(0, int(T / self.cfg.dt))
        return [self.step() for _ in range(steps)]

    def dominant_cluster_series(self, spikes: Sequence[Sequence[int]], win_steps: int) -> List[int]:
        """Compute dominant cluster per window and emit change logs."""
        if win_steps <= 0:
            raise ValueError("win_steps must be positive")
        series: List[int] = []
        prev_cluster: Optional[int] = None
        dt = self.cfg.dt

        for win_start in range(0, len(spikes), win_steps):
            window = spikes[win_start : win_start + win_steps]
            if not window:
                break
            counts = [0 for _ in range(self.cfg.n_clusters)]
            for step in window:
                for cluster_id, members in enumerate(self._clusters):
                    counts[cluster_id] += sum(step[idx] for idx in members)
            if not any(counts):
                continue
            dominant = max(range(self.cfg.n_clusters), key=lambda c: counts[c])
            if prev_cluster != dominant:
                prev_cluster = dominant
                series.append(dominant)
                payload = {
                    "event": "cluster_dominant_change",
                    "meta": {"t": win_start * dt, "cluster": dominant},
                }
                self._logger.info(json.dumps(payload, separators=(",", ":")))
        return series
