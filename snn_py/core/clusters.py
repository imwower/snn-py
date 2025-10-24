"""基于 LIF 网络的簇结构胜者非稳（WLC）模型。"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from snn_py import logging_config

_LOGGER = logging_config.get_logger("snn_py.core.clusters")


def _as_rng_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_rng_state(item) for item in value)
    return value


@dataclass(frozen=True)
class ClusterConfig:
    """簇化 LIF 胜者非稳网络的配置。"""

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
    """依靠适应机制驱动簇间轮换的稀疏 LIF 网络。"""

    def __init__(self, cfg: ClusterConfig, seed: int = 0) -> None:
        if cfg.n_clusters <= 0:
            raise ValueError("簇数量必须为正整数")
        if cfg.n <= 0:
            raise ValueError("神经元数量必须为正整数")
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._logger = _LOGGER
        self._time_step = 0

        self._n_inh = max(1, int(round(cfg.n * cfg.frac_inh)))
        self._n_exc = cfg.n - self._n_inh
        if self._n_exc <= 0:
            raise ValueError("当前配置导致无兴奋性神经元")

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
        """推进簇网络一时步并返回脉冲。"""
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
        """模拟簇网络 T 秒并返回脉冲序列。"""
        steps = max(0, int(T / self.cfg.dt))
        return [self.step() for _ in range(steps)]

    def to_dict(self) -> Dict[str, Any]:
        """序列化配置与内部状态。"""
        return {
            "config": asdict(self.cfg),
            "state": {
                "time_step": self._time_step,
                "cluster_of": self._cluster_of[:],
                "clusters": [cluster[:] for cluster in self._clusters],
                "edges": [
                    [[dst, float(weight)] for dst, weight in row] for row in self._edges
                ],
                "v": self._v[:],
                "refrac": self._refrac[:],
                "adapt": self._adapt[:],
                "incoming": self._incoming[:],
                "rng_state": self._rng.getstate(),
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ClusterWLC":
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        cfg_data = payload.get("config")
        state = payload.get("state")
        if not isinstance(cfg_data, dict) or not isinstance(state, dict):
            raise ValueError("payload must include config/state dictionaries")
        cfg = ClusterConfig(**cfg_data)
        instance: "ClusterWLC" = object.__new__(cls)  # type: ignore[call-arg]
        instance.cfg = cfg
        instance._rng = random.Random()
        rng_state = state.get("rng_state")
        if rng_state is not None:
            instance._rng.setstate(_as_rng_state(rng_state))  # type: ignore[arg-type]
        instance._logger = _LOGGER
        instance._time_step = int(state.get("time_step", 0))

        cluster_of = state.get("cluster_of")
        clusters = state.get("clusters")
        edges = state.get("edges")
        v = state.get("v")
        refrac = state.get("refrac")
        adapt = state.get("adapt")
        incoming = state.get("incoming")

        n = cfg.n
        if not isinstance(cluster_of, list) or len(cluster_of) != n:
            raise ValueError("cluster_of must be a list with length equal to cfg.n")
        if not isinstance(clusters, list):
            raise ValueError("clusters must be a list")
        if not isinstance(edges, list) or len(edges) != n:
            raise ValueError("edges must be a list with length equal to cfg.n")
        if not isinstance(v, list) or len(v) != n:
            raise ValueError("v must be a list with length equal to cfg.n")
        if not isinstance(refrac, list) or len(refrac) != n:
            raise ValueError("refrac must be a list with length equal to cfg.n")
        if not isinstance(adapt, list) or len(adapt) != n:
            raise ValueError("adapt must be a list with length equal to cfg.n")
        if not isinstance(incoming, list) or len(incoming) != n:
            raise ValueError("incoming must be a list with length equal to cfg.n")

        instance._cluster_of = [int(value) for value in cluster_of]
        instance._clusters = [[int(neuron) for neuron in row] for row in clusters]
        instance._edges = [
            [(int(dst), float(weight)) for dst, weight in row] for row in edges
        ]

        instance._n_inh = max(1, int(round(cfg.n * cfg.frac_inh)))
        instance._n_exc = cfg.n - instance._n_inh
        if instance._n_exc <= 0:
            raise ValueError("configuration results in zero excitatory neurons")

        instance._adapt_decay = (
            math.exp(-1.0 / cfg.adapt_tau_steps) if cfg.adapt_tau_steps > 0 else 0.0
        )

        instance._v = [float(value) for value in v]
        instance._refrac = [int(value) for value in refrac]
        instance._adapt = [float(value) for value in adapt]
        instance._incoming = [float(value) for value in incoming]
        return instance

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._logger.info(
            "",
            extra={"event": "checkpoint_saved", "meta": {"path": str(target)}},
        )

    @classmethod
    def load_json(cls, path: Path | str) -> "ClusterWLC":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        instance = cls.from_dict(payload)
        instance._logger.info(
            "",
            extra={"event": "checkpoint_loaded", "meta": {"path": str(source)}},
        )
        return instance

    def dominant_cluster_series(self, spikes: Sequence[Sequence[int]], win_steps: int) -> List[int]:
        """计算每个窗口的主导簇并输出变更日志。"""
        if win_steps <= 0:
            raise ValueError("窗口长度必须为正整数")
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
                    "event": "主导簇变更",
                    "code": "CLUSTER_DOMINANT_CHANGE",
                    "meta": {"t": win_start * dt, "cluster": dominant},
                }
                self._logger.info(json.dumps(payload, separators=(",", ":")))
        return series
