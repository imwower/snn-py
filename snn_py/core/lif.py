"""最小化的泄漏积分发放（LIF）网络模拟器。"""

from __future__ import annotations

import json
import logging
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from snn_py import logging_config


def _get_logger() -> logging.Logger:
    return logging_config.get_logger("snn_py.core.lif")


def _as_rng_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_rng_state(item) for item in value)
    return value


@dataclass(frozen=True)
class LIFConfig:
    """简单 LIF 网络的配置。"""

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
    """带有稀疏连通与延迟的不应期 LIF 网络。"""

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
        logger = _get_logger()
        logger.info(
            json.dumps(
                {
                    "event": "LIF 构建完成",
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
        """推进网络一时步并返回当前脉冲。"""
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
            logger = _get_logger()
            logger.info(
                json.dumps(
                    {
                        "event": "LIF 步进摘要",
                        "meta": {"t": self._time_step * self.cfg.dt, "spike_count": sum(spikes)},
                    },
                    separators=(",", ":"),
                )
            )
        return spikes

    def run(self, T: float) -> List[List[int]]:
        """模拟网络 T 秒并返回脉冲矩阵。"""
        steps = int(T / self.cfg.dt)
        return [self.step() for _ in range(steps)]

    def to_dict(self) -> Dict[str, Any]:
        """序列化网络配置与状态。"""
        return {
            "config": asdict(self.cfg),
            "state": {
                "adj": [row[:] for row in self._adj],
                "is_inhibitory": self._is_inhibitory[:],
                "v": self._v[:],
                "refrac": self._refrac[:],
                "pending": [row[:] for row in self._pending],
                "time_step": self._time_step,
                "rng_state": self._rng.getstate(),
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LIF":
        """根据字典恢复网络。"""
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        cfg_data = payload.get("config")
        state = payload.get("state")
        if not isinstance(cfg_data, dict) or not isinstance(state, dict):
            raise ValueError("payload must include config/state dictionaries")
        cfg = LIFConfig(**cfg_data)
        instance: "LIF" = object.__new__(cls)  # type: ignore[call-arg]
        instance.cfg = cfg
        instance._rng = random.Random()
        rng_state = state.get("rng_state")
        if rng_state is not None:
            instance._rng.setstate(_as_rng_state(rng_state))  # type: ignore[arg-type]
        instance._dt_over_tau = cfg.dt / cfg.tau_m if cfg.tau_m > 0 else 0.0

        adj_raw = state.get("adj")
        is_inh_raw = state.get("is_inhibitory")
        v_raw = state.get("v")
        refrac_raw = state.get("refrac")
        pending_raw = state.get("pending")
        time_step = state.get("time_step")

        if not isinstance(adj_raw, list) or len(adj_raw) != cfg.n:
            raise ValueError("adj must be a list with length equal to cfg.n")
        if not isinstance(is_inh_raw, list) or len(is_inh_raw) != cfg.n:
            raise ValueError("is_inhibitory must be a list with length equal to cfg.n")
        if not isinstance(v_raw, list) or len(v_raw) != cfg.n:
            raise ValueError("v must be a list with length equal to cfg.n")
        if not isinstance(refrac_raw, list) or len(refrac_raw) != cfg.n:
            raise ValueError("refrac must be a list with length equal to cfg.n")
        if not isinstance(pending_raw, list) or len(pending_raw) != cfg.n:
            raise ValueError("pending must be a list with length equal to cfg.n")
        if not isinstance(time_step, int):
            raise ValueError("time_step must be an int")

        instance._adj = [[int(dst) for dst in row] for row in adj_raw]
        instance._is_inhibitory = [bool(x) for x in is_inh_raw]
        instance._v = [float(v) for v in v_raw]
        instance._refrac = [int(r) for r in refrac_raw]
        instance._pending = [[int(dst) for dst in row] for row in pending_raw]
        instance._time_step = time_step
        return instance

    def save_json(self, path: Path | str) -> None:
        """保存状态至 JSON 文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger = _get_logger()
        logger.info(
            "",
            extra={"event": "checkpoint_saved", "meta": {"path": str(target)}},
        )

    @classmethod
    def load_json(cls, path: Path | str) -> "LIF":
        """从 JSON 文件中恢复网络。"""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        instance = cls.from_dict(payload)
        logger = _get_logger()
        logger.info(
            "",
            extra={"event": "checkpoint_loaded", "meta": {"path": str(source)}},
        )
        return instance


def isi_cv(train: Sequence[int], dt: float) -> Optional[float]:
    """计算脉冲间隔的变异系数。"""
    spikes = [idx for idx, value in enumerate(train) if value]
    if len(spikes) < 3:
        return None
    intervals = [b - a for a, b in zip(spikes, spikes[1:])]
    mean_isi = statistics.mean(intervals)
    if mean_isi == 0:
        return None
    std_isi = statistics.pstdev(intervals)
    return std_isi / mean_isi
