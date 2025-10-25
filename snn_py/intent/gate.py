"""基于噪声的意向门控。"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Type

from snn_py import logging_config
from snn_py.event_schema import EVENT_CODES
from snn_py.plugins.registry import load_symbol


@dataclass(frozen=True)
class GateConfig:
    """意向门控的配置参数。"""

    dt: float
    lam: float
    alpha: float
    sigma: float
    theta: float
    refractory: float
    max_rate_hz: float


class IntentGate:
    """依靠内源噪声生成提案的随机意向门控。"""

    def __init__(self, cfg: GateConfig, seed: int = 7) -> None:
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._logger = logging_config.get_logger("snn_py.intent.gate")
        self._state: float = 0.0
        self._time: float = 0.0
        self._last_fire_time: Optional[float] = None
        self._min_interval = 0.0 if cfg.max_rate_hz <= 0 else 1.0 / cfg.max_rate_hz

    def reset(self) -> None:
        """重置内部状态与时间。"""
        self._state = 0.0
        self._time = 0.0
        self._last_fire_time = None

    def step(self, q_t: float = 0.0) -> Tuple[bool, float]:
        """推进门控一步并给出是否触发的决定。"""
        self._time += self.cfg.dt
        drift = -self.cfg.lam * self._state + self.cfg.alpha * q_t
        noise = self.cfg.sigma * math.sqrt(self.cfg.dt) * self._rng.normalvariate(0.0, 1.0)
        self._state += drift * self.cfg.dt + noise

        if self._state < 0.0:
            self._state = 0.0

        since_last = (
            self._time if self._last_fire_time is None else self._time - self._last_fire_time
        )

        if self._state >= self.cfg.theta:
            if self.cfg.refractory > 0 and self._last_fire_time is not None:
                refractory_elapsed = self._time - self._last_fire_time
                if refractory_elapsed < self.cfg.refractory:
                    self._state = min(self._state, self.cfg.theta)
                    self._emit("不应期", q_t, refractory_elapsed)
                    return False, self._state

            if self._min_interval > 0 and self._last_fire_time is not None:
                interval_elapsed = self._time - self._last_fire_time
                if interval_elapsed < self._min_interval:
                    self._state = min(self._state, self.cfg.theta)
                    self._emit("速率限制", q_t, interval_elapsed)
                    return False, self._state

            self._last_fire_time = self._time
            self._state = 0.0
            self._emit("意图触发", q_t, since_last)
            return True, self._state

        return False, self._state

    def _emit(self, event: str, q_t: float, since_last: float) -> None:
        payload = {
            "event": event,
            "ts": time.time(),
            "meta": {"state": float(self._state), "q": float(q_t), "since_last": float(since_last)},
        }
        code = EVENT_CODES.get(event)
        if code is not None:
            payload["code"] = code
        self._logger.info(json.dumps(payload, separators=(",", ":")))


DEFAULT_GATE_SYMBOL = "snn_py.intent.gate:IntentGate"


def resolve_gate(symbol: Optional[str] = None) -> Type[IntentGate]:
    if symbol in (None, "", DEFAULT_GATE_SYMBOL):
        return IntentGate
    obj = load_symbol(symbol)
    if not isinstance(obj, type):
        raise TypeError(f"plugin {symbol} is not a class")
    if not hasattr(obj, "step"):
        raise TypeError(f"plugin {symbol} does not expose 'step'")
    return obj  # type: ignore[return-value]


def create_gate(cfg: GateConfig, seed: int = 7, symbol: Optional[str] = None, **kwargs) -> IntentGate:
    gate_cls = resolve_gate(symbol)
    return gate_cls(cfg, seed=seed, **kwargs)


__all__ = ["GateConfig", "IntentGate", "create_gate", "resolve_gate", "DEFAULT_GATE_SYMBOL"]
