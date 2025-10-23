"""应用配置与随机种子管理。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional

from . import logging_config


@dataclass(frozen=True)
class AppConfig:
    """基础运行配置。"""

    seed: Optional[int] = None
    dt: float = 0.001


def from_env(env: Mapping[str, str] | MutableMapping[str, str] = os.environ) -> AppConfig:
    """从环境变量构建配置并记录日志。"""

    seed_str = env.get("SNN_PY_SEED")
    seed: Optional[int]
    if seed_str is None:
        seed = None
    else:
        try:
            seed = int(seed_str)
        except ValueError:
            seed = None

    dt_str = env.get("SNN_PY_DT")
    dt = AppConfig.dt  # type: ignore[attr-defined]
    if dt_str is not None:
        try:
            dt = float(dt_str)
        except ValueError:
            dt = AppConfig.dt  # type: ignore[attr-defined]

    cfg = AppConfig(seed=seed, dt=dt)
    logger = logging_config.get_logger("snn_py.config")
    logger.info("", extra={"event": "config_loaded", "meta": {"seed": cfg.seed}})
    return cfg


def apply_seed(cfg: AppConfig) -> None:
    """根据配置设置全局随机种子。"""

    if cfg.seed is not None:
        random.seed(cfg.seed)


__all__ = ["AppConfig", "from_env", "apply_seed"]

