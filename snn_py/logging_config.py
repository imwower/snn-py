"""Logging helpers for the snn_py package."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Tuple

_PACKAGE_PREFIX = "snn_py"
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_KNOWN_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


def _determine_level(env_value: str | None) -> Tuple[int, str]:
    candidate = (env_value or "").strip().upper()
    if candidate not in _KNOWN_LEVELS:
        return logging.INFO, "INFO"
    return getattr(logging, candidate), candidate


def _synchronise_package_loggers(level: int) -> None:
    base = logging.getLogger(_PACKAGE_PREFIX)
    base.setLevel(level)
    manager = logging.Logger.manager
    for name, logger in list(manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and (
            name == _PACKAGE_PREFIX or name.startswith(f"{_PACKAGE_PREFIX}.")
        ):
            logger.setLevel(level)


def setup(level_env_var: str = "SNN_PY_LOGLEVEL") -> None:
    """Configure root and package loggers from an environment variable."""
    level, level_name = _determine_level(os.environ.get(level_env_var))

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=_FORMAT)

    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    _synchronise_package_loggers(level)

    payload = {
        "event": "logging_setup",
        "ts": time.time(),
        "meta": {"level": level_name, "env_var": level_env_var},
    }
    message = json.dumps(payload, separators=(",", ":"))
    logging.getLogger(_PACKAGE_PREFIX).log(level, message)


def get_logger(name: str) -> logging.Logger:
    """Return a logger within the snn_py namespace."""
    if not name:
        qualified = _PACKAGE_PREFIX
    elif name.startswith(_PACKAGE_PREFIX):
        qualified = name
    else:
        qualified = f"{_PACKAGE_PREFIX}.{name}"
    return logging.getLogger(qualified)
