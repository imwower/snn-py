"""Minimal plugin registry for dynamic scorer/gate loading."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from snn_py import logging_config

_LOGGER = logging_config.get_logger("snn_py.plugins.registry")


def load_symbol(spec: str) -> Any:
    """Import and return the object referenced by ``module:attr`` path."""
    if ":" not in spec:
        _LOGGER.error("", extra={"event": "plugin_error", "meta": {"symbol": spec, "reason": "missing_module_separator"}})
        raise ValueError("plugin spec must be in 'module:attr' format")
    module_path, attr_path = spec.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        _LOGGER.error(
            "",
            extra={"event": "plugin_error", "meta": {"symbol": spec, "reason": "module_import_failed", "error": str(exc)}},
        )
        raise

    target: Any = module
    for part in attr_path.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            _LOGGER.error(
                "",
                extra={"event": "plugin_error", "meta": {"symbol": spec, "reason": "attribute_missing", "error": str(exc)}},
            )
            raise

    _LOGGER.info("", extra={"event": "plugin_loaded", "meta": {"symbol": spec}})
    return target


__all__ = ["load_symbol"]
