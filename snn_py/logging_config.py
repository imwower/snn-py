"""snn_py 软件包的 JSON 行日志配置。"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Tuple

from .run_context import build_run_context

_PACKAGE_PREFIX = "snn_py"
_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}

_RUN_CONTEXT: Optional[Dict[str, object]] = None


class JSONLineFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON。"""

    def __init__(self, run: Dict[str, object]) -> None:
        super().__init__()
        self._run = run

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        event = getattr(record, "event", None)
        meta = getattr(record, "meta", None)
        code = getattr(record, "code", None)

        message = record.getMessage()
        msg_value = message

        if event is None and message:
            try:
                payload = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                payload = None
            if isinstance(payload, dict):
                event = payload.get("event", event)
                payload_meta = payload.get("meta")
                if meta is None and isinstance(payload_meta, dict):
                    meta = payload_meta
                payload_code = payload.get("code")
                if payload_code is not None:
                    code = payload_code
                msg_payload = payload.get("msg")
                if msg_payload is not None:
                    msg_value = msg_payload

        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            meta = {"value": meta}

        run_data = getattr(record, "run", None) or self._run

        output = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "event": event or "log",
            "msg": msg_value,
            "meta": meta,
            "run": run_data,
        }
        if code is not None:
            output["code"] = code
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


class ContextAdapter(logging.LoggerAdapter):
    """在日志记录中注入运行上下文。"""

    def __init__(self, logger: logging.Logger, run: Dict[str, object]) -> None:
        super().__init__(logger, {})
        self._run = run

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("run", self._run)
        return msg, kwargs


def _determine_level(env_value: Optional[str]) -> Tuple[int, str]:
    candidate = (env_value or "").strip().upper()
    if candidate in _LEVELS:
        return _LEVELS[candidate], candidate
    return logging.INFO, "INFO"


def _ensure_run_context() -> Dict[str, object]:
    global _RUN_CONTEXT
    if _RUN_CONTEXT is None:
        _RUN_CONTEXT = build_run_context()
    return _RUN_CONTEXT


def _ensure_formatter(handler: logging.Handler, run: Dict[str, object], level: int) -> None:
    formatter = handler.formatter
    if not isinstance(formatter, JSONLineFormatter):
        formatter = JSONLineFormatter(run)
        handler.setFormatter(formatter)
    handler.setLevel(level)


def setup(level_env_var: str = "SNN_PY_LOGLEVEL") -> None:
    """配置根记录器，启用 JSON 行格式。"""

    run = _ensure_run_context()
    level, level_name = _determine_level(os.environ.get(level_env_var))

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONLineFormatter(run))
        handler.setLevel(level)
        root_logger.addHandler(handler)

    for handler in root_logger.handlers:
        _ensure_formatter(handler, run, level)

    root_logger.setLevel(level)

    logger = get_logger("")
    logger.info("", extra={"event": "logging_setup", "meta": {"level": level_name}})


def get_logger(name: str) -> logging.Logger:
    """返回注入运行上下文的记录器。"""

    run = _ensure_run_context()
    if not name:
        qualified = _PACKAGE_PREFIX
    elif name.startswith(_PACKAGE_PREFIX):
        qualified = name
    else:
        qualified = f"{_PACKAGE_PREFIX}.{name}"

    logger = logging.getLogger(qualified)
    return ContextAdapter(logger, run)


__all__ = ["JSONLineFormatter", "ContextAdapter", "setup", "get_logger"]
