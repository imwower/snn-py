"""统一的错误类型与退出码映射。"""

from __future__ import annotations

import logging

from snn_py import logging_config


class SnnPyError(Exception):
    """基础异常，携带错误代码。"""

    code = "SNN_ERR"


class PolicyError(SnnPyError):
    code = "POLICY_ERR"


class ConfigError(SnnPyError):
    code = "CONFIG_ERR"


class IOError_(SnnPyError):
    code = "IO_ERR"


_EXIT_CODES = {
    PolicyError: 2,
    ConfigError: 3,
    IOError_: 4,
}


def to_exit_code(exc: BaseException) -> int:
    for cls, code in _EXIT_CODES.items():
        if isinstance(exc, cls):
            return code
    return 1


def log_error(exc: BaseException) -> None:
    logger = logging_config.get_logger("snn_py")
    logger.error(
        "",
        extra={
            "event": "error",
            "meta": {"type": exc.__class__.__name__, "code": getattr(exc, "code", "GENERIC"), "msg": str(exc)},
        },
    )

