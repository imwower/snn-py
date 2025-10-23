"""系统自检工具。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from snn_py import logging_config
from snn_py.errors import ConfigError, PolicyError, log_error, to_exit_code
from snn_py.policy.auditor import Auditor


def _parse_args(args: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="snn_py doctor")
    parser.add_argument("--policy", required=True, help="策略 JSON 文件路径")
    parser.add_argument("--check-dir", default=".", help="检测读写权限的目录")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args(args)


def _check_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True) as fh:
            fh.write(b"healthcheck")
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"目录不可写: {path}") from exc


def _check_env() -> None:
    value = os.environ.get("SNN_PY_LOGLEVEL")
    if value and value.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError(f"SNN_PY_LOGLEVEL 非法: {value}")


def run_checks(policy_path: Path, check_dir: Path) -> dict:
    Auditor(str(policy_path))
    _check_dir(check_dir)
    _check_env()
    return {"policy": str(policy_path), "dir": str(check_dir.resolve())}


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(args)
    previous = os.environ.get("SNN_PY_LOGLEVEL")
    os.environ["SNN_PY_LOGLEVEL"] = parsed.loglevel
    logging_config.setup()
    logger = logging_config.get_logger("snn_py.cli.doctor")
    try:
        meta = run_checks(Path(parsed.policy), Path(parsed.check_dir))
        logger.info("", extra={"event": "doctor_ok", "meta": meta})
        exit_code = 0
    except (PolicyError, ConfigError) as exc:
        log_error(exc)
        logger.error("", extra={"event": "doctor_fail", "meta": {"type": exc.__class__.__name__}})
        exit_code = to_exit_code(exc)
    except Exception as exc:  # noqa: BLE001
        log_error(exc)
        logger.error("", extra={"event": "doctor_fail", "meta": {"type": exc.__class__.__name__}})
        exit_code = to_exit_code(exc)
    finally:
        if previous is None:
            os.environ.pop("SNN_PY_LOGLEVEL", None)
        else:
            os.environ["SNN_PY_LOGLEVEL"] = previous
        logging_config.setup()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

