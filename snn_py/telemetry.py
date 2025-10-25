"""轻量级运行态资源观测。"""

from __future__ import annotations

import logging
import os
import platform
import threading
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Optional

from snn_py import logging_config

try:  # pragma: no cover - platform specific import
    import resource
except ImportError:  # pragma: no cover - Windows / limited platforms
    resource = None  # type: ignore


def _get_logger() -> logging.Logger:
    return logging_config.get_logger("snn_py.telemetry")


_SYSTEM = platform.system().lower()


def _rss_bytes() -> Optional[float]:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    # Linux reports KB, macOS reports bytes.
    if _SYSTEM == "linux":
        rss *= 1024.0
    return rss


def _cpu_time_s() -> Optional[float]:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _count_fds() -> Optional[int]:
    for path in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(path))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
    return None


@dataclass
class TelemetrySample:
    rss_mb: Optional[float]
    fds: Optional[int]
    py_alloc_kb: Optional[float]
    cpu_time_s: Optional[float]

    def to_meta(self) -> Dict[str, Optional[float]]:
        return {
            "rss_mb": self.rss_mb,
            "fds": self.fds,
            "py_alloc_kb": self.py_alloc_kb,
            "cpu_time_s": self.cpu_time_s,
        }


class Telemetry:
    """周期性记录进程资源使用情况。"""

    def __init__(self, interval_s: float = 5.0, logger: Optional[logging.Logger] = None) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval = interval_s
        self._logger = logger or _get_logger()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._owns_tracemalloc = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._owns_tracemalloc = True
        else:
            self._owns_tracemalloc = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            if self._owns_tracemalloc:
                tracemalloc.stop()
                self._owns_tracemalloc = False
            return
        self._stop.set()
        self._thread.join(timeout=self._interval * 2)
        self._thread = None
        if self._owns_tracemalloc:
            tracemalloc.stop()
            self._owns_tracemalloc = False

    def _run(self) -> None:
        while not self._stop.is_set():
            self._emit_sample()
            if self._stop.wait(self._interval):
                break

    def _emit_sample(self) -> None:
        sample = self._capture()
        self._logger.info("", extra={"event": "telemetry", "meta": sample.to_meta()})

    def _capture(self) -> TelemetrySample:
        rss_bytes = _rss_bytes()
        rss_mb = rss_bytes / (1024.0 * 1024.0) if rss_bytes is not None else None
        fds = _count_fds()
        py_alloc_kb: Optional[float]
        if tracemalloc.is_tracing():
            current, _peak = tracemalloc.get_traced_memory()
            py_alloc_kb = current / 1024.0
        else:
            py_alloc_kb = None
        cpu_time = _cpu_time_s()
        return TelemetrySample(rss_mb=rss_mb, fds=fds, py_alloc_kb=py_alloc_kb, cpu_time_s=cpu_time)


__all__ = ["Telemetry", "TelemetrySample"]
