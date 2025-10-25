"""Population metrics helpers."""

from __future__ import annotations

from typing import List, Optional, Dict
import statistics


def population_rate(spikes: List[List[int]], dt: float, win: int) -> List[float]:
    """
    spikes: T×N 的 0/1 列表; 返回每步群体发放率(Hz)，用长度为 win 的滑窗均值平滑。
    """
    if not spikes:
        return []
    T = len(spikes)
    N = len(spikes[0]) if spikes[0] else 0
    counts = [sum(row) for row in spikes]
    # 简单滑窗均值
    out: List[float] = []
    s = 0
    for i, c in enumerate(counts):
        s += c
        if i >= win:
            s -= counts[i - win]
        size = min(i + 1, win)
        out.append((s / size) / (N * dt) if N > 0 and dt > 0 else 0.0)
    return out


def window_counts(series: List[int], win: int) -> List[int]:
    """将计数序列按窗口分块（非重叠），返回每块的和。"""
    if win <= 0:
        raise ValueError("win must be > 0")
    out: List[int] = []
    for i in range(0, len(series), win):
        out.append(sum(series[i : i + win]))
    return out


def fano_factor(counts: List[int]) -> Optional[float]:
    """Fano 因子 = 方差/均值；均值太小则返回 None。"""
    if not counts:
        return None
    mu = statistics.mean(counts)
    if mu < 1e-8:
        return None
    return statistics.pvariance(counts, mu) / mu


def stability(rate: List[float]) -> Dict[str, float]:
    """返回 {"mean":..,"std":..,"cv":..}；均值为 0 时 cv 记为 0。"""
    if not rate:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}
    m = statistics.fmean(rate)
    sd = statistics.pstdev(rate, m)
    cv = (sd / m) if abs(m) > 1e-8 else 0.0
    return {"mean": m, "std": sd, "cv": cv}


def reliability(events: List[str]) -> Dict[str, int]:
    """统计常见事件次数：intent_fired/audit_decision/denied 等，可按需扩展。"""
    keys = ("intent_fired", "audit_decision", "denied", "pipeline_complete")
    return {k: sum(1 for e in events if k in e) for k in keys}


def run_stability(rate: List[float]) -> Dict[str, float]:
    """保留旧 API，等同于 stability。"""
    return stability(rate)


__all__ = [
    "population_rate",
    "window_counts",
    "fano_factor",
    "stability",
    "reliability",
    "run_stability",
]
