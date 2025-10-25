"""Lightweight metrics helpers for rate/stability/reliability analysis."""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def window_counts(series: List[int], win: int) -> List[int]:
    if win <= 0:
        raise ValueError("win must be positive")
    if win > len(series):
        return []
    counts: List[int] = []
    window_sum = sum(series[:win])
    counts.append(window_sum)
    for idx in range(win, len(series)):
        window_sum += series[idx] - series[idx - win]
        counts.append(window_sum)
    return counts


def population_rate(spikes: List[List[int]], dt: float, win: int) -> List[float]:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if win <= 0:
        raise ValueError("win must be positive")
    if not spikes:
        return []
    length = len(spikes[0])
    if length == 0:
        return []
    total: List[int] = [0] * length
    for train in spikes:
        if len(train) != length:
            raise ValueError("spike trains must have equal length")
        for idx, val in enumerate(train):
            total[idx] += int(val)
    counts = window_counts(total, win) if len(total) >= win else []
    factor = dt * win
    return [count / factor for count in counts]


def fano_factor(counts: List[int]) -> Optional[float]:
    if not counts:
        return None
    n = len(counts)
    mean = sum(counts) / n
    if mean <= 1e-9:
        return None
    variance = sum((value - mean) ** 2 for value in counts) / n
    return variance / mean


def stability(rate: List[float]) -> Dict[str, Optional[float]]:
    if not rate:
        return {"mean": 0.0, "std": 0.0, "cv": None}
    n = len(rate)
    mean = sum(rate) / n
    variance = sum((value - mean) ** 2 for value in rate) / n
    std = math.sqrt(variance)
    cv = std / mean if abs(mean) > 1e-9 else None
    return {"mean": mean, "std": std, "cv": cv}


def reliability(events: List[str]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for event in events:
        stats[event] = stats.get(event, 0) + 1
    return stats


def run_stability(rate: List[float]) -> Dict[str, Optional[float]]:
    return stability(rate)


__all__ = ["population_rate", "window_counts", "fano_factor", "stability", "reliability", "run_stability"]
