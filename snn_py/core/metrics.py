"""Metrics utilities for analysing spike trains and rates."""

from __future__ import annotations

import json
import logging
import math
from statistics import fmean, pstdev, pvariance
from typing import Dict, List, Optional

logger = logging.getLogger("snn_py.core.metrics")


def population_rate(spikes: List[List[int]], dt: float, win: int) -> List[float]:
    """Compute population firing rate in Hz using a sliding window.

    Args:
        spikes: Binary spike indicators shaped [T][N].
        dt: Step duration in seconds.
        win: Window size in steps.

    Returns:
        Population firing rate for each valid window.

    Raises:
        ValueError: When input dimensions or parameters are invalid.
    """

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if win <= 0:
        raise ValueError("win must be positive.")
    if not spikes:
        logger.info(
            "%s",
            json.dumps({"event": "metrics_done", "meta": {"len": 0, "win": win}}),
        )
        return []

    neuron_counts = [len(step) for step in spikes]
    if len(set(neuron_counts)) != 1:
        raise ValueError("All spike rows must have the same length.")
    neuron_count = neuron_counts[0]
    if neuron_count <= 0:
        raise ValueError("Spike trains must include at least one neuron.")

    total_steps = len(spikes)
    if win > total_steps:
        logger.info(
            "%s",
            json.dumps({"event": "metrics_done", "meta": {"len": 0, "win": win}}),
        )
        return []

    counts = [sum(step) for step in spikes]
    window_sum = sum(counts[:win])
    scale = neuron_count * win * dt
    rates = [window_sum / scale]

    for idx in range(win, total_steps):
        window_sum += counts[idx] - counts[idx - win]
        rates.append(window_sum / scale)

    logger.info(
        "%s",
        json.dumps({"event": "metrics_done", "meta": {"len": len(rates), "win": win}}),
    )
    return rates


def fano_factor(counts: List[int], win: int) -> Optional[float]:
    """Estimate Fano factor from count data grouped in fixed windows.

    Args:
        counts: Sequence of event counts.
        win: Number of samples aggregated per window.

    Returns:
        The Fano factor (variance / mean) or None when insufficient data.

    Raises:
        ValueError: When win is not positive.
    """

    if win <= 0:
        raise ValueError("win must be positive.")
    if not counts:
        logger.info(
            "%s",
            json.dumps({"event": "metrics_done", "meta": {"len": 0, "win": win}}),
        )
        return None

    windowed = [
        sum(counts[idx : idx + win]) for idx in range(0, len(counts) - win + 1, win)
    ]

    if len(windowed) < 2:
        logger.info(
            "%s",
            json.dumps({"event": "metrics_done", "meta": {"len": 0, "win": win}}),
        )
        return None

    mean_value = fmean(windowed)
    if math.isclose(mean_value, 0.0, abs_tol=1e-12):
        logger.info(
            "%s",
            json.dumps({"event": "metrics_done", "meta": {"len": 0, "win": win}}),
        )
        return None

    variance_value = pvariance(windowed)
    result = variance_value / mean_value
    logger.info(
        "%s",
        json.dumps(
            {"event": "metrics_done", "meta": {"len": len(windowed), "win": win}}
        ),
    )
    return result


def run_stability(rate: List[float]) -> Dict[str, float]:
    """Calculate stability statistics for a rate trace."""

    if not rate:
        logger.info(
            "%s",
            json.dumps({"event": "metrics_done", "meta": {"len": 0, "win": None}}),
        )
        return {"mean": 0.0, "std": 0.0, "cv": math.nan}

    mean_value = fmean(rate)
    std_value = pstdev(rate)
    cv_value = std_value / mean_value if not math.isclose(mean_value, 0.0) else math.inf

    logger.info(
        "%s",
        json.dumps({"event": "metrics_done", "meta": {"len": len(rate), "win": None}}),
    )
    return {"mean": mean_value, "std": std_value, "cv": cv_value}
