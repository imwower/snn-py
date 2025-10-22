"""Novelty scoring utilities for intent gating."""

from __future__ import annotations

import statistics
from collections import deque
from typing import Deque


class NoveltyScorer:
    """Computes novelty scores from a sliding window mean."""

    def __init__(self, history_len: int = 32) -> None:
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        self._history: Deque[float] = deque(maxlen=history_len)

    def score(self, x: float) -> float:
        """Return the absolute deviation from the historical mean."""
        if self._history:
            baseline = statistics.fmean(self._history)
            score = abs(x - baseline)
        else:
            baseline = x
            score = 0.0
        self._history.append(x)
        return float(score)
