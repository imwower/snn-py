"""意向门控的自新评分工具。"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Deque


class NoveltyScorer:
    """基于滑动窗口均值计算新颖度得分。"""

    def __init__(self, history_len: int = 32) -> None:
        if history_len <= 0:
            raise ValueError("history_len 必须为正整数")
        self._history: Deque[float] = deque(maxlen=history_len)

    def score(self, x: float) -> float:
        """返回与历史均值的绝对偏差。"""
        if self._history:
            baseline = statistics.fmean(self._history)
            score = abs(x - baseline)
        else:
            baseline = x
            score = 0.0
        self._history.append(x)
        return float(score)
