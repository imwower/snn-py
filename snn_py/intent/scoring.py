"""意向门控的自新评分工具。"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Deque, Optional, Type, TypeVar

from snn_py.plugins.registry import load_symbol

ScorerT = TypeVar("ScorerT", bound="NoveltyScorer")

DEFAULT_SCORER_SYMBOL = "snn_py.intent.scoring:NoveltyScorer"


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


def resolve_scorer(symbol: Optional[str] = None) -> Type[ScorerT]:
    """Return the scorer class referenced by the symbol (or default)."""
    if symbol in (None, "", DEFAULT_SCORER_SYMBOL):
        return NoveltyScorer  # type: ignore[return-value]
    obj = load_symbol(symbol)
    if not isinstance(obj, type):
        raise TypeError(f"plugin {symbol} is not a class")
    if not hasattr(obj, "score"):
        raise TypeError(f"plugin {symbol} does not provide a 'score' method")
    return obj  # type: ignore[return-value]


def create_scorer(symbol: Optional[str] = None, **kwargs) -> ScorerT:
    """Instantiate a scorer class based on plugin symbol."""
    cls = resolve_scorer(symbol)
    return cls(**kwargs)


__all__ = ["NoveltyScorer", "create_scorer", "resolve_scorer", "DEFAULT_SCORER_SYMBOL"]
