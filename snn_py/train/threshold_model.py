"""Simple threshold-based gate model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

from snn_py.train.dataset import Sample


@dataclass
class ThresholdModel:
    theta: float
    name: str = "threshold"

    def predict(self, q: float) -> int:
        return 1 if q > self.theta else 0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "ThresholdModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        return ThresholdModel(**data)


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch between y_true and y_pred")
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def grid_search_theta(train: List[Sample], val: List[Sample], grid: List[float]) -> Tuple[ThresholdModel, float]:
    if not grid:
        raise ValueError("grid must not be empty")
    if not val:
        # default to first theta if no validation data
        return ThresholdModel(theta=grid[0]), 0.0

    y_val = [sample.y for sample in val]
    best_theta = grid[0]
    best_f1 = -1.0

    for theta in grid:
        preds = [1 if sample.q > theta else 0 for sample in val]
        score = f1_score(y_val, preds)
        if score > best_f1:
            best_f1 = score
            best_theta = theta

    return ThresholdModel(theta=best_theta), best_f1


__all__ = ["ThresholdModel", "f1_score", "grid_search_theta"]
