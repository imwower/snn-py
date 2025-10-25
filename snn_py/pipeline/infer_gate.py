"""Inference helper for gate threshold models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from snn_py.train.threshold_model import ThresholdModel


class GateInfer:
    def __init__(self, model_path: Path):
        self.model = ThresholdModel.load(model_path)

    def batch(self, qs: Iterable[float]) -> List[int]:
        return [self.model.predict(q) for q in qs]


__all__ = ["GateInfer"]
