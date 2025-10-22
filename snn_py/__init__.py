"""Public interface for the snn_py package."""

from .core import detect_avalanches, detect_up_down
from .intent import GateConfig, IntentGate, NoveltyScorer
from .logging_config import get_logger, setup

__all__ = [
    "setup",
    "get_logger",
    "GateConfig",
    "IntentGate",
    "NoveltyScorer",
    "detect_up_down",
    "detect_avalanches",
]
