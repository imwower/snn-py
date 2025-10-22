"""意向子模块，提供门控与评分工具。"""

from .gate import GateConfig, IntentGate
from .scoring import NoveltyScorer

__all__ = ["GateConfig", "IntentGate", "NoveltyScorer"]
