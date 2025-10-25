"""Lightweight text helpers (corpus loading, n-gram models, narration)."""

from __future__ import annotations

from .corpus import load_corpus
from .ngram import NGramModel, NGramConfig
from .narrator import Narrator, NarrationResult

__all__ = ["load_corpus", "NGramModel", "NGramConfig", "Narrator", "NarrationResult"]

