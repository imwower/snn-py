"""Narrator wrapper around the n-gram model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Optional, Sequence

from .ngram import NGramModel


@dataclass(slots=True)
class NarrationResult:
    text: str
    explain: str


class Narrator:
    def __init__(self, model_path: Path | str, seed: int = 7):
        self._model_path = Path(model_path)
        self._model = NGramModel.load(self._model_path)
        self._rng = Random(seed)

    def narrate(self, *, q: float = 0.5, context: Optional[Sequence[str]] = None) -> NarrationResult:
        """Produce narrated text conditioned on proposal confidence `q`."""
        q_norm = max(0.0, min(1.0, float(q)))
        context_tokens = [token for token in (context or []) if token]
        base_length = 6 + int(round(q_norm * 10))
        generated = self._model.generate(
            max_tokens=max(1, base_length),
            rng=self._rng,
            prefix=context_tokens[-self._model.context_size :],
        )
        tokens = context_tokens + generated
        if not tokens:
            tokens = ["silence"]
        text = " ".join(tokens)
        explain = f"q={q_norm:.2f} drove {len(generated)} generated tokens"
        return NarrationResult(text=text, explain=explain)


__all__ = ["Narrator", "NarrationResult"]

